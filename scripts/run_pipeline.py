"""One command for the whole pipeline on the fixed test set.

    python scripts/run_pipeline.py --model gemini-flash --limit 50
    python scripts/run_pipeline.py --model local-gemma            # same chunks, same prompt, different model
    python scripts/run_pipeline.py --model gemini-flash --dry-run # build prompts only, no API calls

Steps (each writes to disk, so reruns resume and cost nothing):
  1. test set     data/test_set.json (created once, never changed afterwards)
  2. index        chunk + embed + store test-set records in ChromaDB
  3. retrieve     per-section retrieval, cached per record in data/cache/context/ (shared by every model)
  4. summarize    model -> grounding -> full-schema validation -> results/<model>/<record>.json
"""
import argparse
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from emr import store
from emr.common import path_for, read_json, summary_errors, write_json
from emr.config import ROOT, load_config, model_spec
from emr.llm.base import make_llm
from emr.retrieve import index_bundle, retrieve_for_record
from emr.summarize import build_prompt, summarize


def make_test_set(config):
    """50 records, stratified across departments, transcription >= 1500 chars, fixed seed."""
    paths, ds = config["paths"], config["dataset"]
    test_path = ROOT / paths["test_set"]
    if test_path.exists():
        return read_json(test_path)
    by_dept = defaultdict(list)
    for f in sorted((ROOT / paths["bundles"]).glob("*.json")):
        b = read_json(f)
        if len(b["documents"][0]["text"]) >= ds["test_set_min_chars"]:
            by_dept[b["encounter"]["department"]].append(b["record_id"])
    rng = random.Random(ds["seed"])
    for ids in by_dept.values():
        rng.shuffle(ids)
    # round-robin over departments (random order) so every specialty is represented before any repeats
    departments = sorted(by_dept)
    rng.shuffle(departments)
    chosen = []
    while len(chosen) < ds["test_set_size"]:
        for dept in departments:
            if by_dept[dept] and len(chosen) < ds["test_set_size"]:
                chosen.append(by_dept[dept].pop())
    write_json(test_path, chosen)
    print(f"Created test set: {len(chosen)} records from {len(departments)} departments -> {paths['test_set']}")
    return chosen


def context_dir(config, context):
    return f"{config['paths']['cache']}/context" + ("" if context == "full" else f"_{context}")


def load_contexts(config, record_ids, dry_run, context="full", k=2):
    """Index + retrieve once; every model then gets exactly these chunks."""
    paths = config["paths"]
    contexts, todo = {}, []
    for rid in record_ids:
        p = path_for(context_dir(config, context), rid)
        if p.exists():
            contexts[rid] = read_json(p)
        else:
            todo.append(rid)
    if todo:
        if dry_run:
            raise SystemExit(f"{len(todo)} records not yet indexed/retrieved; run once without --dry-run first.")
        embedder = make_llm(config["embedding"])
        col = store.collection()
        n_chunks = 0
        for rid in todo:
            n_chunks += index_bundle(col, embedder, read_json(path_for(paths["bundles"], rid)), config["chunking"])
        print(f"Indexed {len(todo)} records ({n_chunks} new chunks) into ChromaDB")
        for rid in todo:
            if context == "retrieved":
                contexts[rid] = retrieve_for_record(col, embedder, rid, k=k, force_retrieved=True)
            else:
                contexts[rid] = retrieve_for_record(col, embedder, rid)
            write_json(path_for(context_dir(config, context), rid), contexts[rid])
    return contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="short name from config.yaml, e.g. gemini-flash")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="no model calls; just show what would be sent")
    parser.add_argument("--redo", action="store_true", help="regenerate even if results exist")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests (API models only)")
    parser.add_argument("--context", choices=["full", "retrieved"], default="full",
                        help="full = whole record when it fits (default); retrieved = ONLY the top-k chunks per "
                             "summary section (RAG experiment); results go to results/<model>__retrieved/")
    parser.add_argument("--k", type=int, default=2, help="chunks per summary section in --context retrieved")
    args = parser.parse_args()

    config = load_config()
    spec = model_spec(config, args.model)
    record_ids = make_test_set(config)[:args.limit]
    contexts = load_contexts(config, record_ids, args.dry_run, args.context, args.k)

    run_name = args.model + ("" if args.context == "full" else f"__{args.context}")
    out_dir = ROOT / config["paths"]["results"] / run_name
    todo = [r for r in record_ids if args.redo or not path_for(out_dir.relative_to(ROOT), r).exists()]
    print(f"{run_name} ({spec['model_id']}): {len(record_ids) - len(todo)} done, {len(todo)} to do", flush=True)
    if args.dry_run:
        prompt = build_prompt(contexts[todo[0]]) if todo else ""
        print(f"Dry run: would make {len(todo)} calls. Example prompt ({len(prompt)} chars):\n{prompt[:1500]}")
        return

    llm = make_llm(spec, config["generation"]["temperature"])
    min_interval = 60.0 / spec["rpm"] if spec.get("rpm") else 0
    rate_lock, last_call, done = threading.Lock(), [0.0], [0]

    def run_one(rid):
        with rate_lock:  # space out request starts to respect the requests-per-minute limit
            wait = min_interval - (time.time() - last_call[0])
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.time()
        bundle = read_json(path_for(config["paths"]["bundles"], rid))
        summary, report = summarize(llm, bundle, contexts[rid])
        if summary is not None:
            report["schema_errors"] = summary_errors(bundle, summary)
            write_json(path_for(out_dir.relative_to(ROOT), rid), summary)
        # a failed record leaves no summary file, so a rerun retries it
        write_json(path_for((out_dir / "_reports").relative_to(ROOT), rid), report)
        g = report.get("grounding") or {}
        status = report["error"] or (f"{g['supported_claims']}/{g['claims']} claims grounded, "
                                     f"{len(report['schema_errors'])} schema errors")
        with rate_lock:
            done[0] += 1
            print(f"[{done[0]}/{len(todo)}] {rid}: {report['latency_s']}s "
                  f"{'(cached) ' if report['cached'] else ''}{status}", flush=True)

    workers = args.workers if spec["provider"] != "ollama" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, todo))


if __name__ == "__main__":
    main()
