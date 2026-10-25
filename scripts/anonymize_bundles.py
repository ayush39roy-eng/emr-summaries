"""Phase 2: data/bundles/raw -> data/bundles/anon. Anonymizes text, validates every bundle, prints counts.

Resumable: bundles already in data/bundles/anon are skipped.
"""
import argparse
from collections import Counter

from tqdm import tqdm

from emr.anonymize import anonymize_bundle
from emr.common import bundle_errors, read_json, write_json
from emr.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    paths = load_config()["paths"]
    raw_files = sorted((ROOT / paths["bundles_raw"]).glob("*.json"))[:args.limit]
    out_dir = ROOT / paths["bundles"]

    counts, invalid, done = Counter(), [], 0
    for raw_path in tqdm(raw_files, desc="anonymize"):
        out_path = out_dir / raw_path.name
        if out_path.exists():
            continue
        bundle, found = anonymize_bundle(read_json(raw_path))
        counts.update(found)
        if errors := bundle_errors(bundle):
            invalid.append((raw_path.name, errors[:3]))
            continue
        write_json(out_path, bundle)
        done += 1

    total = len(list(out_dir.glob("*.json")))
    print(f"\nAnonymized {done} new bundles this run; {total} valid anonymized bundles on disk; {len(invalid)} invalid")
    for name, errors in invalid[:10]:
        print("  INVALID", name, errors)
    print("Replacements this run by entity type:")
    for entity, n in counts.most_common():
        print(f"  {n:6d}  {entity}")


if __name__ == "__main__":
    main()
