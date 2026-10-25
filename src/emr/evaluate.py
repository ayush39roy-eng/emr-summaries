"""Metrics per model over the fixed test set. Missing / failed outputs count against the model."""
import statistics

from emr.common import path_for, read_json
from emr.config import ROOT


def _rouge_l(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]


def _bertscore(predictions, references):
    # distilbert keeps the download small (~250 MB); scores are comparable across models, not to papers.
    from bert_score import score
    _, _, f1 = score(predictions, references, model_type="distilbert-base-uncased", lang="en", verbose=False)
    return f1.tolist()


def load_model_outputs(model_name, record_ids, results_dir):
    rows = []
    for rid in record_ids:
        summary_path = path_for((results_dir / model_name).relative_to(ROOT), rid)
        report_path = path_for((results_dir / model_name / "_reports").relative_to(ROOT), rid)
        rows.append({
            "record_id": rid,
            "summary": read_json(summary_path) if summary_path.exists() else None,
            "report": read_json(report_path) if report_path.exists() else None,
        })
    return rows


def model_metrics(model_name, model_id, record_ids, references, results_dir, price=None, with_bertscore=True):
    rows = load_model_outputs(model_name, record_ids, results_dir)
    n = len(record_ids)
    outputs = [r for r in rows if r["summary"]]
    valid = [r for r in outputs if not (r["report"] or {}).get("schema_errors")]
    claims = sum(r["report"]["grounding"]["claims"] for r in outputs)
    supported = sum(r["report"]["grounding"]["supported_claims"] for r in outputs)
    quotes = sum(r["report"]["grounding"]["quotes"] for r in outputs)
    quotes_found = sum(r["report"]["grounding"]["quotes_found"] for r in outputs)
    unsupported_per = [len(r["report"]["grounding"]["unsupported"]) for r in outputs]
    fresh = [r["report"] for r in rows if r["report"] and not r["report"].get("cached") and r["summary"]]
    latencies = [r["latency_s"] for r in fresh]
    tokens_in = [r["report"]["tokens_in"] for r in outputs]
    tokens_out = [r["report"]["tokens_out"] for r in outputs]

    def coverage(pred):
        return round(100 * sum(pred(r["summary"]) for r in outputs) / n, 1) if n else 0.0

    metrics = {
        "model": model_name,
        "model_id": model_id,
        "records": n,
        "outputs": len(outputs),
        "schema_valid_%": round(100 * len(valid) / n, 1),
        "claims_per_summary": round(claims / len(outputs), 1) if outputs else 0,
        "grounded_claims_%": round(100 * supported / claims, 1) if claims else 0,
        "unsupported_per_summary": round(statistics.mean(unsupported_per), 2) if outputs else None,
        "quotes_found_%": round(100 * quotes_found / quotes, 1) if quotes else 0,
        "has_problems_%": coverage(lambda s: bool(s["problems"])),
        "has_medications_%": coverage(lambda s: any(s["medications"][k] for k in s["medications"])),
        "has_action_items_%": coverage(lambda s: bool(s["action_items"])),
        "one_liner_over_300_chars": sum(len(r["summary"]["one_liner"]) > 300 for r in outputs),
        "latency_s_median": round(statistics.median(latencies), 1) if latencies else None,
        "tokens_in_mean": round(statistics.mean(tokens_in)) if tokens_in else None,
        "tokens_out_mean": round(statistics.mean(tokens_out)) if tokens_out else None,
    }
    if price and tokens_in:
        metrics["est_cost_usd_per_1k_records"] = round(
            1000 * (statistics.mean(tokens_in) * price["input"] + statistics.mean(tokens_out) * price["output"]) / 1e6, 2)

    preds = [r["summary"]["one_liner"] for r in outputs]
    refs = [references[r["record_id"]]["description"] for r in outputs]
    if preds:
        metrics["rougeL_one_liner"] = round(statistics.mean(_rouge_l(preds, refs)), 3)
        if with_bertscore:
            metrics["bertscore_f1_one_liner"] = round(statistics.mean(_bertscore(preds, refs)), 3)
    return metrics, rows
