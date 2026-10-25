"""Retrieved chunks -> prompt -> model -> grounded ActionableSummary (full schema)."""
from datetime import datetime, timezone

from emr.generation_model import GeneratedSummary
from emr.grounding import ground

SYSTEM_PROMPT = """You are a careful clinical documentation assistant. You turn one patient's medical record into a
structured, actionable summary for a busy clinician.

Rules:
1. Use ONLY facts stated in the provided chunks. Never add facts from general medical knowledge.
2. If something is not in the chunks, leave that list empty (or use "unknown" / null). Do not guess.
3. Every claim needs evidence: the chunk_id and an EXACT quote of 3-25 words copied character-for-character
   from that chunk. Do not paraphrase quotes, do not fix typos, do not join text from different places.
4. Placeholders like <PERSON>, <DATE>, <LOCATION>, <ID> are anonymized identifiers. Never try to fill them in.
5. one_liner: one sentence, max 300 characters (age/sex if stated, key history, main problem, outcome/plan).
6. medications: put drugs given or prescribed during this encounter in "started", drugs explicitly stopped/held
   in "stopped", dose changes in "changed", existing home medications in "continued".
7. key_findings: abnormal or decision-relevant labs, vitals, imaging, exam findings only.
8. action_items: concrete next steps actually stated or clearly implied by the plan (follow-up, pending results,
   monitoring, referrals). red_flags: only genuinely safety-critical items; often there are none.
9. course.narrative: at most ~150 words."""

PROMPT_VERSION = "v1"


def build_prompt(context):
    parts = [f"Record {context['record_id']}. The record is split into the following chunks.\n"]
    for c in context["chunks"]:
        parts.append(f"=== chunk_id: {c['chunk_id']} | section: {c['section']} ===\n{c['text']}\n")
    parts.append("Write the structured summary as JSON. Remember: every quote must be copied exactly from its chunk.")
    return "\n".join(parts)


def to_full_schema(generated, bundle, model_id, unsupported_pointers):
    g = generated

    def meds(group):
        return [{"name": m["name"], "dose": m.get("dose"), "route": m.get("route"), "frequency": m.get("frequency"),
                 "previous": m.get("previous"), "reason": m.get("reason"), "evidence": m["evidence"]}
                for m in (g.get("medications") or {}).get(group) or []]

    return {
        "schema_version": "0.1",
        "record_id": bundle["record_id"],
        "patient_id": bundle["patient"]["patient_id"],
        "generator": {"kind": "model", "model": model_id, "prompt_version": PROMPT_VERSION,
                      "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "one_liner": g.get("one_liner", ""),
        "problems": [{"name": p["name"], "code": None, "status": p["status"], "is_new": p.get("is_new"),
                      "evidence": p["evidence"]} for p in g.get("problems") or []],
        "medications": {k: meds(k) for k in ("started", "stopped", "changed", "continued")},
        "allergies": [{"substance": a["substance"], "reaction": a.get("reaction"), "evidence": a["evidence"]}
                      for a in g.get("allergies") or []],
        "key_findings": [{"category": f["category"], "name": f["name"], "value": f.get("value"),
                          "unit": f.get("unit"), "flag": f.get("flag"), "trend": None,
                          "interpretation": f["interpretation"], "time": None, "evidence": f["evidence"]}
                         for f in g.get("key_findings") or []],
        "procedures": [{"name": p["name"], "time": None, "outcome": p.get("outcome"), "evidence": p["evidence"]}
                       for p in g.get("procedures") or []],
        "course": {"narrative": (g.get("course") or {}).get("narrative", ""),
                   "evidence": (g.get("course") or {}).get("evidence", [])},
        "disposition": {"status": (g.get("disposition") or {}).get("status", "unknown"),
                        "destination": (g.get("disposition") or {}).get("destination"),
                        "condition": (g.get("disposition") or {}).get("condition"),
                        "evidence": (g.get("disposition") or {}).get("evidence", [])},
        "action_items": [{"action": a["action"], "owner": a["owner"], "priority": a["priority"], "due": a.get("due"),
                          "category": a.get("category"), "rationale": a.get("rationale"), "evidence": a["evidence"]}
                         for a in g.get("action_items") or []],
        "red_flags": [{"message": r["message"], "severity": r["severity"], "evidence": r["evidence"]}
                      for r in g.get("red_flags") or []],
        "patient_summary": None,
        "quality": {"missing_information": (g.get("quality") or {}).get("missing_information", []),
                    "contradictions": (g.get("quality") or {}).get("contradictions", []),
                    "low_confidence_fields": unsupported_pointers},
    }


def summarize(llm, bundle, context):
    """Returns (summary_or_None, report). report holds grounding stats, latency, tokens, errors."""
    generation = llm.generate(build_prompt(context), GeneratedSummary, system=SYSTEM_PROMPT)
    report = {"record_id": bundle["record_id"], "model": llm.model_id, "latency_s": round(generation.latency_s, 2),
              "tokens_in": generation.tokens_in, "tokens_out": generation.tokens_out, "cached": generation.cached,
              "error": generation.error, "context_mode": context["mode"], "n_chunks": len(context["chunks"])}
    if generation.data is None:
        return None, report
    try:
        GeneratedSummary.model_validate(generation.data)
    except Exception as exc:
        report["error"] = f"output did not match generation model: {str(exc)[:300]}"
        return None, report
    generated = generation.data
    stats = ground(generated, context["chunks"])
    report["grounding"] = stats
    summary = to_full_schema(generated, bundle, llm.model_id, [u["pointer"] for u in stats["unsupported"]])
    return summary, report
