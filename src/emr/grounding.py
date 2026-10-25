"""Turn the model's quoted evidence into character spans, and flag claims we cannot find in the source.

A quote is looked up (1) exactly in the cited chunk, (2) ignoring case/whitespace in the cited chunk,
(3) the same way anywhere in the record's chunks (model cited the wrong chunk_id). If none match,
the quote is unsupported. A claim whose quotes are all unsupported (or that has no quotes) is an
unsupported claim — the main hallucination signal.
"""
import re


def _normalise_with_map(text):
    """Lower-case, collapse whitespace; return (normalised, index_map) where index_map[i] = original index."""
    out, index_map, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        index_map.append(i)
    return "".join(out), index_map


def find_quote(quote, chunk_text):
    """(start, end) of quote inside chunk_text, or None."""
    quote = quote.strip().strip('"').strip()
    if len(quote) < 3:
        return None
    pos = chunk_text.find(quote)
    if pos >= 0:
        return pos, pos + len(quote)
    norm_text, index_map = _normalise_with_map(chunk_text)
    norm_quote = re.sub(r"\s+", " ", quote.lower()).strip(" .,;:")
    pos = norm_text.find(norm_quote)
    if pos < 0 or not norm_quote:
        return None
    return index_map[pos], index_map[pos + len(norm_quote) - 1] + 1


def resolve(evidence, chunks_by_id):
    """[{chunk_id, quote}] -> ([{doc_id, start, end}], n_unresolved, n_wrong_chunk)"""
    spans, unresolved, wrong_chunk = [], 0, 0
    for ev in evidence:
        quote, cited = ev.get("quote", ""), chunks_by_id.get(ev.get("chunk_id"))
        candidates = ([cited] if cited else []) + [c for c in chunks_by_id.values() if c is not cited]
        for n, chunk in enumerate(candidates):
            hit = find_quote(quote, chunk["text"])
            if hit:
                spans.append({"doc_id": chunk["doc_id"], "start": chunk["start"] + hit[0],
                              "end": chunk["start"] + hit[1]})
                wrong_chunk += n > 0 or cited is None
                break
        else:
            unresolved += 1
    return spans, unresolved, wrong_chunk


def claims(generated):
    """Yield (json_pointer, claim_dict) for every evidence-bearing claim in a GeneratedSummary dict."""
    for key in ("problems", "allergies", "key_findings", "procedures", "action_items", "red_flags"):
        for i, item in enumerate(generated.get(key) or []):
            yield f"/{key}/{i}", item
    for group in ("started", "stopped", "changed", "continued"):
        for i, item in enumerate((generated.get("medications") or {}).get(group) or []):
            yield f"/medications/{group}/{i}", item
    for key in ("course", "disposition"):
        if generated.get(key):
            yield f"/{key}", generated[key]


def ground(generated, chunks):
    """Replace every claim's evidence with real spans (in place). Returns grounding stats."""
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    stats = {"claims": 0, "supported_claims": 0, "unsupported": [], "quotes": 0, "quotes_found": 0,
             "wrong_chunk_id": 0}
    for pointer, claim in claims(generated):
        evidence = claim.get("evidence") or []
        spans, unresolved, wrong = resolve(evidence, chunks_by_id)
        claim["evidence"] = spans
        stats["quotes"] += len(evidence)
        stats["quotes_found"] += len(spans)
        stats["wrong_chunk_id"] += wrong
        # course/disposition are narrative containers; count them only when they claim something specific
        if pointer in ("/course", "/disposition") and not evidence:
            continue
        stats["claims"] += 1
        if spans:
            stats["supported_claims"] += 1
        else:
            label = claim.get("name") or claim.get("action") or claim.get("message") or claim.get("substance") \
                or claim.get("status") or (claim.get("narrative") or "")[:60]
            stats["unsupported"].append({"pointer": pointer, "claim": label,
                                         "quotes": [e.get("quote") for e in evidence]})
    return stats
