"""Section-aware chunking with character offsets into the anonymized document text.

MTSamples uses ALL-CAPS headings followed by a colon ("HISTORY OF PRESENT ILLNESS:"). Each heading starts a
section; very short sections are merged into the next one, long ones are split into overlapping word windows.
Every chunk records start/end so that text[start:end] == chunk text, always.
"""
import re

HEADING = re.compile(r"^[ \t]*([A-Z][A-Z0-9 /&,()'.-]{2,60}):", re.MULTILINE)
MIN_SECTION_WORDS = 25


def find_sections(text):
    """[(name, start, end)] covering the whole text. Text before the first heading is 'PREAMBLE'."""
    starts = [(m.group(1).strip(), m.start()) for m in HEADING.finditer(text)]
    if not starts or starts[0][1] > 0:
        starts.insert(0, ("PREAMBLE", 0))
    sections = []
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        if text[start:end].strip():
            sections.append((name, start, end))
    return sections


def merge_short(sections, text):
    merged = []
    pending = None
    for name, start, end in sections:
        if pending:
            name, start = f"{pending[0]} + {name}", pending[1]
            pending = None
        if len(text[start:end].split()) < MIN_SECTION_WORDS:
            pending = (name, start)
            continue
        merged.append((name, start, end))
    if pending:  # short trailing section: attach to the previous chunk
        if merged:
            last_name, last_start, _ = merged.pop()
            merged.append((f"{last_name} + {pending[0]}", last_start, len(text)))
        else:
            merged.append((pending[0], pending[1], len(text)))
    return merged


def split_windows(start, end, text, window_words, overlap_words):
    """Split text[start:end] into word windows; returns [(start, end)] with real character offsets."""
    words = [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]
    if len(words) <= window_words:
        return [(start, end)]
    spans, step = [], window_words - overlap_words
    for i in range(0, len(words), step):
        piece = words[i:i + window_words]
        spans.append((piece[0][0], piece[-1][1]))
        if i + window_words >= len(words):
            break
    return spans


def chunk_bundle(bundle, window_words=500, overlap_words=75):
    record_id, patient_id = bundle["record_id"], bundle["patient"]["patient_id"]
    chunks = []
    for doc in bundle["documents"]:
        text = doc["text"]
        for name, start, end in merge_short(find_sections(text), text):
            for s, e in split_windows(start, end, text, window_words, overlap_words):
                # trim surrounding whitespace but keep offsets exact
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                chunks.append({
                    "chunk_id": f"{record_id}#{doc['doc_id']}-c{len(chunks)}",
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "doc_id": doc["doc_id"],
                    "section": name,
                    "start": s,
                    "end": e,
                    "text": text[s:e],
                })
    return chunks
