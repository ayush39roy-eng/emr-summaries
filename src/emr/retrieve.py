"""Per-section retrieval for one record, plus indexing (chunk -> embed -> store).

For each part of the summary we ask the vector DB for the most relevant chunks of THIS record only.
MTSamples reports are short, so when the whole record fits comfortably in the prompt we pass every chunk
(nothing can be missed), but we still run and log the per-section retrieval so it can be inspected.
"""
from emr import store
from emr.chunking import chunk_bundle

SECTION_QUERIES = {
    "problems": "diagnoses, assessment, impression, problem list, postoperative diagnosis",
    "medications": "medications, drugs and doses, prescribed, started, stopped, discontinued",
    "findings": "laboratory results, vital signs, imaging, x-ray, CT, MRI, physical examination findings",
    "course": "history of present illness, hospital course, procedure performed, what happened",
    "plan": "plan, follow-up, recommendations, discharge instructions, disposition",
    "allergies": "allergies, drug allergy, adverse reaction",
}
ALL_CHUNKS_MAX_WORDS = 3000


def index_bundle(col, embedder, bundle, chunking_cfg):
    """Chunk + embed + upsert one bundle (skipped if already in the store). Returns number of chunks."""
    if store.has_record(col, bundle["record_id"]):
        return 0
    chunks = chunk_bundle(bundle, chunking_cfg["window_words"], chunking_cfg["overlap_words"])
    store.add_chunks(col, chunks, embedder.embed([c["text"] for c in chunks], task="document"))
    return len(chunks)


def retrieve_for_record(col, embedder, record_id, k=4):
    """Returns {"chunks": [...], "mode": "all_chunks"|"retrieved", "per_section": {section: [(chunk_id, distance)]}}"""
    query_vectors = dict(zip(SECTION_QUERIES, embedder.embed(list(SECTION_QUERIES.values()), task="query")))
    per_section, selected = {}, {}
    for section, vector in query_vectors.items():
        hits = store.query(col, vector, record_id, k=k)
        per_section[section] = [(h["chunk_id"], h["distance"]) for h in hits]
        for h in hits:
            selected[h["chunk_id"]] = h

    every_chunk = store.record_chunks(col, record_id)
    total_words = sum(len(c["text"].split()) for c in every_chunk)
    if total_words <= ALL_CHUNKS_MAX_WORDS:
        mode, chunks = "all_chunks", every_chunk
    else:
        mode = "retrieved"
        chunks = sorted(selected.values(), key=lambda c: (c["doc_id"], c["start"]))
    for c in chunks:
        c.pop("distance", None)
    return {"record_id": record_id, "mode": mode, "total_words": total_words,
            "chunks": chunks, "per_section": per_section}
