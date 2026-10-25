"""ChromaDB vector store. We compute embeddings ourselves (cached) and give Chroma the vectors + metadata.

Every query is filtered by record_id, so retrieval can only ever see one patient's own record.
"""
import chromadb

from emr.config import ROOT, load_config

COLLECTION = "emr_chunks"
META_KEYS = ("record_id", "patient_id", "doc_id", "section", "start", "end")


def collection():
    client = chromadb.PersistentClient(path=str(ROOT / load_config()["paths"]["chroma"]))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(col, chunks, embeddings):
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in META_KEYS} for c in chunks],
    )


def has_record(col, record_id):
    return bool(col.get(where={"record_id": record_id}, limit=1)["ids"])


def record_chunks(col, record_id):
    """All chunks of one record, in document order."""
    got = col.get(where={"record_id": record_id})
    chunks = [{"chunk_id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
    return sorted(chunks, key=lambda c: (c["doc_id"], c["start"]))


def query(col, query_embedding, record_id, k=4):
    got = col.query(query_embeddings=[query_embedding], n_results=k, where={"record_id": record_id})
    return [{"chunk_id": i, "text": d, "distance": round(dist, 4), **m}
            for i, d, m, dist in zip(got["ids"][0], got["documents"][0], got["metadatas"][0], got["distances"][0])]
