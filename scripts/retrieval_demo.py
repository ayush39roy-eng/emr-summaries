"""Phase 3 demo: ask the vector DB questions about ONE record and show which chunks come back.

    python scripts/retrieval_demo.py                       # record with the most chunks in the test set
    python scripts/retrieval_demo.py --record mtsamples:3879 --query "what medications was the patient on?"

Uses cached embeddings, so after the pipeline has run once this makes at most a few cheap embedding calls.
"""
import argparse

from emr import store
from emr.common import read_json
from emr.config import ROOT, load_config
from emr.llm.base import make_llm

DEFAULT_QUERIES = [
    "What is the diagnosis?",
    "What medications is the patient taking?",
    "What did the physical examination or tests show?",
    "What is the plan and follow-up?",
    "Does the patient have any allergies?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record")
    parser.add_argument("--query", action="append", help="repeatable; default: 5 standard questions")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    config = load_config()
    col = store.collection()
    record_ids = read_json(ROOT / config["paths"]["test_set"])
    record_id = args.record or max(record_ids, key=lambda r: len(store.record_chunks(col, r)))
    chunks = store.record_chunks(col, record_id)
    print(f"Record {record_id}: {len(chunks)} chunks in ChromaDB (search is filtered to this record only)\n")

    embedder = make_llm(config["embedding"])
    queries = args.query or DEFAULT_QUERIES
    for query, vector in zip(queries, embedder.embed(queries, task="query")):
        print(f"Q: {query}")
        for hit in store.query(col, vector, record_id, k=args.k):
            preview = " ".join(hit["text"].split())[:110]
            print(f"   distance {hit['distance']:.3f}  [{hit['section'][:40]}]  {preview}...")
        print()


if __name__ == "__main__":
    main()
