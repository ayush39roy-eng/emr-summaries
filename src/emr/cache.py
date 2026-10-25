"""Disk cache for every LLM and embedding call, keyed by (kind, model, payload hash).

Reruns are free and reproducible: the same model + same prompt returns the saved answer.
"""
import hashlib
import json

from emr.common import read_json, write_json
from emr.config import ROOT, load_config

CACHE_DIR = ROOT / load_config()["paths"]["cache"]


def key(*parts):
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(kind, model, k):
    return CACHE_DIR / kind / model.replace("/", "_").replace(":", "_") / f"{k}.json"


def get(kind, model, k):
    path = _path(kind, model, k)
    return read_json(path) if path.exists() else None


def put(kind, model, k, value):
    write_json(_path(kind, model, k), value)
