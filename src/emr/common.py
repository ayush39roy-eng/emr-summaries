"""Small shared helpers: JSON files, record_id <-> filename, and the tools/ modules."""
import importlib.util
import json
from pathlib import Path

from emr.config import ROOT


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_tool = _load_tool("validate")
PatientIdRegistry = _load_tool("assign_patient_id").PatientIdRegistry

INPUT_SCHEMA = ROOT / "schemas" / "emr_input.schema.json"
SUMMARY_SCHEMA = ROOT / "schemas" / "emr_summary.schema.json"


def path_for(folder, record_id):
    return ROOT / folder / (record_id.replace(":", "_") + ".json")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def bundle_errors(bundle):
    return validate_tool.schema_errors(bundle, INPUT_SCHEMA)


def summary_errors(bundle, summary):
    return validate_tool.schema_errors(summary, SUMMARY_SCHEMA) + validate_tool.grounding_errors(bundle, summary)
