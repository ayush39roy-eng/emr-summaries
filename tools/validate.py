import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def schema_errors(instance, schema_path):
    validator = Draft202012Validator(load(schema_path), format_checker=FormatChecker())
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(instance)]


def walk_evidence(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "evidence" and isinstance(v, list):
                for i, span in enumerate(v):
                    yield f"{path}/{k}/{i}", span
            else:
                yield from walk_evidence(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_evidence(v, f"{path}/{i}")


def grounding_errors(bundle, summary):
    errors = []
    if bundle["record_id"] != summary["record_id"]:
        errors.append("record_id mismatch between bundle and summary")
    if bundle.get("patient", {}).get("patient_id") != summary.get("patient_id"):
        errors.append("patient_id mismatch between bundle and summary")
    docs = {d["doc_id"]: d["text"] for d in bundle["documents"]}
    for where, span in walk_evidence(summary):
        if "doc_id" in span:
            text = docs.get(span["doc_id"])
            if text is None:
                errors.append(f"{where}: unknown doc_id {span['doc_id']}")
            elif not 0 <= span["start"] < span["end"] <= len(text):
                errors.append(f"{where}: span {span['start']}-{span['end']} outside doc of length {len(text)}")
    return errors


def main(bundle_path, summary_path=None):
    bundle = load(bundle_path)
    errors = [f"input {e}" for e in schema_errors(bundle, SCHEMA_DIR / "emr_input.schema.json")]
    if summary_path:
        summary = load(summary_path)
        errors += [f"summary {e}" for e in schema_errors(summary, SCHEMA_DIR / "emr_summary.schema.json")]
        errors += grounding_errors(bundle, summary)
    for e in errors:
        print(e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
