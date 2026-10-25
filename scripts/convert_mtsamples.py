"""Phase 1: data/raw/mtsamples.csv -> data/bundles/raw/*.json (+ data/references.json), and print stats."""
import statistics
from collections import Counter

from emr.common import PatientIdRegistry, path_for, write_json
from emr.config import ROOT, load_config
from emr.converters.mtsamples import convert


def main():
    config = load_config()
    paths = config["paths"]
    (ROOT / "secrets").mkdir(exist_ok=True)
    registry = PatientIdRegistry(ROOT / paths["patient_id_lookup"])

    bundles, references = convert(ROOT / paths["raw_csv"], registry, config["dataset"]["min_chars"])
    for bundle in bundles:
        write_json(path_for(paths["bundles_raw"], bundle["record_id"]), bundle)
    write_json(ROOT / "data" / "references.json", references)

    lengths = [len(b["documents"][0]["text"]) for b in bundles]
    print(f"Converted {len(bundles)} unique reports (rows < {config['dataset']['min_chars']} chars and duplicates dropped)")
    print(f"Text length: median {statistics.median(lengths):.0f} chars, max {max(lengths)}, "
          f">= 1500 chars: {sum(n >= 1500 for n in lengths)}")
    print("\ndoc_type:", dict(Counter(b["documents"][0]["doc_type"] for b in bundles).most_common()))
    print("encounter.type:", dict(Counter(b["encounter"]["type"] for b in bundles).most_common()))
    print("\nTop departments:")
    for dept, n in Counter(b["encounter"]["department"] for b in bundles).most_common(15):
        print(f"  {n:5d}  {dept}")


if __name__ == "__main__":
    main()
