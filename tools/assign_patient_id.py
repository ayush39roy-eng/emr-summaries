import json
import sys
import uuid
from pathlib import Path


class PatientIdRegistry:
    """Maps a real identifier (name, MRN, MIMIC subject_id) to a random UUIDv4.

    The lookup file is the only place the real identifier exists. Keep it with the
    data custodian, outside the pipeline, and never commit it.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.table = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, real_identifier):
        key = str(real_identifier).strip().lower()
        if key not in self.table:
            self.table[key] = str(uuid.uuid4())
            self.path.write_text(json.dumps(self.table, indent=2))
        return self.table[key]


if __name__ == "__main__":
    registry = PatientIdRegistry(sys.argv[1])
    for identifier in sys.argv[2:]:
        print(registry.get(identifier))
