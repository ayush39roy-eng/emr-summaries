"""MTSamples CSV -> raw EMRBundles (text not yet anonymized).

MTSamples quirks handled here:
- Line breaks were flattened to commas ("...4.7 cm.,2.  Normal ..."), so we restore them.
- The same transcription is often filed under several specialties (e.g. "Surgery" and "Orthopedic").
  We keep one bundle per unique text and pick the most specific specialty as the department.
- There is no patient identifier, so each unique report is its own patient (key "mtsamples:<row>").
"""
import re

import pandas as pd

# Broad "document kind" categories rather than clinical specialties.
GENERIC_SPECIALTIES = {
    "Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
    "Office Notes", "Letters", "Emergency Room Reports", "IME-QME-Work Comp etc.",
}

# First match wins, so the most informative document type is checked first.
DOC_TYPE_BY_SPECIALTY = [
    ("Discharge Summary", "discharge_summary"),
    ("Radiology", "radiology_report"),
    ("Consult - History and Phy.", "history_physical"),
    ("SOAP / Chart / Progress Notes", "progress_note"),
    ("Surgery", "operative_note"),
    ("Lab Medicine - Pathology", "lab_report"),
]

ENCOUNTER_BY_SPECIALTY = {
    "Emergency Room Reports": "emergency",
    "Discharge Summary": "inpatient",
}


def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"\s*,\s*,\s*", "\n", text)        # ", ," was an empty line
    text = re.sub(r":[ \t]*,[ \t]*", ": ", text)       # "DIAGNOSIS: , Coronary" -> "DIAGNOSIS: Coronary"
    text = re.sub(r"(?<=\.),(?=\S)", "\n", text)       # ".,2." was a line break
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pick(specialties, table, default):
    for specialty, value in table:
        if specialty in specialties:
            return value
    return default


def group_rows(df, min_chars):
    """Yield (first_row_index, row, [all specialties]) for each unique transcription."""
    df = df.copy()
    df["transcription"] = df["transcription"].fillna("").str.strip()
    df["medical_specialty"] = df["medical_specialty"].fillna("").str.strip()
    df = df[df["transcription"].str.len() >= min_chars]
    for _, group in df.groupby("transcription", sort=False):
        first = group.iloc[0]
        yield int(first["Unnamed: 0"]), first, list(dict.fromkeys(group["medical_specialty"]))


def row_to_bundle(row_index, row, specialties, patient_id):
    specific = [s for s in specialties if s not in GENERIC_SPECIALTIES]
    department = specific[0] if specific else specialties[0]
    encounter_type = next((ENCOUNTER_BY_SPECIALTY[s] for s in specialties if s in ENCOUNTER_BY_SPECIALTY), "unknown")
    return {
        "schema_version": "0.1",
        "record_id": f"mtsamples:{row_index}",
        "source": {
            "dataset": "mtsamples",
            "native_ids": {"row": row_index, "sample_name": str(row["sample_name"]).strip()},
            "license": "CC0",
        },
        "patient": {"patient_id": patient_id},
        "encounter": {"type": encounter_type, "department": department},
        "documents": [{
            "doc_id": "d1",
            "doc_type": pick(specialties, DOC_TYPE_BY_SPECIALTY, "other"),
            "language": "en",
            "origin": "typed",
            "text": clean_text(row["transcription"]),
        }],
    }


def convert(csv_path, registry, min_chars=300):
    """Return (bundles, references). references[record_id] = MTSamples description (weak gold one-liner)."""
    df = pd.read_csv(csv_path)
    bundles, references = [], {}
    for row_index, row, specialties in group_rows(df, min_chars):
        patient_id = registry.get(f"mtsamples:{row_index}")
        bundle = row_to_bundle(row_index, row, specialties, patient_id)
        bundles.append(bundle)
        references[bundle["record_id"]] = {
            "description": str(row["description"]).strip(),
            "specialties": specialties,
            "sample_name": str(row["sample_name"]).strip(),
        }
    return bundles, references
