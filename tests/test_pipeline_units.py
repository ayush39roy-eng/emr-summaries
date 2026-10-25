"""Unit tests for the pieces that must never be wrong: offsets, grounding, conversion, anonymization."""
import pytest

from emr.chunking import chunk_bundle
from emr.converters.mtsamples import clean_text
from emr.grounding import find_quote, ground

BUNDLE = {
    "record_id": "test:1",
    "patient": {"patient_id": "87ff90af-12c5-4fe6-826c-c4becc85a874"},
    "documents": [{"doc_id": "d1", "text": (
        "CHIEF COMPLAINT: Chest pain for 3 days, worse on exertion, radiating to the left arm and jaw.\n"
        "HISTORY OF PRESENT ILLNESS: The patient is a 64-year-old male with hypertension and diabetes who "
        "presents with substernal chest pain. He takes metformin 500 mg twice daily and lisinopril 10 mg daily.\n"
        "PLAN: Admit to telemetry. Start aspirin 325 mg. Cardiology consult in the morning. Repeat troponin "
        "in 6 hours and obtain an echocardiogram. Continue home metformin. Hold lisinopril tonight "
        "and recheck blood pressure and renal function tomorrow morning."
    )}],
}


def test_clean_text_restores_line_breaks():
    raw = "2-D M-MODE: , ,1.  Left atrial enlargement.,2.  Normal LV.,PLAN: , Follow up."
    assert clean_text(raw) == "2-D M-MODE:\n1.  Left atrial enlargement.\n2.  Normal LV.\nPLAN: Follow up."


def test_chunk_offsets_are_exact():
    chunks = chunk_bundle(BUNDLE)
    text = BUNDLE["documents"][0]["text"]
    assert len(chunks) >= 2
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]
        assert c["record_id"] == "test:1" and c["patient_id"] == BUNDLE["patient"]["patient_id"]


def test_long_section_is_windowed_with_exact_offsets():
    text = "FINDINGS: " + " ".join(f"word{i}" for i in range(1200))
    bundle = {**BUNDLE, "documents": [{"doc_id": "d1", "text": text}]}
    chunks = chunk_bundle(bundle, window_words=500, overlap_words=75)
    assert len(chunks) == 3
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["text"]


def test_find_quote_exact_and_normalised():
    text = "He takes metformin 500 mg twice daily\n and lisinopril."
    assert find_quote("metformin 500 mg", text) == (9, 25)
    start, end = find_quote("Twice  daily and lisinopril", text)
    assert text[start:end] == "twice daily\n and lisinopril"
    assert find_quote("warfarin 5 mg", text) is None


def test_grounding_flags_unsupported_claims():
    chunks = chunk_bundle(BUNDLE)
    cid = chunks[-1]["chunk_id"]
    generated = {
        "problems": [
            {"name": "Chest pain", "status": "confirmed", "evidence": [{"chunk_id": cid, "quote": "Chest pain for 3 days"}]},
            {"name": "Pneumonia", "status": "confirmed", "evidence": [{"chunk_id": cid, "quote": "right lower lobe infiltrate"}]},
        ],
        "medications": {"started": [{"name": "aspirin", "evidence": [{"chunk_id": cid, "quote": "Start aspirin 325 mg"}]}],
                        "stopped": [], "changed": [], "continued": []},
    }
    stats = ground(generated, chunks)
    assert stats["claims"] == 3 and stats["supported_claims"] == 2
    assert stats["unsupported"][0]["claim"] == "Pneumonia"
    span = generated["medications"]["started"][0]["evidence"][0]
    assert BUNDLE["documents"][0]["text"][span["start"]:span["end"]] == "Start aspirin 325 mg"
    assert stats["wrong_chunk_id"] >= 1  # "Chest pain for 3 days" lives in the first chunk, not the cited one


@pytest.fixture(scope="module")
def anonymize_text():
    pytest.importorskip("en_core_web_lg")
    from emr.anonymize import anonymize_text
    return anonymize_text


def test_anonymizer_removes_identifiers_keeps_clinical_facts(anonymize_text):
    note = ("Patient Rahul Sharma, MRN-483920, phone +91 98765 43210, Aadhaar 2345 6789 0123, seen on 12/03/2024 "
            "by Dr. Mehta. Chest pain for 3 days. Foley catheter placed. Lasix 40 mg IV daily. Follow up in 4-6 weeks.")
    out, _ = anonymize_text(note)
    for identifier in ("Rahul", "Sharma", "483920", "98765", "2345 6789", "12/03/2024", "Mehta"):
        assert identifier not in out
    for fact in ("3 days", "Foley catheter", "Lasix 40 mg IV daily", "4-6 weeks"):
        assert fact in out
