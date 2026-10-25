"""Simplified shape the LLM fills in (structured-output modes can't take the full JSON Schema's anyOf/const).

Differences from schemas/emr_summary.schema.json:
- evidence is [{chunk_id, quote}] — an exact short quote — instead of character offsets.
  grounding.py turns quotes into {doc_id, start, end}; the model never produces offsets.
- record_id, patient_id, generator and schema_version are filled in by code, not by the model.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    chunk_id: str = Field(description="ID of the chunk the quote is copied from, e.g. mtsamples:12#d1-c3")
    quote: str = Field(description="Exact, short (3-25 words) quote copied character-for-character from that chunk")


class Problem(BaseModel):
    name: str
    status: Literal["confirmed", "suspected", "ruled_out", "chronic", "resolved"]
    is_new: Optional[bool] = None
    evidence: list[Evidence]


class Med(BaseModel):
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    previous: Optional[str] = None
    reason: Optional[str] = None
    evidence: list[Evidence]


class Medications(BaseModel):
    started: list[Med]
    stopped: list[Med]
    changed: list[Med]
    continued: list[Med]


class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    evidence: list[Evidence]


class Finding(BaseModel):
    category: Literal["lab", "vital", "imaging", "ecg", "exam", "other"]
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    flag: Optional[Literal["low", "high", "critical", "abnormal", "normal"]] = None
    interpretation: str
    evidence: list[Evidence]


class Procedure(BaseModel):
    name: str
    outcome: Optional[str] = None
    evidence: list[Evidence]


class Course(BaseModel):
    narrative: str = Field(description="Short narrative of what happened, max ~150 words")
    evidence: list[Evidence]


class Disposition(BaseModel):
    status: Literal["admitted", "discharged", "transferred", "expired", "left_ama", "other", "unknown"]
    destination: Optional[str] = None
    condition: Optional[str] = None
    evidence: list[Evidence]


class ActionItem(BaseModel):
    action: str
    owner: Literal["treating_team", "primary_care", "specialist", "patient", "lab", "pharmacy", "other"]
    priority: Literal["urgent", "soon", "routine"]
    due: Optional[str] = None
    category: Optional[Literal["follow_up", "pending_result", "monitoring", "medication", "referral",
                               "patient_education", "other"]] = None
    rationale: Optional[str] = None
    evidence: list[Evidence]


class RedFlag(BaseModel):
    message: str
    severity: Literal["critical", "high", "moderate"]
    evidence: list[Evidence]


class Quality(BaseModel):
    missing_information: list[str]
    contradictions: list[str]


class GeneratedSummary(BaseModel):
    one_liner: str = Field(description="One sentence, max 300 characters: age/sex, key history, problem, outcome")
    problems: list[Problem]
    medications: Medications
    allergies: list[Allergy]
    key_findings: list[Finding]
    procedures: list[Procedure]
    course: Course
    disposition: Disposition
    action_items: list[ActionItem]
    red_flags: list[RedFlag]
    quality: Quality
