"""Anonymization: Presidio + spaCy, plus custom recognizers for Indian identifiers.

Two failure modes matter:
- Leaks (an identifier survives)       -> measured by scripts/leak_test.py on fake notes.
- Over-redaction (a clinical fact is replaced, e.g. "3 days" -> <DATE>, "Foley" -> <PERSON>)
  -> controlled by the filters below and measured by the replacement counts on MTSamples.
"""
import re
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Presidio entity -> tag written into the text.
TAGS = {
    "PERSON": "<PERSON>",
    "DATE_TIME": "<DATE>",
    "PHONE_NUMBER": "<PHONE>",
    "IN_PHONE": "<PHONE>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "LOCATION": "<LOCATION>",
    "STREET_ADDRESS": "<LOCATION>",
    "AADHAAR": "<ID>",
    "MRN": "<ID>",
    "US_SSN": "<ID>",
    "IN_PAN": "<ID>",
    "URL": "<URL>",
    "IP_ADDRESS": "<ID>",
}

# Presidio compiles patterns case-INsensitive by default; that made "500 mg BID. Seen by Dr" look like a street
# address ("Dr" = Drive). Case matters for names and addresses, so custom patterns are case-sensitive.
CASE_SENSITIVE = re.MULTILINE


def pattern(entity, regex, score, context=None):
    return PatternRecognizer(entity, name=f"custom_{entity}", patterns=[Pattern(entity.lower(), regex, score)],
                             context=context, global_regex_flags=CASE_SENSITIVE)


CUSTOM_RECOGNIZERS = [
    # 10-digit Indian mobile, optionally +91 / 0 prefix, optional space/dash in the middle.
    pattern("IN_PHONE", r"(?<![\w+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", 0.7),
    # North-American style numbers incl. +1 / 001 prefix and extensions (Presidio's recognizer misses "x115").
    pattern("PHONE_NUMBER", r"(?<![\w+])(?:\+?1[\s.-]?|001[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
                            r"(?:\s?(?:x|ext\.?)\s?\d{1,5})?(?!\d)", 0.7),
    # 12-digit Aadhaar-like number, grouped 4-4-4 or not. First digit is never 0 or 1.
    pattern("AADHAAR", r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)", 0.75),
    # Hospital record numbers: MRN-12345, MRN: 1234567, UHID 12345678, Hospital No. 4567 ...
    pattern("MRN", r"(?i:\b(?:MRN|UHID|IPD|OPD|Reg(?:istration)?\.?\s*No|Hospital\s*No|Medical\s*Record\s*(?:No|Number)))"
                   r"\.?\s*[:#-]?\s*[A-Z]{0,3}[-/]?\d{4,10}\b", 0.9),
    # Names after a title or label: "Dr. Kim", "Name: Asha Rao", "Patient Terry Martin".
    pattern("PERSON", r"\b(?:Dr|Mr|Mrs|Ms|Smt|Shri)\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?", 0.8),
    pattern("PERSON", r"(?<=Name: )[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){0,3}", 0.85),
    pattern("PERSON", r"(?<=Patient )[A-Z][a-z'-]+\s+[A-Z][a-z'-]+", 0.6),
    # Full address lines: house number ... ending in "ST 12345" (US) or "City-123456" / "City 123456" (India).
    pattern("STREET_ADDRESS", r"(?:\bH\.\s?No\.\s?)?\b\d[\w/]*,?\s[\w\s,./-]{3,80}?"
                              r"(?:\b[A-Z]{2}\s\d{5}(?:-\d{4})?|\b[A-Z][a-z]+[-\s]\d{6})(?!\d)", 0.85),
    # Shorter street fragments: "12, MG Road", "4521 Oak Street", "B-14 Lajpat Nagar".
    pattern("STREET_ADDRESS", r"\b(?:[A-Z]-?)?\d{1,5}[A-Z]?,?\s+(?:[A-Z][a-zA-Z.]*\s+){0,4}"
                              r"(?:Street|Road|Avenue|Lane|Nagar|Marg|Colony|Chowk|Vihar|Enclave|"
                              r"Boulevard|Drive|Court|Place|Circle|Highway)\b", 0.8),
    pattern("IN_PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.6),
]

# DATE_TIME from spaCy also fires on durations, ranges and frequencies ("3 days", "4-6 months", "daily").
# Keep only things that look like an actual calendar date or year.
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_LIKE = re.compile(
    rf"\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{1,2}}-\d{{1,2}}|\b{MONTHS}\s+\d{{1,2}}|\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{MONTHS}"
    rf"|\b{MONTHS}\s*,?\s*\d{{4}}|\b(19|20)\d{{2}}\b",
    re.IGNORECASE,
)

# Words spaCy commonly mislabels as PERSON/LOCATION in clinical text. Lower-case match on the span.
CLINICAL_FALSE_POSITIVES = {
    "foley", "babinski", "parkinson", "alzheimer", "crohn", "hodgkin", "doppler", "mayo", "allis", "kocher",
    "bovie", "vicryl", "prolene", "monocryl", "dermabond", "steri-strips", "steri-strip", "tegaderm", "betadine",
    "marcaine", "lidocaine", "xylocaine", "tylenol", "motrin", "coumadin", "lasix", "toradol", "ancef", "versed",
    "kerlix", "kenalog", "depo-medrol", "duragesic", "percocet", "vicodin", "norco", "ativan", "valium",
    "ambien", "zofran", "phenergan", "demerol", "dilaudid", "keflex", "cipro", "flagyl", "zithromax", "lipitor",
    "glucophage", "lantus", "humalog", "neurontin", "lyrica", "xanax", "zoloft", "prozac", "plavix", "nexium",
    "protonix", "synthroid", "lopressor", "toprol", "cardizem", "norvasc", "zestril", "prinivil", "flomax",
    "mallampati", "glasgow", "apgar", "trendelenburg", "fowler", "valsalva", "romberg", "homan", "homans",
    "murphy", "mcburney", "rovsing", "lachman", "phalen", "tinel", "spurling", "hoffmann", "hoffman",
    "bell", "graves", "hashimoto", "cushing", "addison", "raynaud", "sjogren", "wilms", "ewing", "kaposi",
    "hartmann", "whipple", "nissen", "roux", "billroth", "mcmurray", "kirschner", "steinmann", "jackson-pratt",
    "jp", "penrose", "deaver", "metzenbaum", "mayo-hegar", "debakey", "kelly", "army-navy", "richardson",
    "balfour", "bookwalter", "weitlaner", "gelpi", "hemovac", "ioban", "xeroform", "adaptic", "coban",
    "ace", "dvt", "copd", "chf", "mri", "ct", "ekg", "ecg", "iv", "po", "prn", "bid", "tid", "qid", "hpi",
    "ros", "pmh", "heent", "neck", "chest", "abdomen", "extremities", "neuro", "plan", "impression",
    "x", "xyz", "abc", "patient", "the patient", "mr.", "mrs.", "ms.", "dr.", "dr", "mr", "mrs", "ms",
    "lord", "christ", "mercy",
    # eponymous devices / tests / views seen in MTSamples
    "tailor", "rao", "lao", "plano", "cook", "blake", "glenn", "allen", "miller", "case", "conference",
    "mac", "fogarty", "swan", "ganz", "swan-ganz", "hickman", "groshong", "port-a-cath", "greenfield",
    "harrington", "luque", "smith", "nephew", "stryker", "zimmer", "synthes", "medtronic", "pfannenstiel",
    "kerrison", "leksell", "cobb", "langenbeck", "freer", "adson", "frazier", "senn", "hohmann", "bennett",
    "rochester", "carmalt", "babcock", "foerster", "heaney", "sims", "graves", "pederson", "auvard",
}


@lru_cache(maxsize=1)
def gazetteers():
    """Known personal names and place names. spaCy's PERSON/LOCATION guesses on clinical text are noisy
    ("Diverticulosis" -> PERSON, "Isthmus" -> LOCATION), so a guess is kept only if it contains a known name/place."""
    import geonamescache
    from faker import Faker

    names = set()
    for locale in ("en_US", "en_IN", "en_GB"):
        provider = Faker(locale).provider("faker.providers.person")
        for attr in ("first_names", "first_names_male", "first_names_female", "last_names"):
            names.update(n.lower() for n in getattr(provider, attr, ()))
    gc = geonamescache.GeonamesCache()
    places = {c["name"].lower() for c in gc.get_cities().values()}
    places |= {c["name"].lower() for c in gc.get_countries().values()}
    places |= {s["name"].lower() for s in gc.get_us_states().values()}
    return names - CLINICAL_FALSE_POSITIVES, places - CLINICAL_FALSE_POSITIVES


@lru_cache(maxsize=1)
def engines():
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.remove_recognizer("PhoneRecognizer")
    analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=("US", "IN", "GB")))
    for recognizer in CUSTOM_RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
    return analyzer, AnonymizerEngine()


def keep_result(result: RecognizerResult, text: str) -> bool:
    span = text[result.start:result.end].strip()
    lowered = span.lower().strip(".,:;")
    if result.entity_type == "DATE_TIME":
        return bool(DATE_LIKE.search(span))
    if (result.recognition_metadata or {}).get("recognizer_name", "").startswith("custom_"):
        return True  # our own regexes are already specific
    if result.entity_type in ("PERSON", "LOCATION"):
        names, places = gazetteers()
        words = [w.strip(".,:;'s") for w in lowered.split()]
        introduced = re.search(r"(?:Patient|Name:|\b(?:Dr|Mr|Mrs|Ms|Smt|Shri)\.?)\s*$", text[max(0, result.start - 12):result.start])
        if result.entity_type == "PERSON" and introduced and not span.isupper():
            return True
        if result.entity_type == "PERSON" and (not any(w in names for w in words) or (span.isupper() and len(span) <= 4)):
            return False
        if result.entity_type == "LOCATION" and lowered not in places and not any(w in places for w in words if len(w) > 3):
            return False
        if lowered in CLINICAL_FALSE_POSITIVES or all(w in CLINICAL_FALSE_POSITIVES for w in lowered.split()):
            return False
        if span.isupper() and len(span) > 3 and result.entity_type == "LOCATION":
            return False  # ALL-CAPS section headings, e.g. "HEENT", "EXTREMITIES"
        if not any(c.isalpha() for c in span) or span.startswith("<"):
            return False
    return True


def find_identifiers(text):
    analyzer, _ = engines()
    results = analyzer.analyze(text=text, language="en", entities=list(TAGS), score_threshold=0.35)
    return [r for r in results if keep_result(r, text)]


def anonymize_text(text):
    """Return (anonymized_text, counts_by_entity_type)."""
    _, anonymizer = engines()
    results = find_identifiers(text)
    operators = {entity: OperatorConfig("replace", {"new_value": tag}) for entity, tag in TAGS.items()}
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    counts = {}
    for item in anonymized.items:
        counts[item.entity_type] = counts.get(item.entity_type, 0) + 1
    return anonymized.text, counts


def anonymize_bundle(bundle):
    """Return (new_bundle, counts). Only document text changes; patient_id is already a random UUID."""
    new = {**bundle, "documents": []}
    total = {}
    for doc in bundle["documents"]:
        text, counts = anonymize_text(doc["text"])
        new["documents"].append({**doc, "text": text})
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    return new, total
