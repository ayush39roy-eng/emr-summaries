# EMR → Actionable Summary: pipeline and model comparison (DRAFT)

DTU project "An App for Transforming Raw EMRs into Actionable Summaries" (CCDR / Koshika Foundation).
Prepared by: Ayush Roy (Research Intern). Date: 25 Oct 2026. **Status: draft. Model runs still in progress;
tables marked ⏳ are filled in as they finish** (full tables: `results/comparison.md`).

## 1. What was built

One runnable pipeline (`python scripts/run_pipeline.py --model <name>`):

1. **Data**: MTSamples (public, CC0, ~5,000 transcribed reports from 40 specialties) → converted to our
   input schema (EMRBundle). After removing duplicates (the same report is often filed under 2–3 specialties)
   and very short notes: **2,319 unique reports, 100% schema-valid**.
2. **Anonymization**: Microsoft Presidio + spaCy, plus our own rules for Indian identifiers (10-digit / +91
   mobiles, 12-digit Aadhaar-like numbers, MRN/UHID numbers, Indian addresses). Each patient gets a random UUID;
   the lookup table stays in `secrets/` (never committed). All 2,319 bundles anonymized and still schema-valid.
3. **Vector DB (RAG)**: each report is split at its section headings, embedded with `gemini-embedding-001` and
   stored in ChromaDB. Retrieval is always filtered to the one patient's own record.
4. **Summary generation**: the model writes a structured ActionableSummary (problems, medication changes, key
   findings, course, disposition, action items, red flags). **Every claim must quote the source text.**
5. **Grounding check (hallucination measure)**: our code searches for each quote in the source. If the quote
   is found, the claim gets exact character offsets; if not, the claim is marked **unsupported**. The model
   never invents offsets. Every output is validated against the full output schema.

All models get **exactly the same retrieved text and the same prompt**; only the model changes.

## 2. Anonymization leak test

60 fake clinical notes (Indian and US names, +91 and 10-digit phones, Aadhaar-like numbers, MRNs, dates,
addresses; generated with Faker, not real people), tested on a seed **not** used while building the rules.

| Identifier | Planted | Removed | Recall |
|---|---|---|---|
| Aadhaar-like number | 60 | 60 | 100% |
| MRN / UHID | 60 | 60 | 100% |
| Date | 60 | 60 | 100% |
| Patient name | 60 | 59 | 98.3% |
| Doctor name | 60 | 59 | 98.3% |
| Phone | 60 | 59 | 98.3% |
| Address | 60 | 55 | 91.7% |
| **Overall** | **420** | **412** | **98.1%** (target ≥ 95%) |

Clinical phrases that must survive ("3 days", "Foley catheter", "Lasix 40 mg IV daily", "metformin 500 mg BID",
"2 weeks") were kept 100% of the time.

**What leaked:** US military-style addresses ("FPO AE 99784"), one Indian address with an unusual city name,
one US phone written without separators, one rare Indian name, and doctor surnames that are also common words
("Date"). **Honest caveats:** (a) MTSamples is already de-identified, so real-data recall can only be estimated
with synthetic notes; (b) the names in the test come from the same Faker name lists the anonymizer uses, so the
name recall is optimistic; (c) on MTSamples a few clinical eponyms are still removed ("Tanner" stage, "Weber"
test). Before real hospital data, this needs a test on a small set of hand-labelled real notes.

## 3. Test set and models

- **Test set:** 50 MTSamples reports, ≥ 1,500 characters, spread over 38 specialties (fixed seed, saved in
  `data/test_set.json`). Median length 3,200 characters (~510 words); every report fits in one prompt.
- **Models:**

| Short name | Model | Where it runs |
|---|---|---|
| gemini-flash | gemini-3.8-flash | Google API (free tier) |
| gemini-flash-3.5 | gemini-3.5-flash | Google API (free tier) |
| gemini-flash-lite | gemini-3.5-flash-lite | Google API (free tier) |
| local-gemma | gemma4:e4b (Google open model) | this laptop (Ollama, Apple M4) |
| local-qwen | qwen3:8b (Alibaba open model) | this laptop |
| local-medgemma | medgemma:4b (Google, medical fine-tune) | this laptop |
| gemini-pro | gemini-3.1-pro-preview | **not run: needs a paid key** (free-tier quota is 0) |

## 4. Results ⏳

Full-test-set results (50 records):

| Metric | gemini-flash-lite | gemini-flash | gemini-flash-3.5 |
|---|---|---|---|
| Valid outputs (schema) | 98% | ⏳ | ⏳ |
| Claims per summary | 15.0 | ⏳ | ⏳ |
| **Grounded claims** | **99.5%** | ⏳ | ⏳ |
| Unsupported claims / summary | 0.08 | ⏳ | ⏳ |
| Has problems / meds / action items | 100% / 56% / 64% | ⏳ | ⏳ |
| ROUGE-L / BERTScore vs MTSamples description* | 0.22 / 0.81 | ⏳ | ⏳ |
| Median latency | 6.4 s | ⏳ | ⏳ |
| Tokens in / out (mean) | 1,436 / 1,912 | ⏳ | ⏳ |

Local models on the first 10 test records (same records for every model) ⏳: see `results/comparison.md`.

\* The MTSamples "description" field is a very weak reference (often just a procedure name), so ROUGE/BERTScore
mostly measure style overlap, not quality. Grounding and the manual review matter more.

### What "unsupported" catches (examples)

- Gemini Flash-Lite quoted "irrigat**ion** with a solution of Bacitracin…" while the note says
  "irrigat**ed** with a solution of…". The claim is true, but the quote is not verbatim, so it is flagged.
  "Unsupported" therefore means *not verifiable from an exact quote*, which is stricter than "false".
- One Flash-Lite one-liner was 345 characters (limit 300) → counted as schema-invalid.

## 5. Limitations

- MTSamples is transcribed US dictation, mostly single short reports; real Indian EMRs will be longer,
  multi-document, partly scanned/handwritten and partly Hindi. With records this short, retrieval barely
  matters (every record fits in the prompt); RAG will matter for multi-visit patient histories.
- No gold summaries: automatic scores are proxies. A blinded manual review sheet is prepared
  (`results/manual_review.csv`, 10 records × each model, scores 1–5 for correctness, omissions, hallucinations).
- Free-tier Gemini was overloaded on the test day (HTTP 503 errors); latencies for 3.8 Flash and 3.5 Flash
  include retry waiting and are not representative of a paid key.
- Gemini Pro was not tested (needs billing).

## 6. Recommendation ⏳

_To be written when all runs finish._

## 7. Data-use rules followed

Only public MTSamples data was sent to Gemini. On the free tier Google may use prompts to improve its products,
so **no credentialed (MIMIC) or real hospital data may ever be sent to Gemini**. Such data must be processed with
local models only; the local-model rows above show what that option costs in quality and speed.
