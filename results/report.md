# EMR → Actionable Summary: pipeline and first model comparison

DTU project "An App for Transforming Raw EMRs into Actionable Summaries" (CCDR / Koshika Foundation).
Prepared by: Ayush Roy (Research Intern). Date: 25 Oct 2026.
Full metric tables: `results/comparison.md` (50 records) and `results/comparison_first10.md` (10-record subset).

**Status in one line:** the full pipeline (data → anonymization → vector DB → RAG summary → grounding check) works
end to end. Gemini Flash-Lite and a local Gemma model have been compared. Gemini 3.8 Flash, 3.5 Flash and 3.1 Pro
are blocked by free-tier quotas, and Qwen / MedGemma are deferred. Both are listed in §7.

## 1. What was built

One runnable pipeline (`python scripts/run_pipeline.py --model <name>`):

1. **Data**: MTSamples (public, CC0, ~5,000 transcribed reports, 40 specialties) → converted to our input
   schema (EMRBundle). After removing duplicates (the same report is often filed under 2–3 specialties) and
   very short notes: **2,319 unique reports, 100% schema-valid**.
2. **Anonymization**: Microsoft Presidio + spaCy, plus our own rules for Indian identifiers (10-digit / +91
   mobiles, 12-digit Aadhaar-like numbers, MRN/UHID numbers, Indian addresses). Each patient gets a random UUID;
   the lookup table stays in `secrets/` (never committed). All 2,319 bundles anonymized and still schema-valid.
3. **Vector DB (RAG)**: each report is split at its section headings, embedded with `gemini-embedding-001` and
   stored in ChromaDB. Retrieval is per summary section (problems, medications, findings, course, plan,
   allergies) and always filtered to the one patient's own record.
4. **Summary generation**: the model writes a structured ActionableSummary (problems, medication changes, key
   findings, course, disposition, action items, red flags). **Every claim must quote the source text.**
5. **Grounding check (hallucination measure)**: our code searches for each quote in the source. Found → the
   claim gets exact character offsets; not found → the claim is marked **unsupported**. The model never invents
   offsets. Every output is validated against the full output JSON Schema.

All models get **exactly the same retrieved text and the same prompt**; only the model changes.

## 2. Anonymization leak test

60 fake clinical notes (Indian and US names, +91 and 10-digit phones, Aadhaar-like numbers, MRNs, dates,
addresses; generated with Faker, not real people), using a random seed **not** used while building the rules.

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
"2 weeks") were kept 100% of the time. Getting there mattered: out of the box, spaCy labelled words like
"Diverticulosis" and "Anorectal" as person names and "4-6 months" as a date, which would have deleted
clinical facts. We now only accept a name/place if it appears in a names or places list, or follows "Dr.",
"Patient", "Name:" etc.

**What leaked:** US military-style addresses ("FPO AE 99784"), one Indian address with an unusual city name,
one US phone written without separators, one rare Indian name, and a doctor surname that is also a common word
("Date"). **Caveats:** (a) MTSamples is already de-identified, so recall on real data can only be estimated
with synthetic notes; (b) the test names come from the same name lists the anonymizer uses, so name recall is
optimistic; (c) on MTSamples a few clinical eponyms are still removed ("Tanner" stage, "Weber" test);
(d) a single-letter name after "Dr." (e.g. "Dr. X", MTSamples' own placeholder) is not removed. That's a known bug,
fixed in the next anonymization rerun. **Before real hospital data, this needs a test on hand-labelled real notes.**

## 3. Test set and models

- **Test set:** 50 MTSamples reports, ≥ 1,500 characters, spread over 38 specialties (fixed seed, saved in
  `data/test_set.json`). Median length 3,200 characters (~510 words); every report fits in one prompt.
- **Models:**

| Short name | Model | Where it runs | Status |
|---|---|---|---|
| gemini-flash-lite | gemini-3.5-flash-lite | Google API (free tier) | ✅ 50/50 |
| local-gemma | gemma4:e4b (Google open model) | this laptop (Ollama, Apple M4 GPU) | ✅ 10/10 (subset) |
| gemini-flash | gemini-3.8-flash | Google API | ⛔ 2/50, free tier allows 20 requests/day |
| gemini-flash-3.5 | gemini-3.5-flash | Google API | ⛔ 1/50, same limit |
| gemini-pro | gemini-3.1-pro-preview | Google API | ⛔ free-tier quota is 0, needs billing |
| local-qwen, local-medgemma | qwen3:8b, medgemma:4b | this laptop | ⏸ deferred |

## 4. Results

### 4a. Gemini Flash-Lite on all 50 records

| Metric | Value |
|---|---|
| Valid outputs (full schema) | 98% (1 one-liner was 345 chars, limit 300) |
| Claims per summary | 15.0 |
| **Claims backed by an exact quote** | **99.5%** (0.08 unsupported claims per summary) |
| Summaries with problems / medications / action items | 100% / 56% / 64% |
| ROUGE-L / BERTScore of one-liner vs MTSamples description* | 0.22 / 0.81 |
| Median time per record | 6.4 s |
| Tokens per record (in / out) | 1,436 / 1,912 |

(Medications and action items are absent for many records because many MTSamples reports, e.g. radiology
reads and operative notes, genuinely contain none.)

### 4b. Gemini (cloud) vs Gemma (local), same 10 records

| Metric | Gemini Flash-Lite | Gemma 4 e4b (local) |
|---|---|---|
| Valid outputs (full schema) | 100% | 100% |
| Claims per summary | 20.9 | 18.7 |
| **Claims backed by an exact quote** | **100%** | **91.4%** (1.6 unsupported per summary) |
| Summaries with action items | 50% | 100% |
| ROUGE-L / BERTScore vs description* | 0.25 / 0.81 | 0.38 / 0.84 |
| Median time per record | 11 s | 176 s (laptop) |
| Data leaves the machine? | yes (Google) | **no** |

**Why Gemma loses claims:** (1) it fills empty lists with placeholder items ("None", "unknown", "N/A",
"None stated") instead of leaving them empty, so these items have no evidence; (2) it "stitches" quotes, e.g.
`ADMISSION DIAGNOSES: 2. Severe anemia.`, which joins a heading and a list item that are not adjacent in the note.
The facts are usually right, but they can't be verified from an exact quote. Both are fixable: tell the model
explicitly, and drop placeholder items in code. Its 100% action-item coverage partly comes from the same habit
("None stated"). Gemma's one-liners overlap more with MTSamples' own descriptions, but that reference is weak.

\* The MTSamples "description" field is a weak reference (often just a procedure name), so ROUGE/BERTScore
mostly measure wording overlap, not quality. Grounding and the manual review matter more.

### 4c. Does retrieval (the "R" in RAG) matter?

MTSamples reports are short, so by default the pipeline passes the whole record (nothing can be missed). To
test RAG properly we re-ran Flash-Lite with **only the top-2 retrieved chunks per summary section**:

| | Whole record | Retrieved chunks only |
|---|---|---|
| Share of the record sent to the model | 100% | 82% on average (68% on the 29 records where retrieval dropped text) |
| Input tokens per record | 1,436 | 1,194 (−17%) |
| Claims backed by an exact quote | 99.5% | 99.4% |
| Valid outputs | 98% | 100% |
| Summaries with action items | 64% | 60% |

Retrieval cut the input by ~17% with almost no loss in grounding. The small drop in action items shows the
risk: if the "plan" chunk is not retrieved, its follow-ups are lost. On long, multi-visit patient records (which
do not fit in one prompt) retrieval becomes necessary, and this is where it should be evaluated next.

### 4d. Example disagreements

`mtsamples:3054` (nephrology, acute kidney failure):
- Flash-Lite: *"68-year-old Korean man with CAD, heart failure, diabetes, and stage III CKD presented with acute
  on chronic renal failure, orthostatic hypotension, and elevated CK, likely secondary to supplements,
  antihypertensives, and statin/fibrate therapy."*: includes the likely causes, which matter for action.
- Gemma: *"68-year-old male with a history of CAD, HTN, DM, and stage III CKD presenting for evaluation of acute on
  chronic kidney failure and orthostatic hypotension."*: correct but omits the causes and the CK finding in the
  one-liner, and its problem list has 4 items vs 8–9 for Flash-Lite (it leaves out CAD, hypertension, diabetes).

`mtsamples:1308` (sleep-study follow-up): the retrieved-only run missed the main problems (insomnia, fatigue)
because the chunk with the sleep-study conclusion was not retrieved. This is a concrete example of the RAG risk above.

More in `results/comparison.md` / `comparison_first10.md`.

## 5. Recommendation (preliminary)

1. **Gemini is good to go for open/public data.** Even the cheapest tier (Flash-Lite) produced schema-valid,
   almost fully grounded summaries (99.5% of claims verifiable by exact quote) in ~6 s per record.
   Before choosing between Flash-Lite, Flash and Pro we need the remaining runs, which require a **paid key**
   (free tier: 20 requests/day for Flash, 0 for Pro). Expected cost for the 50-record test set is small,
   since each record uses ~1,500 input and ~2,000–6,000 output tokens. Check current prices at ai.google.dev/pricing.
2. **Gemini cannot be used for real hospital data or credentialed MIMIC** (data-use rules). For that, a local
   model is required. Gemma 4 (4B) already works on a laptop with 91% grounding but needs prompt fixes and is
   slow (~3 min/record on an M4 laptop; a GPU server would be much faster). Next step: compare Qwen3-8B and
   MedGemma on the same 10 records, and fix the two Gemma failure patterns.
3. **Keep the grounding check in the product.** It catches exactly the cases a clinician must not trust and
   gives every claim a clickable link to its source text.

## 6. Limitations

- MTSamples is transcribed US dictation, mostly single short reports; real Indian EMRs will be longer,
  multi-document, partly scanned/handwritten and partly Hindi.
- No gold summaries: automatic scores are proxies. A **blinded manual review sheet** is ready
  (`results/manual_review.csv`, models hidden behind letters; key in `manual_review_key.csv`). It needs a
  clinician or the team to score correctness, omissions and hallucinations (1–5).
- "Unsupported" means *not verifiable by an exact quote*. That's stricter than "false": e.g. Flash-Lite quoted
  "irrigation with a solution of…" where the note says "irrigated with a solution of…".
- Local-model comparison is on 10 records only so far; latencies are laptop numbers.
- Gemini runs on the test day hit HTTP 503 "high demand" errors on 3.8 Flash; latencies there are not representative.

## 7. Next steps

| Step | Needs |
|---|---|
| Run Gemini 3.8 Flash, 3.5 Flash, 3.1 Pro on all 50 records | paid Gemini key (or ~3 days of free quota for Flash; Pro impossible on free tier) |
| Run Qwen3-8B and MedGemma-4B (10 → 50 records) | ~3 h laptop time each for 50 records; no cost |
| Fix Gemma placeholders / stitched quotes; fix "Dr. X" anonymizer rule and re-run | ~1 day |
| Fill the manual review sheet | clinician / team time (~2 h) |
| Test on long multi-document records (RAG actually needed) | ACI-Bench or synthetic multi-visit histories |

## 8. Data-use rules followed

Only public MTSamples data was sent to Gemini. On the free tier Google may use prompts to improve its products,
so **no credentialed (MIMIC) or real hospital data may ever be sent to Gemini**; such data must be processed with
local models only.
