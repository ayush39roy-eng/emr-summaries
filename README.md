# EMR Summarizer

Research prototype for the DTU project **"An App for Transforming Raw EMRs into Actionable Summaries"**
(CCDR / Koshika Foundation; PI: Dr. Anurag Goel, Dept. of CSE, DTU).

The pipeline takes a raw medical report, anonymizes it, stores it in a vector database, and uses an LLM
(Gemini, plus other models for comparison) to write a structured **ActionableSummary** where every clinical
claim points back to the exact text it came from.

> Status: pipeline working end to end; model comparison in `results/report.md`.

## Data rules (read before running anything)

- **Only open/public data goes to Gemini** (MTSamples, MIMIC-IV demo, ACI-Bench).
- On the Gemini **free tier**, Google may use prompts to improve its products and humans may review them,
  and the terms say not to send personal or sensitive data. Prefer a **paid** API key, and even then send
  only open data.
- **Never** send credentialed data (MIMIC-IV-Note, MIMIC-IV-ED full, MIMIC-IV-Ext-BHC) or any real hospital
  data to Gemini or any other third-party API. PhysioNet's data-use agreement forbids it. Such data may be
  processed with **local models only** (Ollama).
- Real names / MRNs never appear in bundles, chunks, prompts, logs or outputs. The name → UUID lookup lives
  in `secrets/`, which is git-ignored, as are `data/` and `.env`.

## Install

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
uv pip install -e .                  # makes `import emr` work
cp .env.example .env                 # then paste your GEMINI_API_KEY into .env
python -m spacy download en_core_web_lg
python scripts/check_gemini.py --hello gemini-flash
pytest
```

Local models (optional, run on your own machine — nothing leaves it): install [Ollama](https://ollama.com), then
`ollama pull gemma4:e4b`, `ollama pull qwen3:8b`, `ollama pull medgemma:4b`.

## Run

Put the Kaggle file `mtsamples.csv` (dataset `tboyle10/medicaltranscriptions`, needs a Kaggle login) in
`data/raw/`, then:

```bash
python scripts/convert_mtsamples.py          # Phase 1: CSV -> data/bundles/raw (2,319 unique reports)
python scripts/anonymize_bundles.py          # Phase 2: -> data/bundles/anon (Presidio + custom recognizers)
python scripts/leak_test.py --seed 2026      # Phase 2: anonymizer recall on synthetic Indian/US identifiers
python scripts/run_pipeline.py --model gemini-flash --workers 4   # Phases 3-4: test set, index, retrieve, summarize
python scripts/run_pipeline.py --model local-gemma               # Phase 5: same chunks + prompt, other model
python scripts/compare.py                    # Phase 6: results/comparison.{csv,md}, manual_review.csv
python scripts/retrieval_demo.py             # ask the vector DB questions about one record
```

RAG experiment (only the top-k retrieved chunks per summary section are sent, instead of the whole record):
`python scripts/run_pipeline.py --model gemini-flash-lite --context retrieved --k 2` → `results/gemini-flash-lite__retrieved/`.
For a step-by-step walk through one record, open `notebooks/demo.ipynb`.

`run_pipeline.py` options: `--limit N`, `--dry-run` (no API calls, prints the prompt), `--redo`, `--workers N`.
It resumes automatically (records with a result file are skipped) and every LLM/embedding call is cached in
`data/cache/`, so reruns are free. Model short names (`gemini-flash`, `gemini-flash-3.5`, `gemini-flash-lite`,
`gemini-pro`, `local-gemma`, `local-qwen`, `local-medgemma`) are defined in `config.yaml`.

### How the pipeline works

1. **Convert**: each MTSamples row → an EMRBundle (`schemas/emr_input.schema.json`). Duplicate reports are merged.
2. **Anonymize**: Presidio + spaCy find names, dates, phones, places, IDs (incl. Indian mobile, Aadhaar-like
   numbers, MRN/UHID) and replace them with tags like `<PERSON>`. Each patient gets a random UUID.
3. **Chunk**: split each report at its ALL-CAPS headings (`HISTORY OF PRESENT ILLNESS:`); every chunk keeps
   exact character offsets into the anonymized text.
4. **Embed + store**: chunks → Gemini embeddings → ChromaDB (`data/chroma/`), with record/patient metadata.
5. **Retrieve**: per summary section (problems, meds, findings, course, plan, allergies), only from this
   record. Short reports are passed whole. The retrieved chunks are cached so every model sees the same input.
6. **Generate**: the model fills a simplified JSON shape where each claim has `{chunk_id, quote}` evidence.
7. **Ground + validate**: code finds each quote in the source and converts it to `{doc_id, start, end}`;
   claims whose quotes can't be found are marked unsupported. Output is checked against the full schema.

## Repo layout

| Path | What |
|---|---|
| `schemas/` | Input (EMRBundle) and output (ActionableSummary) JSON Schemas |
| `tools/` | `validate.py` (schema + span checks), `assign_patient_id.py` (name → UUID) |
| `examples/` | One hand-written valid input/output pair |
| `src/emr/` | Pipeline code |
| `scripts/` | Command-line entry points |
| `config.yaml` | All model IDs and settings |
| `notebooks/demo.ipynb` | One record end to end, explained step by step |
| `results/` | Summaries per model, comparison tables, leak test, `report.md`, manual-review sheet |
| `memory.md` | Running work log: what was done, decisions, open questions |

## Results (25 Oct 2026, preliminary)

- Anonymization leak test: **98.1%** of 420 synthetic identifiers removed; clinical phrases kept 100%.
- Gemini 3.5 Flash-Lite, 50 records: **98% schema-valid, 99.5% of claims backed by an exact quote**, ~6 s/record.
- Local Gemma 4 (4B, laptop), 10 records: 100% valid, 91.4% grounded, ~3 min/record.
- Retrieval-only context: 17% fewer input tokens, grounding unchanged (99.4%).
- Gemini 3.8 Flash / 3.5 Flash / 3.1 Pro: blocked by free-tier quotas (need a paid key). Qwen3 / MedGemma: deferred.

See `results/report.md` (summary for the team), `results/comparison.md` (full tables) and
`results/leak_test_seed2026.md` (anonymization).
