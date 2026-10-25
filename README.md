# EMR Summarizer

Research prototype for the DTU project **"An App for Transforming Raw EMRs into Actionable Summaries"**
(CCDR / Koshika Foundation; PI: Dr. Anurag Goel, Dept. of CSE, DTU).

The pipeline takes a raw medical report, anonymizes it, stores it in a vector database, and uses an LLM
(Gemini, plus other models for comparison) to write a structured **ActionableSummary** where every clinical
claim points back to the exact text it came from.

> Status: **Phase 0 (setup)**. Sections marked _TODO_ get filled in as later phases land.

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
python scripts/check_gemini.py --hello gemini-flash
pytest
```

## Run

_TODO (Phase 4):_ `python scripts/run_pipeline.py --model gemini-flash --limit 50`

## Repo layout

| Path | What |
|---|---|
| `schemas/` | Input (EMRBundle) and output (ActionableSummary) JSON Schemas |
| `tools/` | `validate.py` (schema + span checks), `assign_patient_id.py` (name → UUID) |
| `examples/` | One hand-written valid input/output pair |
| `src/emr/` | Pipeline code |
| `scripts/` | Command-line entry points |
| `config.yaml` | All model IDs and settings |
| `memory.md` | Running work log: what was done, decisions, open questions |

## Results

_TODO (Phase 6):_ see `results/report.md`.
