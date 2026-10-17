# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Rakshaka AI is a TensorForge 2.0 hackathon entry, theme "Scam-Proof by Design". It flags scam SMS and WhatsApp messages for Sri Lankan users in English, Sinhala, Singlish, Tamil and Tanglish, and it detects and responds to drift when new scam tactics appear (for example fake SL-UDI digital ID scams). Judges weight deployment, monitoring and drift response more heavily than raw accuracy, and they review commit history, so make small, frequent commits. The competition rules are in `TensorForge 2.0 Grand Finale.pdf`, which is gitignored. `workflow.md` logs the setup steps, measured results and open items. Add new work to it.

## Commands

Use the project venv: `.venv/bin/python`. The system `python3` is pyenv 3.8 and won't work. Docker and CI use Python 3.12. Keep `requirements.txt` exactly pinned. A `model.joblib` pickled with one scikit-learn version may not load under another.

```bash
# Main product (src/ + app/)
python src/make_data.py                          # synthetic data -> data/{train,val,test,shifted}.csv (seed 42)
cd src && python train.py && python evaluate.py  # -> models/{model.joblib,vocab.json,meta.json}, reports/metrics.json
cd src && uvicorn api:app --port 8000            # API; must be run from src/ (imports `features`)
streamlit run app/dashboard.py                   # UI on :8501, run from repo root; API_URL env overrides the API address
./run.sh                                         # all of the above
docker compose up --build                        # API :8000 (/docs) + dashboard :8501

# Tests
python -m unittest discover -s tests
python -m unittest tests.test_src_pipeline.TemplateSplitTests.test_no_template_appears_in_more_than_one_split
```

There is no linter. CI runs `compileall`, the unit tests, the full make_data → train → evaluate pipeline and `docker build`. The Docker image generates the data and trains the model at build time.

## Architecture

There are two pipelines that share the `rakshaka/` library.

**1. The main product: `src/` and `app/`, trained on synthetic data.**
- `src/make_data.py` generates data from hand-written templates (`SCAM`, `HAM`, `SHIFT_SCAM`, `SHIFT_HAM`), with labels `scam` and `legitimate` and with fake placeholders such as `DemoBank`, `07X-XXXXXXX` and `*.example`.
  - **Each template is one group.** `split_templates` keeps whole templates out of training, stratified by (label, language), so test and validation scores measure performance on unseen wording.
  - `val.csv` is used only to measure the drift baselines. `test.csv` is the held-out test set that gets reported. `shifted.csv` contains *new tactics* for the drift demo.
  - `data/extra_{train,test}.csv` (columns `text,label[,language]`, with labels 0/1 or named) are appended if present.
- `src/train.py` builds `make_classifier(class_weight="balanced")` and adds the human labels from `data/reviewed.csv`. It writes three files to `models/`:
  - the artifact `{classifier, model_version, languages}`, which has the same format as `scripts/train_model.py` produces
  - `vocab.json`, the word vocabulary used for the unknown-token rate
  - `meta.json`, which holds the version (`v1`, `v2`, … incremented from the previous meta file) and the baselines measured on **val**: `mean_conf`, `unk_rate`, `oov_rate`
- `src/api.py` is the FastAPI service. It calls `rakshaka.service.predict()` and logs every prediction to SQLite at `logs/predictions.db`.
  - **Privacy:** raw text is stored only for messages routed to human review; all others keep only a hash. Keep it that way, in line with the PDPA.
  - **Drift signal:** a rolling window of 50 predictions, judged once it holds 20. It fires if mean confidence drops more than 0.10 below the val baseline, or the unknown-token rate rises more than 15 points above it.
  - **Response:** drift, confidence below 0.60, or a per-message unknown-token rate of 0.50 or more routes the message to `human_review`, with the verdict "UNCERTAIN - be careful, sent for review". The verdict never says "safe" in that case.
  - An alert is written once per drift episode. `POST /review` appends labels to `data/reviewed.csv`. `POST /retrain` runs `train.py` as a subprocess and hot-reloads the model.
  - The request body is `{"text", "language"}`. Note that the legacy server uses `message`.
- `app/dashboard.py` is the Streamlit UI. It has three tabs: check a message; monitoring, where "Simulate traffic" sends the test or shifted set; and the human review queue with a save + retrain button.

**2. Optional public-dataset pipeline: `scripts/`, writing to `artifacts/`.**
- It trains on two English SMS datasets that are not redistributed (`spam.csv` and the `SMS PHISHING DATASET .../Dataset_5971.csv` folder) plus *human-verified* translations from `data/annotation/translation_sample.csv`. Each translation row needs status `verified`, the text and a reviewer. The code never translates automatically.
- Messages are grouped by the sha256 of their normalized text (`group_id_for`). Duplicates across the two sources merge, a conflicting label is a hard error, and the split is by group.
- `scripts/serve.py` is the legacy standard-library HTTP server.

**Shared code in `rakshaka/`:**
- `modeling.py` builds the `FeatureUnion` of character TF-IDF (2–5 grams, which covers Sinhala and Tamil scripts and romanized text) and word TF-IDF (1–2 grams), followed by `LogisticRegression`.
  - `preprocess_text` appends `urltoken`, `emailtoken` and `phonetoken` markers and must be used at both training and inference.
  - `char_ngram_oov_ratio` is the secondary drift signal. It is logged as `char_oov_ratio`.
- `service.py` contains `predict()` and `LANGUAGE_VALUES`.
- `data.py` handles labels and loading. `normalize_label` accepts 0/1, `scam`/`legitimate`, and `ham`/`spam`/`smishing`.
- **Adding a language** means updating `LANGUAGES` and `ANNOTATION_FIELDS` in `data.py`, `LANGUAGE_VALUES` in `service.py`, and `LANGUAGES` in `app/dashboard.py`. Templates for the new language go in `make_data.py`.

**Known weak spot:** on the test set, Sinhala balanced accuracy is 0.50, because every held-out Sinhala legitimate template is classified as scam. More Sinhala legitimate data is the first fix, and `data/PROMPTS.md` holds the generation prompt for it.
