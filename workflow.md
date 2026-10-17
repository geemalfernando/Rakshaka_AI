# Workflow Log

This log records what was done to set up, run, improve and test Rakshaka AI on 17 October 2026, during the TensorForge 2.0 build window, working with Claude Code (Claude Opus 5.5). It also notes what was learned along the way. Times are local (UTC+5:30). Sections are in time order. Older sections describe the state at that moment; later sections say when something was superseded.

## At a glance (state at about 11:20)

- **Product:** one Streamlit dashboard (`app/dashboard.py`) on top of a FastAPI service (`src/api.py`). The model is TF-IDF character + word n-grams with logistic regression, trained on synthetic English, Sinhala, Singlish, Tamil and Tanglish messages.
- **Run it:**
  - Without Docker: `./run.sh`
  - With Docker: `docker compose down -v && docker compose up --build`
  - On this Mac the local `.env` moves the API to port **8001**, because 8000 is taken. The UI is at http://localhost:8501.
- **Clean v1 results** (held-out test): balanced accuracy 0.926, macro F1 0.946, 0 scams missed, 11 of 74 legitimate messages flagged (all of them Sinhala water-bill copies). Sinhala 0.63, the other languages 1.00. The drift alarm fires on the shifted set: unknown-token rate 38.1% against an alert line of 30.7%.
- **Model report:** `reports/model_report.pdf`, built by `reports/build_model_report.py`.
- **Live model right now:** v3, trained with 38 review labels from demo runs. One of those labels is wrong (section 18).
- **What's left:** see section 18, "Open items".

| # | Section | Time |
|---|---|---|
| 0 | Starting point: reading the project and the rules | ~09:20 |
| 1–7 | First pull, dependency fix, English pipeline, legacy API, first tests, CLAUDE.md | ~09:20–09:45 |
| 8 | Second pull: the FastAPI + Streamlit product | ~09:50 |
| 9 | Third pull: the `app.py` UI, a requirements merge bug | ~10:00 |
| 10 | Sinhala, Tamil, Singlish and Tanglish training data | ~10:15 |
| 11 | One UI: dashboard plus charts | ~10:30 |
| 12 | Clean reset before the demo | ~10:40 |
| 13 | Removed the old UI, a CI YAML bug | ~10:45 |
| 14 | Model report PDF and model comparison | ~11:00 |
| 15 | One-command local run (`run.sh`) | ~11:10 |
| 16 | Docker port clash, `.env` | ~11:15 |
| 17 | Pushing to GitHub, CI billing lock | ~10:55 |
| 18 | Open items, consolidated | ~11:20 |
| 19 | Final pull: impact statement and README reorder | ~11:30 |
| 20 | Acting on an external review: safe retraining, independent Sinhala/Tanglish test | ~11:45 |
| 21 | Reviewer checklist: auth, rollback, privacy, drift tests, required labels | ~11:50 |
| 22 | Redeploy with Docker | ~11:55 |
| 23 | Pull of PR #3-#5; main's dashboard was crashing, rebuilt | ~12:05 |
| 24 | Public vs admin split; server-side drift simulation; rate limit | ~12:15 |

## 0. Starting point: reading the project and the rules (about 09:20)

- **Project read in full:**
  - `rakshaka/data.py`, `modeling.py` and `service.py`
  - `scripts/prepare_data.py`, `train_model.py`, `monitor_drift.py` and `serve.py`
  - `tests/test_pipeline.py`, the README, the data folders and `artifacts/metrics.json`
- **Competition rules** (`TensorForge 2.0 Grand Finale.pdf`): a 6-hour build (08:00–14:00) from an empty repository, with the last commit before the deadline judged. The theme is "Scam-Proof by Design": detect scams **and** notice and respond to drift.
  - Required: a running service started from documented commands (a container is preferred), a reproducible pipeline with exact dependency versions, held-out test results in the README, prediction logging plus a drift signal demonstrated on a shifted set, and a drift response.
  - Also required: an architecture diagram, a list of data sources and licences, the AI tools used, a one-page impact statement, a pitch deck as a PDF, and an open licence.
  - Judges weight deployment, monitoring and drift response more heavily than accuracy, and they review commit history.
- **State at that point:**
  - an English-only TF-IDF + logistic regression pipeline on the two public SMS datasets
  - a standard-library HTTP API
  - an empty drift CSV and 200 translation rows, all still `pending`

## 1. Syncing with GitHub

- `git pull` first failed with *"You have not agreed to the Xcode license agreements"*. On macOS, git won't run until the license is accepted:
  ```bash
  sudo xcodebuild -license accept
  ```
- Before the license was accepted, the GitHub API was used to compare local and remote main:
  ```bash
  curl -s 'https://api.github.com/repos/geemalfernando/Rakshaka_AI/compare/<local_sha>...main'
  ```
- After the license was accepted, local main fast-forwarded from `6d9a4be` to `7a65a1e` (the merge of PR #1, `feat/tamil-tanglish-drift-deployment`). The pull brought in:
  - Tamil and Tanglish annotation columns, and API support for the language values `tamil` and `tanglish`
  - reviewed annotations are now kept when `prepare_data.py` is run again
  - `scripts/demo_drift.py`
  - `Dockerfile` and `.dockerignore`
  - the CI workflow `.github/workflows/tests.yml`
  - `tests/test_multilingual.py`
  - fixed versions in `requirements.txt`

## 2. Fixing the dependency versions (superseded by section 8)

**Problem:** PR #1 pinned `scikit-learn==1.5.2` and `joblib==1.4.2`, and the Docker image and CI used Python 3.11. The local `.venv` runs Python 3.14 with scikit-learn 1.9.1, so the pinned versions didn't match what's installed. There was a second risk: a `model.joblib` pickled with one scikit-learn version may fail or behave differently when loaded with another.

**Fix:** pin exactly what the venv runs, and move Docker and CI to the same Python version.

| File | Change |
|---|---|
| `requirements.txt` | `scikit-learn==1.9.1`, `joblib==1.6.0`, `numpy==2.5.4`, `scipy==1.18.1`, `threadpoolctl==3.7.0` |
| `Dockerfile` | `python:3.11-slim` → `python:3.14-slim` |
| `.github/workflows/tests.yml` | `python-version: "3.11"` → `"3.14"` |

numpy 2.5.4 and scipy 1.18.1 require Python 3.12 or newer, which is why 3.11 had to go.

**Rule from now on:** `requirements.txt`, the venv, the Dockerfile and CI must all use the same versions. After any version change, retrain the model.

## 3. Running the full pipeline

Always use the project venv (`.venv/bin/python`), not the pyenv Python 3.8 on the system path.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train_model.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/demo_drift.py
```

Results:

- **Data preparation:** 6,573 unique messages (5,304 legitimate, 1,269 scam), with 4,970 duplicate rows removed.
  - `spam.csv` had 1,182 non-UTF-8 bytes, which were decoded as Windows-1252.
  - `Dataset_5971.csv` has 185 replacement characters (`�`) whose original text can't be recovered.
  - `translation_sample.csv` was regenerated with the Tamil and Tanglish columns.
- **Training:** model version `20261017T041405Z`. Held-out test balanced accuracy 0.9744, macro F1 0.9802. This is an English-only baseline, because there are 0 verified translations.
- **Tests:** all 10 passed.
- **Drift demo:**

  | Set | Messages | Mean unseen character-ngram ratio | Messages over threshold |
  |---|---|---|---|
  | Baseline | 200 | 0.129 | 11 |
  | Shifted (built in) | 5 | 0.602 | 4 |

  With a threshold of 0.25, the drift alert fired (`drift_alert: true`) and the response was `human_review_and_retraining_recommended`.
- **Caveat:** the baseline is the first 200 cleaned messages, and these overlap with the training data. That makes the baseline ratio look lower than it would on truly new messages.

## 4. Running the API

```bash
.venv/bin/python scripts/serve.py --port 8001
```

- Port 8000 was already in use by an older `serve.py`. That process (PID 63267) was started at 09:07 with pyenv's Python 3.8 and was serving the earlier model `20261017T032837Z`. It was left running, and the new server started on 8001 instead.
  - To check what holds a port: `lsof -nP -iTCP:8000 -sTCP:LISTEN`
  - To free port 8000: `kill <pid>`, then run `serve.py` without `--port`.
- Health check: `curl http://127.0.0.1:8001/health`

## 5. Test predictions

All test messages are synthetic and use `.test` domains, as the competition rules require.

| Message | Language | Prediction | Scam probability | Unseen ratio | Drift alert / human review |
|---|---|---|---|---|---|
| "Hi, are we still meeting at 6 today?" | english | legitimate | 0.18 | 0.14 | no / no |
| "URGENT! You have won a 1000 prize. Claim now at www.example-prize.test" | english | scam | 0.93 | 0.12 | no / no |
| "Your bank account is suspended. Verify now at www.verify-acct.test to avoid closure" | english | scam | 0.75 | 0.18 | no / no |
| "ඔබගේ ගිණුම පරීක්ෂා කරන්න" ("check your account") | sinhala | **legitimate** (missed) | 0.23 | 1.00 | yes / yes |
| Tamil "account frozen, click link" + `www.verify-acct.test` | tamil | scam | 0.79 | 0.83 | yes / yes |
| The same Tamil message without the link | tamil | **legitimate** (missed) | 0.30 | 1.00 | yes / yes |

Tamil test commands:

```bash
# With link: scam, drift alert
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" \
  -d '{"message":"உங்கள் வங்கி கணக்கு முடக்கப்பட்டுள்ளது. உடனே சரிபார்க்க இந்த இணைப்பை கிளிக் செய்யவும் www.verify-acct.test","language":"tamil"}'

# Without link: legitimate (missed), drift alert, human review recommended
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" \
  -d '{"message":"உங்கள் வங்கி கணக்கு முடக்கப்பட்டுள்ளது. உடனே சரிபார்க்க இந்த இணைப்பை கிளிக் செய்யவும்","language":"tamil"}'
```

What the results show:

- **The model doesn't understand Sinhala or Tamil yet.** In the first Tamil message, the "scam" verdict came from the URL marker (`urltoken`), not from the Tamil words. Without the link, the same message is missed.
- **Misses in new languages are flagged, not silent.** Every Sinhala and Tamil message pushed the unseen ratio to 0.83 or higher. Each one triggered `drift_alert` and `human_review_recommended`, and a `drift_response` event was written to `artifacts/predictions.jsonl`. Message text is never logged.
- **This is the demo story for the judges:** the model misses a scam in a language it hasn't seen, and drift monitoring flags it for human review.

## 6. Documentation added

- **`CLAUDE.md`** is guidance for future Claude Code sessions. It covers:
  - the commands, including how to run a single test
  - the rule that versions must match across `requirements.txt`, Docker and CI
  - the architecture: messages grouped by a hash of their text, splits by group, rules for verified translations, the drift and review thresholds, and logging that omits message text
  - the checklist of places to edit when adding a language
- **`workflow.md`** is this file.

## 7. Open items

Open items were tracked at the end of sections 8, 9 and 10 as work went on. Section 18 is the current, consolidated list.

## 8. Second pull: the new FastAPI and Streamlit product (about 09:50)

- `git fetch` showed 7 new commits on main, ending at `9b7e51b` ("Merge READMEs and add impact statement and licence templates").
- **Conflict avoided:** my local edits to `requirements.txt`, `Dockerfile` and `tests.yml` (section 2) touched the same files as the new commits. The remote versions already fixed the mismatch: scikit-learn 1.9.1 and joblib 1.6.0, with **Python 3.12** in Docker and CI. So I discarded my edits (`git restore`), then ran `git pull --ff-only` and `pip install -r requirements.txt`.
- **What changed:**
  - `src/make_data.py`: a synthetic multilingual template generator. Train, val, test and shifted sets are split by template.
  - `src/train.py` and `src/evaluate.py`: drift baselines measured on the validation set, results written to `reports/metrics.json`.
  - `src/api.py`: a FastAPI service with SQLite logging, a rolling-window drift monitor, a review queue and a `/retrain` trigger with hot reload.
  - `app/dashboard.py`: a Streamlit UI.
  - Also new: `docker-compose.yml`, `run.sh`, `IMPACT.md` (a template), `LICENSE` (MIT), `data/PROMPTS.md`, and `tests/test_src_pipeline.py`.
  - `scripts/` is now the optional public-dataset pipeline, and `scripts/serve.py` is legacy.
- **Verified locally:**
  ```bash
  .venv/bin/python src/make_data.py
  cd src && ../.venv/bin/python train.py && ../.venv/bin/python evaluate.py
  .venv/bin/python -m unittest discover -s tests     # 13 tests OK
  ```
  | Set | Rows | Balanced acc. | Macro F1 | Mean conf. | Unknown-token rate |
  |---|---|---|---|---|---|
  | val (baseline) | 276 | | | 0.791 | 23.5% |
  | test | 264 | 0.899 | 0.925 | 0.823 | 18.5% |
  | shifted | 152 | 0.950 | 0.970 | 0.828 | **50.4%** |

  These match the README exactly, so the pipeline is reproducible.
- **Running it:** I stopped the legacy server on 8001 and started the FastAPI service there instead. Port 8000 is still held by the old PID 63267.
  ```bash
  cd src && ../.venv/bin/uvicorn api:app --port 8001    # or ../.venv/bin/python -m uvicorn ...
  ```
  The request body is now `{"text": ..., "language": ...}`, not `message`. Test results:

  | Message | Label | Conf. | Verdict | Routed |
  |---|---|---|---|---|
  | Tamil "account frozen, click link" (no URL) | legitimate | 0.58 | UNCERTAIN - be careful, sent for review | human_review |
  | Sinhala "check your account" | scam | 0.56 | UNCERTAIN - be careful, sent for review | human_review |
  | "Hi, are we still meeting at 6 today?" | legitimate | 0.88 | LIKELY SAFE | auto |

  The Tamil scam is still misclassified, but the safe fallback catches it: confidence is below 0.60, so it goes to review instead of being called safe.
  ```bash
  curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" \
    -d '{"text":"உங்கள் வங்கி கணக்கு முடக்கப்பட்டுள்ளது. உடனே சரிபார்க்க இந்த இணைப்பை கிளிக் செய்யவும்","language":"tamil"}'
  ```
- **`CLAUDE.md` rewritten** for the two-pipeline architecture.

### Open items (updated)
- [ ] Commit and push `CLAUDE.md` and `workflow.md`.
- [ ] Fill in the TODOs: README "Who it protects", the data-source licences and AI tools; `IMPACT.md` (who is protected, reach, harm, adoption); the model name in `data/PROMPTS.md`.
- [ ] Fix the weak spot: Sinhala balanced accuracy on test is 0.50 (all Sinhala legitimate messages flagged as scam). Add Sinhala legitimate rows to `data/extra_train.csv`.
- [ ] Rehearse the drift demo in the dashboard: send the test set (OK) → the shifted set (DRIFT) → review → retrain → v2.
- [ ] Test `docker compose up --build`.
- [ ] Make the pitch deck as a PDF.
- [ ] Stop the old server on port 8000 (PID 63267) so the API can use its default port.

## 9. Third pull: the new Streamlit UI (about 10:00)

- I pulled `9791631` (PR #2, `feat/streamlit-ui`). It added:
  - `app.py`, a three-screen Streamlit app: check a message, drift watch, and who this protects
  - `rakshaka/ui_data.py`
  - `.streamlit/config.toml`, the theme
  - `tests/test_ui_data.py`
- **Merge bug fixed locally (not yet committed):** `requirements.txt` ended up with the branch's stale pins (`scikit-learn==1.5.2`, `joblib==1.4.2`) alongside the main ones, plus a second `streamlit` line. pip refused with `ResolutionImpossible`, which also breaks CI. I removed the duplicates so each package is pinned once, at the versions main already uses.
- The README merge also interleaved old and new sections, for example the problem statement now sits under "Monitoring and response". It still needs a tidy-up.
- Tests: all pass.
- **Running services:**

  | Service | Command | URL |
  |---|---|---|
  | Backend (FastAPI, model v1, synthetic data) | `cd src && ../.venv/bin/python -m uvicorn api:app --port 8001` | http://127.0.0.1:8001/docs |
  | Frontend 1: new `app.py` | `.venv/bin/python -m streamlit run app.py --server.port 8501` | http://localhost:8501 |
  | Frontend 2: API dashboard | `API_URL=http://127.0.0.1:8001 .venv/bin/python -m streamlit run app/dashboard.py --server.port 8502` | http://localhost:8502 |

- **Important:** `app.py` does **not** call the FastAPI backend. It loads `artifacts/model.joblib` directly, which is the English-only public-dataset model from `scripts/train_model.py`, and it logs to `artifacts/`. Only `app/dashboard.py` uses the FastAPI service, the `models/` v1 model, the drift window and the review/retrain loop. The team should pick one UI for the demo, or point `app.py` at the API.
- Both apps were smoke-tested with Streamlit's `AppTest`: no exceptions.

### New open items
- [ ] Commit the `requirements.txt` fix, or CI on main will fail at `pip install`.
- [ ] Clean up the merged README.
- [ ] Decide which UI to demo (`app.py` vs `app/dashboard.py`) and which model it uses.

## 10. Sinhala, Tamil, Singlish and Tanglish training data (about 10:15)

- **Goal:** fix the Sinhala weak spot (test balanced accuracy 0.50) and add Tanglish, which had no data anywhere.
- **File:** `data/extra_train.csv` (columns `text,label,language`). `.gitignore` allows it to be committed, and `src/make_data.py` appends it to **train only**. It is generated by a scratch script, `write_extra_train.py`; the texts are hard-coded there.

  | Language | scam | legitimate |
  |---|---|---|
  | sinhala | 24 | 24 |
  | tamil | 24 | 24 |
  | singlish | 23 | 23 |
  | tanglish | 23 | 23 |
  | **total** | | **188** |

- **Rules followed:**
  - Fake placeholders only (`DemoBank`, `TestPay`, `07X-XXXXXXX`, `*.example`).
  - No copies or translations of held-out test or val templates. Excluded, among others: the Sinhala water bill, the Tamil prize, the Tamil "amma I'm coming home", the English SIM/KYC, tax refund, loan insurance fee, card-OTP, Aswesuma, bread and pension-credited templates.
  - None of the shifted tactics (digital ID/QR, USDT/crypto, "new number", voice notes, NIC photo subsidy), so the drift demo still fires.
  - Leakage check: the highest `difflib` similarity between a new row and any val, test or shifted text is 0.62, from shared common words only.
  - Legitimate examples deliberately include bank-like safe messages (an OTP with "don't share", debits, "call the number on the back of your card") so the model doesn't learn that bank-related wording means scam.
- **Results (v3 → v4):**

  | Metric | v3 (before) | v4 (after) |
  |---|---|---|
  | Test balanced accuracy | 0.899 | **0.926** |
  | Test macro F1 | 0.925 | **0.946** |
  | Test Sinhala balanced accuracy | 0.50 | **0.63** |
  | Test confusion matrix | `[[59,15],[0,190]]` | `[[63,11],[0,190]]` |
  | Shifted balanced accuracy | 0.95 | 1.00 |
  | Val baseline unknown-token rate | 22.2% | 18.7% |
  | Shifted unknown-token rate | 50.4% | 38.1% |

  The drift alarm still fires: the shifted unknown-token rate is 19 points above baseline, against a 15-point threshold. The margin was 27 points before. Adding more data that resembles the shifted tactics would stop the alarm firing.
- Tanglish is in training, but test and val have no Tanglish, so it **can't be evaluated yet**. It needs a `data/extra_test.csv` written in different wording.
- I restarted the API on 8001 so it serves v4.
- **Bug found in review data:** `data/reviewed.csv` rows 3 and 5 label the Tamil "your bank account is frozen, click to verify" scam as `legitimate`. The likely cause is that the dashboard's `is_scam` checkbox starts at the model's prediction and the reviewer saved without ticking it. Effect: v4 now calls that exact scam "LIKELY SAFE" with confidence 0.75. Bad human labels poison retraining.
- README data-sources table updated for `extra_train.csv`.

### New open items
- [ ] Fix `data/reviewed.csv` rows 3 and 5 (set them to `scam`), then retrain.
- [ ] Consider making the dashboard review checkbox start unticked, or add a confirmation step.
- [ ] Write `data/extra_test.csv` so Sinhala and Tanglish can be evaluated properly.
- [ ] Commit `data/extra_train.csv`, the README, the `requirements.txt` fix, `CLAUDE.md` and `workflow.md`.

## 11. One UI: merged the dashboard and the app.py charts (about 10:30)

- **Request:** keep a single UI. Use the layout from the dashboard (`app/dashboard.py`, port 8502) and the charts from `app.py` (port 8501), with all data coming from the API.
- **`src/api.py` changes:**
  - New `GET /predictions?limit=N` returns recent prediction metadata from SQLite: rowid, ts, language, label, confidence, unk, oov, model_version, routed. It **never returns message text**.
  - `/monitor` now also returns `thresholds`: `conf_drop`, `unk_rise`, `min_window`, `window`.
- **`app/dashboard.py` changes** (everything comes from the API):
  - Check a message: the verdict, plus three metrics (chance of scam, unseen words, unfamiliar letter patterns). The raw JSON is in an expander.
  - Monitoring & drift:
    - the status banner
    - live-window metrics
    - **two bar charts**, "Baseline / Live window / Alert line", for the unknown-token rate and for mean confidence
    - **a line chart of checks over time** (unknown-token rate, confidence and the alert line)
    - **bar charts** by language and by result (Scam / Legitimate / Sent for review)
    - Simulate traffic and Recent alerts, as before
  - Human review: unchanged.
- The `app.py` UI on 8501 is stopped. The file is still in the repo; deleting it is up to the team.
- Verified with Streamlit `AppTest`: no exceptions. I clicked "Send 50 from test set": status "Model healthy", mean confidence 0.829, unknown-token rate 10%, 5 charts rendered. Unit tests pass.
- **Running now:** API http://127.0.0.1:8001/docs and UI http://localhost:8502.

## 12. Clean reset before the demo (about 10:40)

- **Why:** the dashboard demo had been run a few times, leaving the model at v7 with 83 review labels in `data/reviewed.csv`. Five of those labels were wrong: four Sinhala water-bill messages from the **held-out test set** labelled `scam` (they are legitimate), and one Tamil "account frozen" scam labelled `legitimate`. Wrong labels poison retraining, and test messages in training leak into evaluation.
- **Backup first:** `backups/pre-demo-reset-20261017-103939/` holds reviewed.csv, the model, meta.json, vocab.json, predictions.db, service.log and metrics.json. `/backups/` is added to `.gitignore`.
- **Reset:**
  ```bash
  rm data/reviewed.csv models/meta.json logs/predictions.db   # meta.json removed so the version restarts at v1
  cd src && python train.py && python evaluate.py
  ```
  Then I restarted the API on 8001.
- **Clean v1** (synthetic data + `extra_train.csv`, no review labels):

  | Set | Balanced acc. | Macro F1 | Mean conf. | Unknown-token rate | Confusion matrix |
  |---|---|---|---|---|---|
  | val (baseline) | | | 0.828 | 18.7% | |
  | test | 0.926 | 0.946 | 0.823 | 12.3% | [[63,11],[0,190]] |
  | shifted | 1.000 | 1.000 | 0.799 | **38.1%** | [[20,0],[0,132]] |

  Per language on test: English 1.00, Singlish 1.00, Tamil 1.00, Sinhala 0.63. **These are the honest numbers for the README.**
  Drift should still fire on shifted traffic: the alert line is 18.7% + 15 = 33.7%, and shifted messages average 38.1%.
- **Clean state:** `/monitor` WARMING_UP 0, `/predictions` empty, `/alerts` empty, `/review_queue` empty. The UI on 8502 is up.
- **To reset again before the pitch:** stop the API, back up and delete the same three files, retrain, then start the API.
- **During the live demo:** in the review tab, **tick or untick every row deliberately**. The checkbox starts at the model's guess.

## 13. Removed the old UI; single UI on 8501 (about 10:45)

- Deleted `app.py`, `rakshaka/ui_data.py` (used only by `app.py`) and `tests/test_ui_data.py` with `git rm`. They're recoverable from git history (PR #2). `.streamlit/config.toml` stays; its theme also applies to the dashboard.
- **Merge bug fixed:** `.github/workflows/tests.yml` had two `run:` keys in the "Compile Python files" step, which is invalid YAML, so CI would fail. Kept `compileall -q rakshaka scripts tests src app`.
- Removed the `app.py` mentions from the README and `CLAUDE.md`. Tests: 13 OK.
- **Running now:**

  | What | URL |
  |---|---|
  | UI (`app/dashboard.py`) | http://localhost:8501 |
  | API (`src/api.py`, clean v1) | http://127.0.0.1:8001/docs |

  ```bash
  cd src && ../.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8001
  API_URL=http://127.0.0.1:8001 .venv/bin/python -m streamlit run app/dashboard.py --server.port 8501
  ```
  The API uses 8001 because port 8000 is taken on this machine (an old pyenv `serve.py`, PID 63267, and another project's Docker container).

## 14. Model report PDF (about 11:00)

- **Output:** `reports/model_report.pdf` (6 pages). Built by `reports/build_model_report.py` (`.venv/bin/python reports/build_model_report.py`), which writes the charts to `reports/figures/`. It needs `matplotlib` and `reportlab`, installed in `.venv` only; they are not in `requirements.txt` because the product doesn't need them.
- **Model comparison:** run once, saved to `reports/model_comparison.json`. The same preprocessing for every model. Each was scored by 5-fold `GroupKFold` cross-validation by template on train+val+test, by the held-out test of the English public datasets, by the shifted set, by confidence on val vs shifted, and by ms per message.

  | Model | CV bal. acc. | English bal. acc. | Shifted | ms/msg |
  |---|---|---|---|---|
  | Naive Bayes | 0.891 | 0.970 | 0.884 | 0.11 |
  | LogReg, words only | 0.926 | 0.970 | 0.977 | 0.21 |
  | Random forest | **0.970** | 0.959 | 0.958 | 25.5 |
  | Linear SVM, calibrated | 0.957 | **0.982** | **0.992** | 1.31 |
  | **LogReg char+word (ours)** | 0.936 | 0.974 | 0.973 | 0.57 |

  Keyword rules (no ML) reached 0.724 on the test set.

  **Honest conclusion:** our model is not the most accurate. We kept it for speed (about 45 times faster than the random forest), built-in probabilities (the API's routing and drift monitor depend on them), consistency across both datasets (the random forest is worst on the English data), Sinhala/Tamil/romanised handling through character n-grams, and fast retraining. The calibrated SVM is the recommended next experiment. Transformers were not benchmarked.
- The report uses the **clean v1** numbers. It reads the drift baseline from `reports/metrics.json` (saved by `evaluate.py`), not from the live `models/meta.json`. Note that the API had been retrained to **v2** by a later demo run, with 37 review labels.
- **Corrected a claim:** the README said `extra_train.csv` was "reviewed by the team". It now says "not yet checked by native speakers".
- **Update after merging the teammate's commits:** they lowered `UNK_RISE` from 0.15 to 0.12 in `src/api.py`. The report builder now reads `WINDOW`, `MIN_WINDOW`, `CONF_DROP`, `UNK_RISE`, `LOW_CONF` and `MSG_UNK` straight from `src/api.py`, so its alert line (now 30.7%) always matches the API. `.gitignore` has `!/reports/model_report.pdf` so the report can be committed.

## 15. One-command local run (about 11:10)

- **Problems found:** `python` in a normal terminal here is pyenv 3.8, which can't install the pinned packages (they need 3.12+). The API exits if no model has been trained. `run.sh` wasn't executable and used the system Python. Port 8000 is taken on this machine.
- **`run.sh` rewritten:**
  - creates `.venv` with Python 3.12+ if it's missing
  - installs `requirements.txt` only when its hash changes
  - checks that `API_PORT` (default 8000) and `UI_PORT` (default 8501) are free, and if not, prints who holds them and how to choose other ports
  - generates data and trains only when something is missing (`RETRAIN=1` forces it)
  - starts the API, waits for `/health`, then starts the dashboard; a `trap` stops the API on exit or Ctrl+C
  - made executable, including in git (`git update-index --chmod=+x`)
- **Tested:** the default ports fail cleanly because 8000 is busy. `API_PORT=8001 UI_PORT=8501 ./run.sh` brought up the API (v3) and the dashboard, and a Tanglish scam was predicted `SCAM`. Stopping the script released both ports.
- README "Streamlit app" and "How to run" now give `./run.sh` and working manual steps, using the venv.
- **Docker was not started.** The safety check blocked stopping PID 63267 (the old pyenv `serve.py`) and the other project's `secure-email-service` containers that hold port 8000. The user has to stop those, or run on other ports.

## 16. Docker port clash (about 11:15)

- `docker compose up` failed: `Bind for 0.0.0.0:8000 failed: port is already allocated`. The other project's `secure-email-service-ses-1` container holds port 8000.
- `docker-compose.yml` host ports are now configurable, `${API_PORT:-8000}` and `${UI_PORT:-8501}`, matching `run.sh`. Inside Docker the dashboard still reaches `http://api:8000`. Run with `API_PORT=8001 docker compose up --build`. README updated.
- **Follow-up:** the user ran plain `docker compose up --build` (no `API_PORT`), which still bound 8000, and ran `streamlit run app/dashboard.py` from inside `src/` ("File does not exist"). Fix: added a local, gitignored `.env` with `API_PORT=8001` and `UI_PORT=8501`. Docker Compose reads it automatically, so the default stays 8000 for everyone else. `run.sh` now reads `.env` too, and variables set in the shell still win; this was tested with none, `UI_PORT=8502` and `API_PORT=9000`. I fixed a `set -e` bug in the loader (`|| true`). `../run.sh` from inside `src/` brought the API up on 8001 and the UI on 8501, and stopping it released both ports.

## 17. Pushing to GitHub (about 10:55)

- **Team commits along the way:** the user committed and pushed parts of this work themselves:
  - `17411e0` "fix ui" (requirements fix, `CLAUDE.md`, `workflow.md`)
  - `c4015c2` "fix all the ui" (dashboard charts, `/predictions`, `extra_train.csv`)
  - `6b2c873` "update readme" (old UI removal, README; local only at first)
- **Push request:** local main was 1 ahead and 4 behind. The teammate (ThisukaMW) had pushed:
  - "Reports"
  - a metrics merge
  - "Fix duplicate run key that made the CI workflow invalid"
  - "Recalibrate drift threshold": `UNK_RISE` went from 0.15 to **0.12**, and the README results were updated to the clean v1 numbers
- **Merge `34f280a`:** two conflicts.
  - `.github/workflows/tests.yml`: main still compiled `app.py`, which our side had deleted. Resolved to `compileall -q rakshaka scripts tests src app`.
  - `reports/metrics.json`: both sides held the same clean v1 results, differing only in the training timestamp. Kept main's copy.
- **`3dcc990`:** the model report (PDF, builder, figures) and the `.gitignore` exception `!/reports/model_report.pdf`. The report now reads its thresholds from `src/api.py`, so its alert line became 30.7%.
- **`38b7bb7`:** the `workflow.md` log. Then `git push origin main` (`665a3a7..38b7bb7`).
- **CI:** every recent run on main fails within seconds with *"The job was not started because your account is locked due to a billing issue."* No step runs, so this is not a code failure. Fix billing in the GitHub account settings, then re-run from the Actions tab. Locally `compileall` and all 13 tests pass.

## 18. Open items, consolidated (about 11:20)

**Done** (from earlier lists): pin dependencies consistently (Python 3.12 in Docker and CI), fix the `requirements.txt` merge bug, fix the CI duplicate `run:` key, pick one UI, add Sinhala, Tamil, Singlish and Tanglish training data, write the model report, make the app start with one command, commit and push `CLAUDE.md` and `workflow.md`.

**Still open:**
- [ ] **Commit and push** the current local changes: `run.sh` (venv, port checks, `.env`), `docker-compose.yml` (configurable ports), README run instructions and this `workflow.md`. `.env` stays local; it is gitignored.
- [ ] **Fix the wrong review label:** `data/reviewed.csv` row 39, the Sinhala "Digital ID notice: go only to the official office" message (shifted-set, **legitimate**), is saved as `scam`. The live model (v3) was trained with it. Fix the row and retrain, or reset as in section 12 before the pitch.
- [ ] **Stop wrong labels at the source:** make the dashboard's `is_scam` review checkbox start unticked or require confirmation. This caused six of the 121 human labels saved today to be wrong.
- [ ] **Get Docker running:** run `docker compose down -v && docker compose up --build`. The local `.env` puts the API on 8001. Alternatively stop the other project's `secure-email-service` containers and the old pyenv `serve.py` (PID 63267) to use port 8000.
- [ ] **Fix CI billing** on the GitHub account and re-run the workflow.
- [x] ~~**Fill the README TODOs:**~~ done by the teammate in section 19. (Kept for the record:) "Who it protects" (line ~92), the licence of the two English datasets (UCI SMS Spam Collection, Mendeley SMS Phishing Dataset), the `extra_test.csv` source, and any other AI tools used. Fill `IMPACT.md` (who is protected, reach, harm prevented, path to adoption) and the model name in `data/PROMPTS.md`.
- [ ] **Tidy the README:** the merge left the problem statement under "Monitoring and response", and the old `scripts/` sections sit above the main product sections.
- [ ] **Data:**
  - have native speakers check `data/extra_train.csv`
  - add more legitimate Sinhala messages (Sinhala test 0.63)
  - write `data/extra_test.csv` so Tanglish can be evaluated
  - complete the 200-row human translation sample (0 verified)
- [ ] **Model:** try the calibrated linear SVM. It had the best English and shifted-set scores in the comparison.
- [ ] **Pitch:** make the deck as a PDF, rehearse the drift demo (test set → shifted set → DRIFT → review → retrain), and record a backup video.
- [ ] **Before the demo:** reset to a clean v1 (section 12) so the charts, alerts and review queue start empty.

## 19. Final pull: impact statement and README reorder (about 11:30)

- `git fetch` showed 2 new commits from the teammate (ThisukaMW):
  - `13b09ad` "Add impact statement with sourced figures": `IMPACT.md`, with CBSL-sourced groups, 727,092 government pensioners, 1,251,522 Aswesuma families and 13.9 million internet users
  - `679aa91` "Reorder README and fill its gaps": the problem statement is now at the top, the old `scripts/` sections are moved down, and **all README TODOs are filled**
- Local uncommitted work (`run.sh`, `docker-compose.yml`, README run instructions, this log) was set aside with `git stash`. Then `git pull --ff-only` to `679aa91`, then `git stash pop`.
- **One conflict, README.md:** the teammate deleted the old top section ("Data workflow", the old "Streamlit app" text), and the local side had edited text inside it. Resolved to the teammate's version. The local `./run.sh` and `API_PORT` Docker instructions in "How to run" merged cleanly.
- **Fixed in the new README:**
  - the "Streamlit dashboard" section still had the commands that failed on this Mac (plain `python -m pip`, an unclear working directory); it now points to `./run.sh`, with manual steps inside the 3.12+ `.venv`
  - added the missing `GET /predictions` endpoint to the API list
- Tests: 13 OK. `run.sh` syntax OK.
- **Still uncommitted:** README, `run.sh`, `docker-compose.yml`, `workflow.md`.

## 20. Acting on an external review (about 11:45)

An external review of the repo listed three critical items: Sinhala false positives, no Tanglish testing, and
retraining that can make the model worse. Its numbers checked out: 95.8% accuracy is 253 of 264, the Sinhala
false positives are 11 of 15 legitimate messages, and `/retrain` deployed anything that trained. The
retraining problem had already happened today: six wrong review labels went straight into live models.

**1. Safe retraining (regression gate)**
- `src/train.py` takes `--out DIR`, `--version vN` and `--no-reviewed`. Versions follow the live `models/meta.json`.
- `rakshaka/modeling.py` has a new `retrain_gate(current, candidate, max_drop=0.01)`. It rejects a candidate if
  balanced accuracy, scam recall or legitimate recall drops by more than 0.01. Unit-tested in
  `tests/test_retrain_gate.py`.
- `src/api.py` `/retrain` now:
  - trains into `models/candidate/`
  - scores the live model and the candidate on `GATE_SETS` = `val.csv` + `test.csv`, leaving out any text in
    `reviewed.csv` (demo traffic had sent test messages into review)
  - promotes the candidate with `os.replace` only if it passes; otherwise keeps the live model
  - logs a `RETRAIN_DEPLOYED` or `RETRAIN_REJECTED` alert
  - returns `deployed`, `candidate_version` and `gate`
- The dashboard's review tab shows deployed / not deployed, with a live vs candidate table.
- **Gating on `val.csv` alone was tried first and failed:** a poisoned candidate passed, because val has only one
  legitimate Sinhala template. With val + test, an offline test caught 2 of 3 simulated bad-label batches. The gate
  is a safety net, not a guarantee.
- **End to end through the real API**, in a throwaway copy of the project on port 8011:
  - poisoned labels (all 20 legitimate shifted-set messages as scam) were REJECTED: legitimate recall
    0.948 → 0.929, v1 stayed live
  - correct labels were DEPLOYED: balanced accuracy 0.968 → 0.973, v2 went live

**2. Independent Sinhala and Tanglish test set**
- `data/extra_test.csv`: 65 messages. Sinhala has 15 legitimate (bills, appointments, notices, family) and
  10 scams (cash-back, child-accident, mistaken-transfer, court-order ...). Tanglish has 20 legitimate and 20 scam.
- Written before touching training data, separately from the templates and `extra_train.csv`, with no
  shifted-set tactics. The closest match to any training text is 0.73 (two bill reminders).
- `src/make_data.py` no longer merges `extra_test.csv` into `test.csv`. `src/evaluate.py` scores it as a separate
  `independent` set, and takes an optional model-folder argument. The set is **not** in the retrain gate.

**3. Sinhala false positives: diagnosed, not tuned away**
- Clean model on the independent set: **Sinhala 1.00** (15 of 15 legitimate, 10 of 10 scam), **Tanglish 0.97**.
  The one miss is a "boss asks for a gift-card code" message with no link, scam probability 0.47, which is routed
  to human review.
- So all 11 test false alarms are **one held-out water-bill template**. Its "pay via the official app" wording
  overlaps with Sinhala scams, and the template test has only one legitimate Sinhala template. Adding bill-style
  Sinhala messages to training would tune the model to the test set, so it was deliberately not done.

**Results and docs**
- `reports/metrics.json` was regenerated from a clean v1 (`train.py --out <scratch> --version v1 --no-reviewed`,
  then `evaluate.py <scratch>`), matching CI:
  - test 0.926 / F1 0.946
  - shifted 1.000
  - **independent 0.983 / F1 0.984**
- The live `models/` (v3) was not touched. The running API must be restarted to get the new `/retrain`.
- README: Evaluation has the independent row and the "one template" explanation; Monitoring has the regression
  gate; the data-sources row for `extra_test.csv` is updated. `CLAUDE.md` describes the retrain flow.
- Model report: the stale "Tanglish cannot be scored yet" is replaced by the independent results; the gated
  retrain is described; next steps are updated.
- Tests: 17 OK.

**Not done (from the review), for the pitch Q&A:**
- reviewer authentication on `/review`, `/retrain` and `/review_queue`
- consent, retention and deletion controls
- an integration test of API + drift + retrain in CI
- automatic language identification
- native-speaker review of all synthetic text
- the pitch PDF

## 21. Reviewer checklist: auth, rollback, privacy, drift tests (about 11:50)

The reviewer's 8-item checklist was written against GitHub and didn't see section 20's local work. Status after
this round:

| # | Item | Status |
|---|---|---|
| 1 | Sinhala misclassification | Diagnosed (one template; independent Sinhala 1.00). Not tuned to the test set. Native-speaker data still needed |
| 2 | Tanglish evaluation | Done: 40 independent messages, 0.97. Native-speaker review still needed |
| 3 | Safe retraining **and rollback** | Done: gate (section 20) + `models/previous/` + `POST /rollback` |
| 4 | Admin auth | Done: `X-Admin-Token` on all admin endpoints |
| 5 | Drift tests | Done: 6 scenario tests in `tests/test_monitoring.py` |
| 6 | Privacy | Done: erase on label or discard, `RETENTION_DAYS` purge, token-protected queue |
| 7 | Per-language confusion matrices | Done: README table (test + independent) |
| 8 | Pitch PDF, demo recording | Team |

**Code**
- `rakshaka/monitoring.py`: `drift_signals()`, the pure drift decision. `src/api.py` `drift_status()` uses it.
- `src/api.py`:
  - `RAKSHAKA_HOME` (test isolation)
  - `ADMIN_TOKEN` + `require_admin`: 401 for a missing or wrong token, 503 if unset (off, not open), compared
    with `hmac.compare_digest`
  - `RETENTION_DAYS` (default 7) + `purge_expired_text()`, run at startup and on every `/review_queue`
  - `/review` sets text to NULL once labelled
  - `DELETE /review_queue/{rowid}` discards a message
  - `/retrain` copies the outgoing model to `models/previous/` before promoting, and candidate versions skip
    numbers used before a rollback
  - `POST /rollback` swaps live and previous, with a `ROLLBACK` alert
- `app/dashboard.py` review tab:
  - an admin-token field
  - **no pre-filled labels**: a "your label" select column, so only explicitly labelled rows are saved. This
    removes the cause of today's six wrong labels.
  - Save is disabled until something is labelled; Discard and Roll back buttons; the live model version is shown
- `run.sh`: reads `ADMIN_TOKEN` and `RETENTION_DAYS` from `.env`, generates a random token if none is set, and
  prints it. `docker-compose.yml` passes `ADMIN_TOKEN` and `RETENTION_DAYS` to the API. The local `.env` has a
  random `ADMIN_TOKEN` (gitignored).

**Tests: 30 pass**
- `tests/test_monitoring.py`: low volume, normal traffic over 20 random windows, new tactics, falling confidence,
  mixed-language windows (20% vs 60% shifted), after a retrain.
- `tests/test_api_admin.py`: the real FastAPI app with a tiny model in a temporary `RAKSHAKA_HOME`. Covers
  401 on all 5 admin endpoints (missing and wrong token), public endpoints open, no text in `/predictions`, text
  erased after labelling, discard, retention, rollback 409 / swap / swap back / alert.

**End to end** (throwaway copy, real API on 8012 + the real dashboard via Streamlit `AppTest`): 50 shifted
messages → DRIFT (19% → 37%); a wrong token was refused; the right token showed 35 queued messages; Save was
disabled with no labels; labelling 35 left 0 texts in the log; the gated retrain deployed v2; rollback went to
v1, then back to v2. All of it was logged as alerts.

**Docs:** the README covers the admin token for Docker, the admin endpoints, the review tab, rollback, privacy,
the end-to-end evidence, the per-language confusion matrices and the 30-test list. `CLAUDE.md` and the model
report are updated.

**Not done:** reviewer accounts (one shared token, not per-person logins), consent text for message submitters,
a CI integration test of the full retrain via subprocess, automatic language identification, native-speaker
review, and the pitch PDF.

## 22. Redeploy with Docker (about 11:55)

- The user's earlier Docker run was already up on 8001/8501 (via `.env`) and already served the new API, but
  might have predated the 11:36 dashboard changes. Rebuilt and restarted this project only:
  `docker compose up -d --build`.
- **Verified:**
  - API http://127.0.0.1:8001 serves model v1, built in the image from synthetic data + `extra_train.csv`
  - UI http://localhost:8501 is up
  - `/review_queue` returns 401 without a token and 200 with the `.env` `ADMIN_TOKEN`
  - `/rollback` returns 409 ("no previous model") as expected on a fresh deploy
  - the Tanglish boss/gift-card message is routed to review as UNCERTAIN (confidence 0.53)
  - drift threshold 12 points
- In Docker the admin token is the `ADMIN_TOKEN` line in `.env`, not a printed one (`run.sh` prints it; compose
  doesn't).

## 23. Pull of PR #3–#5; main's dashboard was crashing (about 12:05)

- **What happened upstream:**
  - The user's `acc6ed5` "fixIng Docker" pushed part of the local work (API, `train.py`, `evaluate.py`,
    `make_data.py`, `run.sh`, docs) but **not** `rakshaka/monitoring.py` or `retrain_gate`. So main's API could not
    import.
  - Teammates then pushed 13 commits:
    - **PR #3** re-added `drift_signals` and `retrain_gate`. Same signatures, plus input validation; reason
      strings use `scam_recall` and similar.
    - **PR #4** put `/predictions` and `/alerts` behind the admin token, added token fields to the dashboard, and
      added `tests/test_admin_routes.py` and `tests/test_monitoring_gate.py`.
    - **PR #5** is the "fraud desk" UI: an altair theme, `app/styles.css`, multi-page navigation with an admin
      login.
- **Pulled** with `git stash -u` → `git pull --ff-only` (to `04dc9c5`) → `git stash pop`.
  - Conflicts: `rakshaka/modeling.py` (kept main's `retrain_gate`) and `app/dashboard.py`.
  - `rakshaka/monitoring.py`: main's version kept.
  - `docker-compose.yml`: merged to a single `environment:` block. Two would have been invalid.
- **Main's dashboard crashed on start:** `NameError: name 'tab3' is not defined`. The PR #5 ← main merge
  (`bb80151`) deleted `render_review`, `render_login`, `render_admin` and the `st.navigation` wiring, and kept
  PR #4's `with tab3:` block. **The dashboard on GitHub main does not run.**
- **The fraud-desk branch also committed a hard-coded password**, `ADMIN_PASSWORD = "scamproof"`. That breaks
  rule 6.2 (no passwords in the repository), and the secret is now in git history.
- **Rebuilt `app/dashboard.py`** from the fraud-desk branch (`6a9639d`), combining all three designs:
  - Login page: the "password" is the **API admin token**, checked by calling `/alerts` with it. 503 means the
    API has no token ("admin disabled"); 401 means wrong. The token is kept only in `st.session_state` and
    cleared on sign-out. No secret lives in the dashboard (PR #4's rule). `ADMIN_PASSWORD` is removed.
  - Monitoring: `/predictions` and `/alerts` are sent with the token (PR #4).
  - Review: a "Your label" select column with no default (only labelled rows are saved), "Model said" shown for
    reference, the gate result (deployed or rejected, with a live vs candidate table), Discard unlabelled rows,
    and Model rollback.
- **Tests: 44 pass.**
  - My tests were aligned with main: gate reason strings; `/predictions` and `/alerts` added to the 401 list;
    the no-text test now sends a token instead of passing on a 401.
  - Main's string-matching dashboard test was replaced by an AST check that **every** `requests` call to an
    admin path sends `headers=`, plus a test that `ADMIN_PASSWORD` and `scamproof` are absent.
- **End to end** (merged API in a throwaway copy on 8013, plus the real page functions through a session-state
  test harness, because `AppTest.switch_page` can't open function pages):
  - public page and a check: OK
  - wrong token refused; right token opened the admin desk (two tabs)
  - 50 shifted messages: unknown-token rate 39%, 5 charts, 37 waiting for review
  - Save disabled with no labels; Rollback present; no exceptions
- README: the endpoint list matches (`/predictions` and `/alerts` are admin), and the test count is 44.
- **Action needed:**
  1. Push soon: GitHub main's dashboard is broken until this merge is pushed.
  2. Treat "scamproof" as a burned password: it's in the history of `6a9639d`. It is no longer used for anything.

## 24. Public vs admin split, server-side drift simulation (about 12:15)

A reviewer's architecture note asked for: a public checker; everything operational behind one authenticated admin
console; `/monitor` admin-only; and **drift simulation moved off the public `/predict`** into an admin endpoint
with its own monitoring state, bounded counts, rate limiting and audit logging. The public page, login and admin
desk existed already (section 23). This round added:

- **`src/api.py`**
  - `/monitor` is admin-only, with `mode=live|simulation`.
  - Separate windows `windows["live"]` and `windows["simulation"]`, with per-source `last_status`.
  - The `score()` helper is shared by `/predict` and `/simulate`.
  - A `source` column (`live` / `simulation`) is added with `ALTER TABLE` if missing, so existing DBs keep working.
  - `/predictions?source=`.
  - `POST /simulate` (admin): `{"dataset": "test"|"shifted", "n": 1..100}` via pydantic `Field(le=100)`; a lock
    (one run at a time); `SIM_MIN_INTERVAL` (default 3 s, then 429); fixed `SIM_SETS` only; a `SIMULATION`
    audit alert; simulated drift logged as `DRIFT_SIMULATION`.
  - `POST /simulate/reset`.
  - `/predict` is rate-limited per client IP (`PREDICT_PER_MINUTE`, default 120, sliding window, then 429), and
    no longer returns the `drift` field (drift statistics are admin-only).
- **`app/dashboard.py`:** the admin desk has 4 tabs:
  - **Monitoring:** live only, plus alerts.
  - **Drift simulation:** the buttons call `/simulate`, with its own status, charts and reset. The client-side
    loop over `/predict` and the CSV reads are removed.
  - **Human review.**
  - **Models:** version, training size, labels used, rollback.
  - Shared `render_window()`, `render_history()` and `get_monitor()` helpers.
- **Tests: 48 pass.**
  - `test_admin_routes.py` also requires `require_admin` on `/monitor`, `/simulate` and `/simulate/reset`; checks
    that every dashboard call to `/monitor` and `/simulate` sends the header; and checks the dashboard never
    simulates through `/predict`.
  - `test_api_admin.py` covers simulation isolation (live `n` unchanged, simulation DRIFT, `source=simulation`
    only, `SIMULATION` + `DRIFT_SIMULATION` alerts, reset), bounds (n=1000 → 422, dataset "train" → 400), spacing
    (an immediate second run → 429), and the `/predict` rate limit ([200, 200, 200, 429, 429] at a limit of 3).
- **End to end** (throwaway copy on 8013, real data, page functions via the harness):
  - public `/monitor` and `/simulate` → 401
  - login → 4 tabs
  - "Send new scam tactics" (50) → simulation window DRIFT (19% → 40%), while the **live window stayed at n=0**
  - 39 messages queued for review; Models tab shows "Live model v1" with rollback
  - alerts: `SIMULATION`, `DRIFT_SIMULATION`; no exceptions
- **Still production-only (not done):** per-user accounts and roles, HTTPS, server-side sessions, and a shared
  rate-limit store (the current limiter is in-memory per API process).

## 25. Local Docker redeploy (about 12:20)

- `docker compose up -d --build`: rebuilt the image and recreated both containers.
- **Verified:**
  - API http://127.0.0.1:8001 serves model v1; UI http://localhost:8501 is up
  - public `/monitor` → 401, with the `.env` token → 200; public `/simulate` → 401
  - `/predict` on a Tanglish bill scam → SCAM 0.78, with no `drift` field
  - admin `/simulate` of 50 shifted messages → 37 sent to review, simulation window DRIFT (19% → 36%); the
    **live window stayed at n = 1** (only the one real check)
  - simulation window reset afterwards
- **Side effect:** that verification run put 37 simulated messages in the review queue and added
  `SIMULATION` / `DRIFT_SIMULATION` entries to the alert log. Use "Discard unlabelled rows" before the demo, or
  reset as in section 12.

## 26. Light theme with dark green; removed "02 · Desk" (about 12:25)

- **Request:** show the admin token; make the UI a light background with dark green; remove the
  "02 · Desk / Claim assessment" heading.
- **Admin token:** the `ADMIN_TOKEN=` line in the local, gitignored `.env`. It was shown to the user. It is never
  committed.
- **`app/styles.css`:** palette variables swapped:
  - `--bg` #f4f7f2, `--surface` #fff, `--text` #10261a, `--muted` #55675b
  - accent `--lime` → dark green #1f5f3a, `--lime-deep` #154a2c, `--teal` #2f7d5c
  - `--amber` #a8650a, `--alert` #b3261e
  - near-black surfaces → white; neon glows → faint dark-green tints; light hairlines → dark hairlines
  - primary button text → white
  - the "screen" spotlight blend → "multiply"
- **`.streamlit/config.toml`:** base light, primary #1f5f3a.
- **`app/dashboard.py`:** altair axis, grid and legend colours, and all series colours mapped to the same palette.
  The "02 · Desk / Claim assessment" heading is removed.
- **Checked visually:** Docker redeployed (`docker compose up -d --build`), then screenshots via Playwright
  (installed in a scratch folder) driving the local Chrome:
  - public page with a Tax-refund check: SCAM 68%, light background, dark green headings and button, no "02 · Desk"
  - login page
  - admin Drift simulation tab: DRIFT alert (simulation) 19% → 37%, readable charts in grey / dark green / red
  - tests: 48 OK
- **Side effect:** the screenshot run added one public check and a 50-message simulation (to the review queue) on
  the local Docker deployment. Clear the queue before the demo.
- **Follow-up:** removed the "Claim desk" label above "No claim on file" in the empty result card (`app/dashboard.py`). Redeployed; a screenshot confirms it is gone; 48 tests OK.
- **Follow-up:** removed the whole empty-state card ("No claim on file" + its help text) from the public page. The right column now stays empty until a message is assessed, then shows the verdict card. Redeployed; screenshot confirms; 48 tests OK.

## 27. Fuller public page (about 12:40)

- **Request:** make the UI fuller and better-looking.
- **`app/styles.css`:**
  - container widened from 1120 to 1280px
  - new components: `.rk-strip` / `.rk-fact` (fact tiles), `.rk-panel` / `.rk-signs` (warning signs and next
    steps), `.rk-steps` / `.rk-step` (how it works) and `.rk-foot` (dark-green footer note)
  - responsive: four columns, then two at ≤860px; steps go to one column at ≤520px
- **`app/dashboard.py`, public page:**
  - **Fact strip** under the title. "5 languages" and "24/7 drift watch" are fixed; "190/190 test scams caught"
    and "98% independent test" are read live from `reports/metrics.json` and labelled synthetic, so the numbers
    stay honest and update after evaluation.
  - **Right column before a check:** "Before you reply / Warning signs" (OTP/PIN/CVV requests, release fees,
    urgency or threats, unknown links or numbers), as useful content rather than an empty placeholder.
  - **After a check:** the verdict card plus a **"What to do now"** list tailored to scam / uncertain / safe.
  - **Bottom:** "How it works", four steps (paste → model scores → unfamiliar wording held → reviewed, then
    safely retrained), and a footer saying it uses synthetic data only, with privacy and emergency advice.
- **Verified:** redeployed Docker. Playwright screenshots of desktop before a check (top and scrolled bottom),
  after a Bank-alert check (SCAM 71%, with next steps) and a 400px phone (facts two per row). Tests: 48 OK.

## 28. Hosting on Streamlit Community Cloud: embedded mode (about 12:50)

- **Problem:** Streamlit Cloud runs a single Streamlit app, but the dashboard calls a separate FastAPI server.
- **Solution: `app/embedded.py`.** It provides the same `get`, `post`, `delete` and `RequestException` interface
  as `requests`, and routes calls to `src/api.py`'s FastAPI app **in-process** with Starlette's `TestClient`, so
  the real routes, admin checks, drift windows, gate and rollback all still apply. Responses are wrapped with `.ok`,
  `.status_code`, `.text`, `.json()` and `raise_for_status()`. On first start, if the model or data files are
  missing, it runs `make_data.py`, `train.py` and `evaluate.py`.
- **`app/dashboard.py`:** copies root-level `st.secrets` into `os.environ` (shell values win). If
  `RAKSHAKA_EMBEDDED=1`, it uses `requests = embedded` and `API = "http://embedded"`. Otherwise nothing changes,
  so `run.sh` and Docker still call the real API.
- **`.gitignore`:** `.streamlit/secrets.toml`.
- **README:** "Hosting on Streamlit Community Cloud" section: main file `app/dashboard.py`, Python 3.12, and
  secrets `RAKSHAKA_EMBEDDED`, `ADMIN_TOKEN`, `PREDICT_PER_MINUTE`. `CLAUDE.md` is updated too.
- **Verified like the Cloud would run it:** a clean repo copy with no models or generated data, a
  `.streamlit/secrets.toml`, and `streamlit run app/dashboard.py` on 8520.
  - first start trained the model (model files created; "loaded model v1")
  - Chrome via Playwright: public Parcel-fee check → SCAM 83%; wrong token refused; right token → admin desk;
    "Send new scam tactics (50)" → "Drift alert (simulation)"
  - tests: 48 OK
- **Limits on the free tier:** the file system is temporary, so review labels, retrained models and logs reset when
  the app restarts. All visitors share one in-process rate-limit bucket. For a persistent pilot, use Docker on a
  server.
- **To go live,** a person must: push to GitHub, then create the app on share.streamlit.io with the secrets
  (this needs the user's Streamlit account).
