# Workflow Log

This log records what was done to set up, run and test Rakshaka AI on 17 October 2026, during the TensorForge 2.0 build window. It also notes what was learned along the way. Times are local (UTC+5:30).

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

See the updated list at the end of section 8.

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
