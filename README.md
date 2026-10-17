# Rakshaka AI

**Problem statement:** An ML service that flags scam SMS and WhatsApp messages in Sinhala, Tamil and Singlish, aimed at first-time mobile-banking users and pensioners in Sri Lanka. It detects and responds when new scam tactics (such as fake SL-UDI digital ID verification) make the model drift.

Rakshaka AI classifies messages in English, Sinhala, Singlish, Tamil and Tanglish (Sinhala and Tamil written in
Latin letters). A drift monitor notices when incoming messages stop looking like the training data, switches to a
safe "uncertain" verdict, and routes those messages to human review and one-click retraining.

## Who it protects
First-time mobile-banking users, pensioners and welfare recipients who receive scam SMS and WhatsApp messages in
Sinhala, Tamil or Singlish. These are the groups the Central Bank says scams hit hardest. They include 727,092
government pensioners and 1,251,522 Aswesuma beneficiary families, within a population of 13.9 million internet
users. Reach, harm and the path to a pilot are in [IMPACT.md](IMPACT.md).

## Architecture
![Architecture](download.svg)

```mermaid
flowchart LR
  U[User / Streamlit UI] -->|POST /predict| API[FastAPI service src/api.py]
  API --> P[rakshaka.service.predict: preprocessing + scam probability + char-ngram OOV]
  P --> M[TF-IDF char 2-5 + word 1-2 n-grams + Logistic Regression]
  API --> L[(SQLite prediction log: hash + metadata; raw text only if routed to review)]
  L --> D[Drift monitor: rolling 50 predictions vs validation baseline - confidence + unknown-token rate]
  D -->|DRIFT| A[Alert + safe fallback: UNCERTAIN verdict]
  A --> R[Human review queue]
  R -->|POST /review -> data/reviewed.csv| T[POST /retrain -> new model version, hot reload]
  T --> M
```

### Repository layout
| Path | What it is |
|---|---|
| `src/make_data.py` | Synthetic multilingual data generator: train / val / test / shifted, split by template |
| `src/train.py`, `src/evaluate.py` | Training (+ drift baselines from `val.csv`) and held-out evaluation, writes `reports/metrics.json` |
| `src/api.py`, `src/features.py` | FastAPI service: prediction, logging, drift monitor, review queue, retrain trigger |
| `app/dashboard.py` | Streamlit UI: check a message, monitoring and drift demo, human review + retrain |
| `rakshaka/` | Shared library: text cleaning, label handling, preprocessing, classifier, group splits, per-language metrics, char-ngram OOV signal, `predict()` |
| `scripts/` | Optional pipeline for the public English SMS datasets and human-verified translations (see below) |
| `tests/` | Unit tests (`python -m unittest discover -s tests`) |

## How to run
```bash
docker compose up --build                      # API: http://localhost:8000/docs   UI: http://localhost:8501
API_PORT=8001 docker compose up --build        # if port 8000 is already taken (UI_PORT works the same way)
```
or, without Docker, one command (macOS / Linux):
```bash
./run.sh                               # API: http://127.0.0.1:8000/docs   UI: http://localhost:8501
API_PORT=8001 UI_PORT=8502 ./run.sh    # if 8000 or 8501 is already taken
RETRAIN=1 ./run.sh                     # regenerate data and retrain first
```
`run.sh` creates `.venv` on Python 3.12 or newer if it is missing (a plain `python` may be older and fail), installs
`requirements.txt` when it changes, and checks that both ports are free. It generates the data and trains the model
only if they are missing, so an existing model and its review history are kept. Then it starts the API, waits for
`/health`, and starts the dashboard. Ctrl+C stops both.

By hand, the same steps are:
```bash
python3.12 -m venv .venv && source .venv/bin/activate     # any Python 3.12+
pip install -r requirements.txt
python src/make_data.py && (cd src && python train.py && python evaluate.py)   # first run: data + model
cd src && uvicorn api:app --port 8000                     # terminal 1
streamlit run app/dashboard.py                            # terminal 2: activate .venv, from the repo root
```
On Windows the dashboard calls the API on `127.0.0.1`, not `localhost`. Using `localhost` adds about 2 s per
request because of the IPv6 fallback.

API endpoints: `GET /health`, `POST /predict` (`{"text": "...", "language": "sinhala"}`; `language` is one of
`english`, `sinhala`, `singlish`, `tamil`, `tanglish`, `unknown`), `GET /monitor`, `GET /alerts`,
`GET /predictions` (recent prediction metadata for the charts, never message text), `GET /review_queue`,
`POST /review`, `POST /retrain`.

## Streamlit dashboard

There is **one UI**: `app/dashboard.py`. It reads all its data from the FastAPI service (`src/api.py`). `./run.sh`
starts both (see [How to run](#how-to-run)). To start them by hand, inside the Python 3.12+ `.venv`:

```bash
cd src && uvicorn api:app --port 8000          # terminal 1: API (needs a trained model in models/)
streamlit run app/dashboard.py                 # terminal 2, from the repo root: UI on http://localhost:8501
```

If the API runs on another port, set `API_URL`, for example
`API_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py`.

Three tabs:

- **Check a message.** Paste a message and choose its language. The result shows the verdict (SCAM, LIKELY SAFE, or
  UNCERTAIN - sent for review), the scam probability, the share of unseen words, and the share of unfamiliar
  character patterns.
- **Monitoring & drift.** Shows the drift status and two bar charts that compare the live window with the validation
  baseline and the alert line: one for the unknown-token rate and one for mean confidence. Below them are a line
  chart of every check over time, and bar charts by language and by result. "Simulate traffic" sends test or shifted
  messages so you can watch the drift alert fire. The charts use `GET /predictions`, which never returns message text.
- **Human review.** Label the queued messages, then **Save labels + retrain**. That writes the labels to
  `data/reviewed.csv`, retrains, and hot-reloads the new model version. Check each label carefully: every row is
  used for training.

## Data sources
| Source | Licence / method | Used for |
|---|---|---|
| `src/make_data.py` templates | Synthetic, hand-written by the team with AI assistance (Claude). All names, numbers and links are fake placeholders (`DemoBank`, `07X-XXXXXXX`, `*.example`) | train / val / test / shifted |
| `data/extra_train.csv` | Synthetic, 188 hand-curated messages written with Claude Code (Claude Opus 5.5), not yet checked by native speakers: Sinhala, Tamil, Singlish and Tanglish, about 23-24 scam and 23-24 legitimate per language. Fake placeholders only (`DemoBank`, `TestPay`, `07X-XXXXXXX`, `*.example`). The wording avoids the held-out test/val templates and their translations, and the shifted-set tactics | train |
| `data/extra_test.csv` (optional) | Not used in this submission. `make_data.py` adds it to the test set if present (prompt template in `data/PROMPTS.md`) | test |
| `data/reviewed.csv` (runtime) | Labels entered by human reviewers in the dashboard | retraining |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Almeida & Hidalgo, 2011, DOI 10.24432/C5CC84) and [SMS Phishing Dataset for Machine Learning and Pattern Recognition](https://data.mendeley.com/datasets/f45bkkt8pr/1) (Mishra & Soni, 2022, DOI 10.17632/f45bkkt8pr.1) (optional, not included in the repo) | Both CC BY 4.0 | `scripts/` pipeline only |

No real personal data, phone numbers, NIC numbers or screenshots are used, in line with the PDPA No. 9 of 2022.
Demo messages that mention real programmes or services (Aswesuma, SL-UDI, eZ Cash) are clearly labelled synthetic examples.

The data is split **by template**: every message generated from one template lands in exactly one split,
so test scores measure generalisation to unseen wording instead of memorised templates. The split is
stratified by (label, language) with seed 42.
- **Train** (`data/train.csv`): 1,365 rows: 1,177 from 46 templates plus the 188 messages in `data/extra_train.csv`.
- **Validation** (`data/val.csv`): 276 rows from 10 unseen templates. Used only to measure the drift-monitor baseline.
- **Held-out test set** (`data/test.csv`): 264 rows from 12 unseen templates, covering the same tactics as training.
- **Shifted set** (`data/shifted.csv`): 152 rows from 15 templates with **new tactics**: SL-UDI digital ID / QR
  scams, biometric "re-verification" fees, USDT/AI crypto groups, "new number" impersonation, voice-note lures and
  heavier code-mixing.

## Evaluation (model v1, from `reports/metrics.json`)
| Set | Rows | Balanced acc. | Macro F1 | F1 (scam) | Mean confidence | Unknown-token rate |
|---|---|---|---|---|---|---|
| validation baseline | 276 | | | | 0.828 | 18.7% |
| **test** (unseen templates) | 264 | 0.926 | 0.946 | 0.972 | 0.823 | 12.3% |
| shifted (new tactics) | 152 | 1.000 | 1.000 | 1.000 | 0.799 | **38.1%** |

Test confusion matrix (rows = actual legitimate/scam, columns = predicted): `[[63, 11], [0, 190]]`.
No scam was missed. 11 legitimate messages were flagged as scam.

Per language on test (balanced accuracy): English 1.00, Singlish 1.00, Tamil 1.00, **Sinhala 0.63**.
Adding `data/extra_train.csv` raised Sinhala from 0.50 and test macro F1 from 0.925. Some held-out Sinhala
legitimate messages are still flagged as scam, so Sinhala legitimate data remains the first target for more
data (`data/PROMPTS.md`). These scores come from a small synthetic template set and are not a claim about
real-world traffic.

## Monitoring and drift response
- **Prediction logging:** every prediction goes to `logs/predictions.db` with a timestamp, text hash, language,
  label, confidence, unknown-token rate, char-ngram OOV ratio, model version and routing. Operational logs go to
  `logs/service.log`. **Privacy:** raw message text is stored **only** for messages routed to human review,
  because a reviewer has to read them. Auto-handled messages keep only a hash.
- **Drift signal:** a rolling window of the last 50 predictions (judged after 20) is compared with baselines
  measured on the validation templates:
  - mean confidence drops by more than 0.10, or
  - the unknown-token rate (share of words never seen in training) rises more than 12 points. In 1,000 random
    50-message batches, normal test traffic never rose more than -1 point, and shifted traffic always rose at
    least +13.5 points.
- **Response:** a `DRIFT` alert is logged once per episode (`/alerts`, dashboard banner). Verdicts switch to
  "UNCERTAIN - be careful, sent for review", a safe fallback that never says "safe". Messages go to the
  **human review queue**, where a reviewer labels them (`POST /review` → `data/reviewed.csv`) and triggers
  **`/retrain`**. That creates a new model version and hot-reloads it. Individual messages also go to review when
  confidence is below 0.60 (about 3% of normal traffic) or when 50% or more of their words are unseen.

**Measured drift demo** (`uvicorn` + 50 messages per step, as in the dashboard's "Simulate traffic"):

| Step | Status | Mean conf. | Unknown-token rate | Sent to review |
|---|---|---|---|---|
| 50 test messages | OK | 0.823 | 15.8% | 2 |
| 50 shifted messages | **DRIFT** ("unknown-token rate 19% -> 39%") | 0.798 | 38.9% | 27 |
| review 29 queued messages → `/retrain` → v2 | | | | |
| 50 shifted messages | OK | 0.812 | 1.5% | 1 |
| 50 test messages | OK | 0.835 | 10.6% | 3 |

The drift signal fires on input shift **while accuracy is still high** (shifted F1 1.00). That is the point:
the product notices new scam tactics before it quietly starts failing. The post-retrain shifted batch shares
templates with the reviewed messages, so it shows that the loop works, not unseen-tactic accuracy.

`python scripts/demo_drift.py --model models/model.joblib --baseline data/test.csv --shifted data/shifted.csv`
gives an offline char-ngram OOV comparison on the same model: mean unseen char-ngram ratio 0.20 on test vs
0.33 on shifted. Its fixed per-message threshold of 0.25 also flags 39 of 200 normal test messages, so the
API's main drift signal is the calibrated window comparison above. The char-ngram ratio is logged as a
secondary signal.

## Optional: public English datasets and human-verified translations (`scripts/`)
The `scripts/` pipeline trains the same classifier from the two public English SMS datasets plus
**human-verified** Sinhala, Singlish, Tamil and Tanglish translations. It writes `artifacts/` and is
separate from the main demo.

1. Place the datasets at `spam.csv` and
   `SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION/Dataset_5971.csv`. They are not
   included in this repository. Both are CC BY 4.0 (see Data sources), so credit the authors if you share them.
2. `python scripts/prepare_data.py` maps `ham` to `legitimate` and `spam`/`smishing` to `scam`, removes exact
   duplicates, keeps the original labels and source row references, and fails on conflicting labels. It also
   writes `data/annotation/translation_sample.csv`, with up to 100 unique messages per label (`--sample-per-label`).
   Existing reviewed annotations are preserved when it runs again.
3. Native speakers fill `data/annotation/translation_sample.csv` (Sinhala, Singlish, Tamil, Tanglish columns)
   independently, and set each language's status to `verified` with a reviewer identifier only after confirming
   that the translation keeps the message's meaning and label. **The code never translates automatically.** Only
   verified rows are used.
4. `python scripts/train_model.py` splits by original message group, so all language variants of one message
   share a split. It balances training groups by downsampling the majority class, keeps the natural class mix in
   validation and test, and writes overall and per-language metrics to `artifacts/metrics.json`. A language with
   only one label in a split is marked not evaluable rather than given a misleading score.
5. Add newly observed, reviewed messages to `data/drift/drift_samples.csv` (keep its header), then run
   `python scripts/monitor_drift.py`; `python scripts/demo_drift.py` compares a baseline file with a shifted one.
   Both report unseen char-ngram ratios. That is a text-shift signal, not a labelled accuracy measure and not a
   finding that a message is malicious. They recommend human review and retraining but never retrain automatically.
6. `python scripts/serve.py` is the original standard-library HTTP server (legacy). It takes
   `{"message": "...", "language": "sinhala"}` and logs predictions and drift events, without message text, to
   `artifacts/predictions.jsonl`. The FastAPI service in `src/api.py` uses the same `rakshaka.service.predict()`
   and is the supported entry point.

Language is declared by the caller; the service does not try to tell English from romanized Sinhala on its own.
Character features can represent any script, but accuracy in a language is not established until reviewed
examples in it are added and the model is retrained.

## Validation
`python -m unittest discover -s tests` runs 13 unit tests. CI (`.github/workflows/tests.yml`) runs the tests,
the full data → train → evaluate pipeline and a container build.

## Impact statement
See [IMPACT.md](IMPACT.md).

## AI tools used
- Claude (Anthropic) generated the initial project scaffold (`src/`, `app/`, Docker files), which the team declares as a reused template.
- Claude Code (Claude Opus 5.5) helped merge the scaffold with the team's `rakshaka/` pipeline, fix bugs,
  write extra synthetic templates and calibrate the drift thresholds. Every change was reviewed by the team.
- Synthetic data: the `src/make_data.py` templates and the 188 messages in `data/extra_train.csv` were written
  with Claude Code (Claude Opus 5.5). The `extra_train.csv` messages have not yet been checked by native speakers.
- ChatGPT (OpenAI) was used by a team member to generate Singlish versions of messages from an English dataset.

## Licence
MIT (see [LICENSE](LICENSE)).


## Admin access for review and monitoring
The public message checker stays unauthenticated for local demos. Review queue,
label submission, retraining, rollback, deletion of queued review text,
prediction history and alert history require `X-Admin-Token` sent to the API.
With `ADMIN_TOKEN` unset, administrative routes return 503 (disabled).

For local Docker Compose, set a long random `ADMIN_TOKEN` in your shell or in
an ignored `.env` file before running `docker compose up --build`.
Never commit that secret. For example, generate a token using
`python -c "import secrets; print(secrets.token_urlsafe(32))"`.
The Streamlit admin tabs ask the reviewer to enter the token as a password field.
It is forwarded only when that reviewer requests sensitive information or actions.
Do not configure Streamlit to automatically use an administrator secret for all
visitors. Access should be restricted to trusted reviewers and protected by
HTTPS outside localhost.

Note: token authentication is a basic demonstration control, not a complete
production identity solution. Production deployment additionally requires TLS,
rate limiting, identity/role management, audit trails, and cross-user access
isolation. The automated route-guard tests are static regression checks; run
end-to-end authenticated API tests before public deployment.
