# Rakshaka AI

Rakshaka AI detects scam messages submitted in English, Sinhala, Singlish, Tamil, and Tanglish
(Sinhala/Tamil written with Latin characters). The supplied CSVs are English-only,
so this workflow preserves them and creates a balanced, human-review translation
sample. It does not generate translations automatically.

## Data workflow

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Place the two source datasets at `spam.csv` and
   `SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION/Dataset_5971.csv`.
   They are not included in this repository because redistribution rights were
   not established.
3. Clean and combine both source datasets:
   `python scripts/prepare_data.py`.
   This standardizes `ham` as `legitimate` and `spam`/`smishing` as `scam`,
   removes exact text duplicates, retains original labels and source row
   references, and fails if the same message has conflicting labels.
4. Review `data/annotation/translation_sample.csv` with native Sinhala
   speakers. Fill the Sinhala, Singlish, Tamil and Tanglish text independently, and set each
   language's status to `verified` only after a human reviewer confirms that
   the translation preserves the message's meaning and scam/legitimate label.
   Enter the reviewer's name or identifier. The initial sample contains up to
   100 unique messages per label; change `--sample-per-label` to adjust it.
5. Train and evaluate:
   `python scripts/train_model.py`.
   Only verified translations are used. All language variants sharing an
   original message are assigned to one train/validation/test split. The
   training groups are balanced by downsampling the majority class; validation
   and test sets retain their natural class distributions. Overall and
   per-language metrics are written to `artifacts/metrics.json`; a language
   with only one label in a split is marked not evaluable rather than assigned
   a misleading score.
6. Add newly observed and reviewed messages to
   `data/drift/drift_samples.csv` using its existing header, then run
   `python scripts/monitor_drift.py`. Its unseen character-ngram ratio is a
   text-shift signal, not a substitute for labeled accuracy evaluation. Alerts
   recommend human review and retraining; they never retrain automatically.
7. Run the local prediction API:
   `python scripts/serve.py`.
   `GET /health` reports service/model status. `POST /predict` accepts JSON
   such as `{"message":"...","language":"sinhala"}`; `language` can be
   `english`, `sinhala`, `singlish`, `tamil`, `tanglish`, or `unknown`. Predictions and drift events
   are written to `artifacts/predictions.jsonl`. Message contents are not
   written to prediction logs.

The classifier combines character and word TF-IDF features with logistic
regression. Character features can represent Sinhala/Tamil scripts and romanized text,
but performance in those languages is not established until reviewed examples
are supplied and the model is retrained. Language metadata is caller-provided;
the service does not pretend to distinguish English from romanized Sinhala
automatically.

## Streamlit app

The browser app is the place a person brings a message they already received,
from a phone text, an email, or anywhere else. Scoring starts when they press
**Check this message**. The app does not read a phone or an inbox.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Three screens:

- **Check a message.** The form asks where the message came from (phone text,
  email, or other), the language, and the message text. The result appears
  under the form after submit.
- **Drift watch.** Admins see a before/after comparison of cleaned messages
  against a shifted set, a chart of submitted checks, and the review inbox.
  The watch line is an unseen character-ngram ratio of 0.25. A shift recommends
  human review and retraining. The app never retrains by itself.
- **Who this protects.** A one-page impact note: who is protected, team
  estimates of reach and harm, and a pilot path. Edit `IMPACT_ESTIMATES` in
  `app.py` to replace the placeholder figures. They are labelled as team
  estimates on the page.

`artifacts/predictions.jsonl` records the time, model version, language,
source, prediction, scam probability, drift ratio, and latency. It does not
store the message text. `artifacts/review_queue.jsonl` stores the message text
only when the person presses **Send this to a reviewer**.

If `artifacts/model.joblib` is missing, the app still opens and says to train
first. Drift watch uses `data/processed/messages_clean.csv` and, when present,
`data/drift/drift_samples.csv`. Otherwise it uses the same built-in sample
shift as `scripts/demo_drift.py`. The prediction API is unchanged:
`python scripts/serve.py`.

## Monitoring and response

The API records timestamp, model version, declared language, prediction,
probability, latency, and a character-ngram drift signal without storing raw
messages. A drift threshold is a review trigger only; it is not a finding that
a message is malicious. Reviewers should verify new patterns and labels before
adding examples to training data or triggering retraining.

## Validation

Run tests with `python -m unittest discover -s tests`.

## Extended multilingual workflow
The two original English CSVs are not redistributed. The annotation CSV generated
by `prepare_data.py` now has Tamil and Tanglish columns as well as Sinhala
and Singlish. Existing reviewed annotations are preserved when preparation runs
again for the same source message groups. Older annotation CSVs without the new
columns remain readable; only rows marked `verified` with reviewer identifiers
are included in training. **The code does not automatically translate text.**
Recruit fluent reviewers and document translation/generation methods, licences,
and review decisions. Never claim accuracy in a language without sufficient
independent, labeled held-out examples.

## Drift demonstration
After training, run:
```bash
python scripts/demo_drift.py
python scripts/demo_drift.py --shifted data/drift/drift_samples.csv
```
The first command compares a sample of cleaned source messages with *illustrative*
synthetic language shifts; the second uses a reviewed CSV with a `text` column.
It reports observed OOV values and whether the chosen review threshold actually
fired. A shift is **not** proof of a scam, nor proof of reduced accuracy.
Review shifted records with human evaluators before using them for training.
For the judges, show both the original prediction and drift signal / review response.

## Container deployment
After generating a local model with `scripts/train_model.py`:
```bash
docker build -t rakshaka-ai .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" rakshaka-ai
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"message":"Sample notification","language":"english"}'
```
The model and private datasets are not included in the image. Do not expose this
unauthenticated demonstration service directly to the public internet. Add network
controls, rate limits and authentication before any public deployment.

## Competition submission evidence (must complete with real results)
- **Training data**: record dataset names, links, licence terms, row counts,
  classes, translation provenance and review process. Never upload private data.
- **Evaluation**: paste actual values from `artifacts/metrics.json`: overall
  and per-language test counts, balanced accuracy, macro F1, precision, recall
  and confusion matrix. Describe split by source message group.
- **Shift monitoring**: paste actual baseline/shifted rates, chosen threshold,
  whether alert fired, and the human review action.
- **Impact**: attach a one-page beneficiary/reach/adoption statement.
- **Architecture**: English CSVs -> normalization & group split -> verified
  translated samples -> TF-IDF/logistic regression -> HTTP API -> prediction
  logs -> OOV drift monitoring -> human review/retraining recommendation.
- **AI coding tools**: record assistant name/tool usage and human validation.
- **Pitch**: provide the required 8-minute presentation as a PDF.

The scripts and tests establish an implementation workflow, **not** verified
multilingual model performance or an observed drift alert. Those require a
successful run with local datasets and annotations.
