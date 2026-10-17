# Rakshaka AI

Rakshaka AI detects scam messages submitted in English, Sinhala, and Singlish
(Sinhala written with Latin characters). The supplied CSVs are English-only,
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
   speakers. Fill the Sinhala and Singlish text independently, and set each
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
   `english`, `sinhala`, `singlish`, or `unknown`. Predictions and drift events
   are written to `artifacts/predictions.jsonl`. Message contents are not
   written to prediction logs.

The classifier combines character and word TF-IDF features with logistic
regression. Character features support Sinhala script and romanized Sinhala,
but performance in those languages is not established until reviewed examples
are supplied and the model is retrained. Language metadata is caller-provided;
the service does not pretend to distinguish English from romanized Sinhala
automatically.

## Monitoring and response

The API records timestamp, model version, declared language, prediction,
probability, latency, and a character-ngram drift signal without storing raw
messages. A drift threshold is a review trigger only; it is not a finding that
a message is malicious. Reviewers should verify new patterns and labels before
adding examples to training data or triggering retraining.

## Validation

Run tests with `python -m unittest discover -s tests`.
