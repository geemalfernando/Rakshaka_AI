"""Data helpers for the Streamlit app.

Scoring runs only when the caller asks, after a person submits the form.
Prediction logs store the result and where the message came from, never the text.
The review queue stores the text only when the person sends it to a reviewer.
"""
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .modeling import char_ngram_oov_ratio
from .service import DRIFT_OOV_THRESHOLD, load_artifact, predict


ROOT = Path(__file__).resolve().parents[1]
SOURCE_VALUES = ("phone", "email", "other")
LOG_FIELDS = (
    "timestamp",
    "model_version",
    "language",
    "source",
    "prediction",
    "scam_probability",
    "drift_oov_ratio",
    "latency_ms",
)
# Same illustrative shift used by scripts/demo_drift.py. Samples, not real notices.
SHIFTED_TEXTS = (
    "உங்கள் கணக்கு பாதுகாப்பிற்காக சரிபார்க்கவும்",
    "Ungal kanakkai udane sariparkkavum",
    "Oyage ginuma ikmanata pariksha karanna",
    "ඔබගේ ගිණුම පරීක්ෂා කරන්න",
    "Please check your delivery verification notice",
)
SAMPLE_MESSAGES = (
    {
        "label": "Sinhala sample",
        "language": "sinhala",
        "text": "ඔබගේ ගිණුම පරීක්ෂා කරන්න",
    },
    {
        "label": "Tamil sample",
        "language": "tamil",
        "text": "உங்கள் கணக்கு பாதுகாப்பிற்காக சரிபார்க்கவும்",
    },
    {
        "label": "English sample",
        "language": "english",
        "text": "Please check your delivery verification notice",
    },
)


def app_paths(root=None):
    root = Path(root) if root else ROOT
    return {
        "model": root / "artifacts" / "model.joblib",
        "log": root / "artifacts" / "predictions.jsonl",
        "review": root / "artifacts" / "review_queue.jsonl",
        "baseline": root / "data" / "processed" / "messages_clean.csv",
        "shifted": root / "data" / "drift" / "drift_samples.csv",
    }


def load_model(path):
    path = Path(path)
    if not path.exists():
        return None
    return load_artifact(path)


def submit_check(artifact, message, language, source, log_path):
    """Score one submitted message and append a log line without the text."""
    if source not in SOURCE_VALUES:
        raise ValueError("source must be one of: {}".format(", ".join(SOURCE_VALUES)))
    started = time.monotonic()
    result = predict(artifact, message, language)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": artifact.get("model_version", "unknown"),
        "language": result["language"],
        "source": source,
        "prediction": result["prediction"],
        "scam_probability": result["scam_probability"],
        "drift_oov_ratio": result["drift_oov_ratio"],
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
    }
    if set(event) != set(LOG_FIELDS):
        raise RuntimeError("Prediction log fields drifted from the allowed set")
    append_jsonl(log_path, event)
    scored = dict(result)
    scored["source"] = source
    scored["model_version"] = event["model_version"]
    scored["timestamp"] = event["timestamp"]
    scored["latency_ms"] = event["latency_ms"]
    return scored


def submit_review(message, result, note, review_path):
    """Store the message text because the person asked a reviewer to see it."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    item = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "source": result["source"],
        "language": result["language"],
        "prediction": result["prediction"],
        "scam_probability": result["scam_probability"],
        "drift_oov_ratio": result["drift_oov_ratio"],
        "drift_alert": bool(result["drift_alert"]),
        "note": (note or "").strip(),
    }
    append_jsonl(review_path, item)
    return item


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def read_reviews(path):
    return list(reversed(read_jsonl(path)))


def summarise_predictions(events, threshold=DRIFT_OOV_THRESHOLD):
    checks = []
    for event in events:
        if "prediction" not in event or "drift_oov_ratio" not in event:
            continue
        try:
            ratio = float(event["drift_oov_ratio"])
        except (TypeError, ValueError):
            continue
        checks.append(
            {
                "timestamp": event.get("timestamp", ""),
                "language": event.get("language") or "unknown",
                "source": event.get("source") or "unrecorded",
                "prediction": event["prediction"],
                "drift_oov_ratio": ratio,
            }
        )
    by_prediction = {"scam": 0, "legitimate": 0}
    by_language = {}
    by_source = {}
    for check in checks:
        by_prediction[check["prediction"]] = by_prediction.get(check["prediction"], 0) + 1
        by_language[check["language"]] = by_language.get(check["language"], 0) + 1
        by_source[check["source"]] = by_source.get(check["source"], 0) + 1
    mean_ratio = None
    if checks:
        mean_ratio = sum(check["drift_oov_ratio"] for check in checks) / len(checks)
    over = sum(check["drift_oov_ratio"] >= threshold for check in checks)
    return {
        "count": len(checks),
        "by_prediction": by_prediction,
        "by_language": by_language,
        "by_source": by_source,
        "mean_drift_oov_ratio": mean_ratio,
        "messages_over_threshold": over,
        "drift_alert": mean_ratio is not None and mean_ratio >= threshold,
        "series": checks,
        "threshold": threshold,
    }


def compare_drift(artifact, baseline_path, shifted_path, threshold=DRIFT_OOV_THRESHOLD):
    """Compare cleaned messages with a shifted set. Does not score a user submission."""
    baseline = _texts_from_csv(baseline_path, limit=200)
    if not baseline:
        return {
            "available": False,
            "reason": (
                "Cleaned messages are not in data/processed/messages_clean.csv yet. "
                "Run python scripts/prepare_data.py and this comparison can show "
                "how new wording differs."
            ),
        }
    shifted = _texts_from_csv(shifted_path)
    if shifted:
        shifted_source = "reviewed_file"
    else:
        shifted = list(SHIFTED_TEXTS)
        shifted_source = "built_in_samples"
    classifier = artifact["classifier"]
    before = _summarise_ratios(classifier, baseline, threshold)
    after = _summarise_ratios(classifier, shifted, threshold)
    alert = after["mean_unseen_char_ngram_ratio"] >= threshold
    return {
        "available": True,
        "threshold": threshold,
        "model_version": artifact.get("model_version", "unknown"),
        "baseline": before,
        "shifted": after,
        "shifted_source": shifted_source,
        "drift_alert": alert,
        "response": (
            "human_review_and_retraining_recommended"
            if alert
            else "continue_monitoring"
        ),
    }


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _texts_from_csv(path, limit=None):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "text" not in (reader.fieldnames or ()):
            return []
        texts = [(row.get("text") or "").strip() for row in reader]
    texts = [text for text in texts if text]
    if limit is not None:
        return texts[:limit]
    return texts


def _summarise_ratios(classifier, messages, threshold):
    ratios = [char_ngram_oov_ratio(classifier, message) for message in messages]
    return {
        "count": len(ratios),
        "mean_unseen_char_ngram_ratio": sum(ratios) / len(ratios),
        "messages_over_threshold": sum(ratio >= threshold for ratio in ratios),
    }
