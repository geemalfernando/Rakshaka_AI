"""Generate reproducible drift comparison from real local clean messages and safe synthetic shifts.

No user data or scam content is sent to anyone. Drift alarms depend on actual trained model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rakshaka.modeling import char_ngram_oov_ratio

SHIFTED = [
    "உங்கள் கணக்கு பாதுகாப்பிற்காக சரிபார்க்கவும்",
    "Ungal kanakkai udane sariparkkavum",
    "Oyage ginuma ikmanata pariksha karanna",
    "ඔබගේ ගිණුම පරීක්ෂා කරන්න",
    "Please check your delivery verification notice",
]


def summarise(model, messages, threshold):
    values = [char_ngram_oov_ratio(model, message) for message in messages]
    return {
        "count": len(values),
        "mean_unseen_char_ngram_ratio": sum(values) / len(values),
        "messages_over_threshold": sum(value >= threshold for value in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/processed/messages_clean.csv")
    parser.add_argument("--shifted", type=Path, help="Optional reviewed shifted CSV with a text column")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    artifact = joblib.load(args.model)
    with args.baseline.open(encoding="utf-8-sig", newline="") as stream:
        baseline = [row["text"] for row in csv.DictReader(stream) if row.get("text")]
    if not baseline:
        parser.error("No baseline messages found")
    if args.shifted:
        with args.shifted.open(encoding="utf-8-sig", newline="") as stream:
            shifted = [row["text"] for row in csv.DictReader(stream) if row.get("text")]
    else:
        shifted = SHIFTED
    if not shifted:
        parser.error("No shifted messages found")
    model = artifact["classifier"]
    before = summarise(model, baseline[:min(len(baseline), 200)], args.threshold)
    after = summarise(model, shifted, args.threshold)
    result = {
        "model_version": artifact.get("model_version"),
        "baseline": before,
        "shifted": after,
        "threshold": args.threshold,
        "drift_alert": after["mean_unseen_char_ngram_ratio"] >= args.threshold,
        "response": "human_review_and_retraining_recommended"
        if after["mean_unseen_char_ngram_ratio"] >= args.threshold else "continue_monitoring",
        "note": "Synthetic shifted examples are illustrative; actual alert is measured, not guaranteed.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
