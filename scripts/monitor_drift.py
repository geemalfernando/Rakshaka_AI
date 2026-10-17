import argparse
import csv
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rakshaka.modeling import char_ngram_oov_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Compare curated/new messages with the trained character-ngram baseline."
    )
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/drift/drift_samples.csv",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not args.model.exists():
        parser.error("Model artifact not found; run scripts/train_model.py first")

    artifact = joblib.load(args.model)
    if not isinstance(artifact, dict) or "classifier" not in artifact:
        parser.error("Model artifact has an unsupported format")
    if not args.input.exists():
        parser.error("Drift dataset not found: {}".format(args.input))

    with args.input.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "text" not in (reader.fieldnames or ()):
            parser.error("Drift dataset must contain a 'text' column")
        messages = [row for row in reader if (row.get("text") or "").strip()]
    if not messages:
        print("No reviewed drift messages to compare.")
        return

    ratios = [char_ngram_oov_ratio(artifact["classifier"], row["text"]) for row in messages]
    mean_ratio = sum(ratios) / len(ratios)
    alert_count = sum(ratio >= args.threshold for ratio in ratios)
    report = {
        "model_version": artifact.get("model_version", "unknown"),
        "messages_compared": len(ratios),
        "mean_unseen_char_ngram_ratio": mean_ratio,
        "threshold": args.threshold,
        "messages_over_threshold": alert_count,
        "drift_alert": mean_ratio >= args.threshold,
        "response": (
            "human_review_and_retraining_recommended"
            if mean_ratio >= args.threshold
            else "continue_monitoring"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["drift_alert"]:
        print("DRIFT ALERT: review new wording and verify labels before retraining.")


if __name__ == "__main__":
    main()
