import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rakshaka.data import build_training_rows, load_and_clean_sources, load_verified_translations
from rakshaka.modeling import (
    balance_training_groups,
    evaluate_by_language,
    evaluate_classifier,
    make_classifier,
    preprocess_text,
    split_by_group,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Rakshaka's group-split message classifier."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/messages_clean.csv")
    parser.add_argument(
        "--translations",
        type=Path,
        default=ROOT / "data/annotation/translation_sample.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        parser.error("Clean dataset not found; run scripts/prepare_data.py first")

    import csv

    with args.data.open("r", encoding="utf-8-sig", newline="") as csv_file:
        messages = list(csv.DictReader(csv_file))
    translations = load_verified_translations(args.translations, messages)
    rows = build_training_rows(messages, translations)
    splits = split_by_group(rows, seed=args.seed)
    training_rows = [row for row in rows if row["group_id"] in splits["train"]]
    training_rows = balance_training_groups(training_rows, seed=args.seed)
    validation_rows = [row for row in rows if row["group_id"] in splits["validation"]]
    test_rows = [row for row in rows if row["group_id"] in splits["test"]]

    classifier = make_classifier()
    classifier.fit(
        [preprocess_text(row["text"]) for row in training_rows],
        [row["label"] for row in training_rows],
    )
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "message_groups": len({row["group_id"] for row in rows}),
        "verified_translation_rows": len(translations),
        "training_rows_after_balancing": len(training_rows),
        "training_groups_after_balancing": len(
            {row["group_id"] for row in training_rows}
        ),
        "training_label_groups": dict(
            sorted(
                Counter(
                    message["label"]
                    for message in messages
                    if message["group_id"] in {row["group_id"] for row in training_rows}
                ).items()
            )
        ),
        "languages_in_training": sorted({row["language"] for row in training_rows}),
        "validation": evaluate_classifier(classifier, validation_rows),
        "test": evaluate_classifier(classifier, test_rows),
        "validation_by_language": evaluate_by_language(classifier, validation_rows),
        "test_by_language": evaluate_by_language(classifier, test_rows),
        "split_group_counts": {
            split: len(groups) for split, groups in splits.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    joblib.dump(
        {
            "classifier": classifier,
            "model_version": model_version,
            "languages": metrics["languages_in_training"],
        },
        args.output_dir / "model.joblib",
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
        metrics_file.write("\n")

    print("Model saved: {}".format(args.output_dir / "model.joblib"))
    print("Metrics saved: {}".format(args.output_dir / "metrics.json"))
    print("Languages in training: {}".format(", ".join(metrics["languages_in_training"])))
    print("Test balanced accuracy: {:.4f}".format(metrics["test"]["balanced_accuracy"]))
    print("Test macro F1: {:.4f}".format(metrics["test"]["macro_f1"]))
    if not translations:
        print(
            "WARNING: no verified Sinhala/Singlish translations were found; "
            "this is an English-only baseline."
        )


if __name__ == "__main__":
    main()
