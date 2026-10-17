import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rakshaka.data import (
    ANNOTATION_FIELDS,
    DRIFT_FIELDS,
    MESSAGE_FIELDS,
    create_translation_sample,
    load_and_clean_sources,
    write_csv,
)

DEFAULT_SOURCES = [
    ROOT / "spam.csv",
    ROOT
    / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION"
    / "Dataset_5971.csv",
]


def main():
    parser = argparse.ArgumentParser(
        description="Clean the provided SMS datasets and create a human translation sample."
    )
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=100,
        help="Maximum number of unique English messages per label for translation review.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--sources", nargs="+", type=Path, default=DEFAULT_SOURCES
    )
    args = parser.parse_args()
    if args.sample_per_label < 1:
        parser.error("--sample-per-label must be positive")

    messages, duplicate_count = load_and_clean_sources(args.sources)
    labels = collections.Counter(message["label"] for message in messages)
    output_dir = args.output_dir
    write_csv(output_dir / "processed/messages_clean.csv", MESSAGE_FIELDS, messages)
    write_csv(
        output_dir / "annotation/translation_sample.csv",
        ANNOTATION_FIELDS,
        create_translation_sample(messages, args.sample_per_label, args.seed),
    )
    drift_path = output_dir / "drift/drift_samples.csv"
    if not drift_path.exists():
        write_csv(drift_path, DRIFT_FIELDS, [])

    print("Unique messages: {}".format(len(messages)))
    print("Removed duplicate rows: {}".format(duplicate_count))
    print("Labels: {}".format(dict(sorted(labels.items()))))
    print(
        "Translation review sample: {}".format(
            min(args.sample_per_label, labels.get("legitimate", 0))
            + min(args.sample_per_label, labels.get("scam", 0))
        )
    )
    print("Clean data: {}".format(output_dir / "processed/messages_clean.csv"))
    print("Translations: {}".format(output_dir / "annotation/translation_sample.csv"))
    print("Drift template: {}".format(drift_path))


if __name__ == "__main__":
    main()
