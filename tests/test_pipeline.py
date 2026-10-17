import tempfile
import unittest
from pathlib import Path

from rakshaka.data import (
    ANNOTATION_FIELDS,
    build_training_rows,
    load_and_clean_sources,
    load_verified_translations,
    write_csv,
)
from rakshaka.modeling import (
    balance_training_groups,
    evaluate_by_language,
    split_by_group,
)
from rakshaka.service import predict


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cleans_schemas_combines_duplicates_and_standardizes_labels(self):
        first = self.root / "spam.csv"
        second = self.root / "dataset.csv"
        first.write_text(
            "v1,v2,,,\nham,Hello there,,,\nspam,Visit https://bad.example,,,\n",
            encoding="utf-8",
        )
        second.write_text(
            "LABEL,TEXT,URL,EMAIL,PHONE\n"
            "ham, hello   there ,No,No,No\n"
            "Smishing,Call 0771234567,No,No,yes\n",
            encoding="utf-8",
        )
        messages, duplicates = load_and_clean_sources([first, second])
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(messages), 3)
        self.assertEqual({row["label"] for row in messages}, {"legitimate", "scam"})
        repeated = next(row for row in messages if row["text"].casefold() == "hello there")
        self.assertIn("spam.csv", repeated["source_refs"])
        self.assertIn("dataset.csv", repeated["source_refs"])

    def test_rejects_conflicting_labels_for_duplicate_messages(self):
        first = self.root / "spam.csv"
        second = self.root / "dataset.csv"
        first.write_text("v1,v2\nham,Same text\n", encoding="utf-8")
        second.write_text("LABEL,TEXT\nspam,Same text\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            load_and_clean_sources([first, second])

    def test_recovers_windows_1252_bytes_in_mixed_encoding_source(self):
        source = self.root / "spam.csv"
        source.write_bytes(b"v1,v2\nspam,Pay \xa3 now\n")
        messages, _ = load_and_clean_sources([source])
        self.assertEqual(messages[0]["text"], "Pay \u00a3 now")

    def test_only_verified_reviewed_translations_join_training(self):
        messages = [
            {
                "group_id": "group-one",
                "label": "scam",
                "text": "Please send money",
            }
        ]
        annotations = self.root / "translations.csv"
        write_csv(
            annotations,
            ANNOTATION_FIELDS,
            [
                {
                    "group_id": "group-one",
                    "label": "scam",
                    "english_text": "Please send money",
                    "sinhala_text": "reviewed Sinhala",
                    "sinhala_status": "verified",
                    "sinhala_reviewer": "reviewer",
                    "singlish_text": "unreviewed",
                    "singlish_status": "pending",
                }
            ],
        )
        translations = load_verified_translations(annotations, messages)
        training_rows = build_training_rows(messages, translations)
        self.assertEqual([row["language"] for row in training_rows], ["english", "sinhala"])

    def test_split_keeps_every_group_in_one_partition_and_balances_training(self):
        rows = []
        for label in ("legitimate", "scam"):
            for index in range(20):
                group = "{}-{}".format(label, index)
                rows.append(
                    {
                        "group_id": group,
                        "label": label,
                        "language": "english",
                        "text": "message {} {}".format(label, index),
                    }
                )
                rows.append(
                    {
                        "group_id": group,
                        "label": label,
                        "language": "singlish",
                        "text": "translated {}".format(index),
                    }
                )
        splits = split_by_group(rows)
        self.assertFalse(splits["train"] & splits["validation"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["validation"] & splits["test"])
        train_rows = [row for row in rows if row["group_id"] in splits["train"]]
        balanced = balance_training_groups(train_rows)
        labels = {row["label"] for row in balanced}
        self.assertEqual(labels, {"legitimate", "scam"})
        self.assertEqual(
            len(
                {
                    row["group_id"]
                    for row in balanced
                    if row["label"] == "legitimate"
                }
            ),
            len(
                {
                    row["group_id"]
                    for row in balanced
                    if row["label"] == "scam"
                }
            ),
        )


class PredictionTests(unittest.TestCase):
    def test_rejects_empty_messages(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            predict({}, " ")

    def test_reports_metrics_separately_for_each_language(self):
        class Predictor:
            def predict(self, texts):
                return [
                    "scam" if "scam" in text else "legitimate" for text in texts
                ]

        rows = [
            {
                "group_id": language + "-" + label,
                "language": language,
                "label": label,
                "text": label,
            }
            for language in ("english", "sinhala")
            for label in ("legitimate", "scam")
        ]
        results = evaluate_by_language(Predictor(), rows)
        self.assertEqual(set(results), {"english", "sinhala"})
        self.assertTrue(results["english"]["evaluable"])
        self.assertEqual(results["sinhala"]["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
