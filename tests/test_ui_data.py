import json
import tempfile
import unittest
from pathlib import Path

from rakshaka.modeling import make_classifier, preprocess_text
from rakshaka.ui_data import (
    LOG_FIELDS,
    compare_drift,
    read_jsonl,
    read_reviews,
    submit_check,
    submit_review,
    summarise_predictions,
)


def tiny_artifact():
    legitimate = [preprocess_text("hello see you at the meeting tomorrow")] * 4
    scam = [preprocess_text("urgent claim your prize send the code now")] * 4
    classifier = make_classifier()
    classifier.fit(legitimate + scam, ["legitimate"] * 4 + ["scam"] * 4)
    return {"classifier": classifier, "model_version": "test-model"}


class UiDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_log_keeps_source_and_drops_message_text(self):
        log_path = self.root / "predictions.jsonl"
        secret = "SECRETPHRASE do not store this body"
        result = submit_check(tiny_artifact(), secret, "english", "email", log_path)
        self.assertEqual(result["source"], "email")
        self.assertNotIn("message", result)
        raw = log_path.read_text(encoding="utf-8")
        self.assertNotIn("SECRETPHRASE", raw)
        event = json.loads(raw)
        self.assertEqual(set(event), set(LOG_FIELDS))
        self.assertEqual(event["source"], "email")
        self.assertEqual(event["language"], "english")

    def test_review_stores_the_message_the_person_sent(self):
        review_path = self.root / "review_queue.jsonl"
        result = {
            "source": "phone",
            "language": "sinhala",
            "prediction": "scam",
            "scam_probability": 0.91,
            "drift_oov_ratio": 0.4,
            "drift_alert": True,
        }
        submit_review("please review this text", result, "felt wrong", review_path)
        saved = read_reviews(review_path)
        self.assertEqual(saved[0]["message"], "please review this text")
        self.assertEqual(saved[0]["note"], "felt wrong")
        self.assertTrue(saved[0]["drift_alert"])

    def test_summary_ignores_non_prediction_lines(self):
        events = [
            {"event": "drift_response", "action": "human_review_and_retraining_recommended"},
            {
                "prediction": "scam",
                "drift_oov_ratio": 0.5,
                "language": "tamil",
                "source": "phone",
            },
            {"prediction": "legitimate", "drift_oov_ratio": "not-a-number"},
        ]
        summary = summarise_predictions(events)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["by_source"]["phone"], 1)
        self.assertTrue(summary["drift_alert"])

    def test_missing_baseline_is_an_empty_comparison(self):
        result = compare_drift(
            tiny_artifact(),
            self.root / "missing.csv",
            self.root / "also-missing.csv",
        )
        self.assertFalse(result["available"])
        self.assertIn("messages_clean.csv", result["reason"])

    def test_rejects_an_unknown_source(self):
        with self.assertRaisesRegex(ValueError, "source"):
            submit_check(
                tiny_artifact(),
                "hello",
                "english",
                "whatsapp",
                self.root / "predictions.jsonl",
            )

    def test_reads_an_empty_log_as_no_rows(self):
        self.assertEqual(read_jsonl(self.root / "absent.jsonl"), [])
