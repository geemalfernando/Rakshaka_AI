import tempfile
import unittest
from pathlib import Path

from rakshaka.data import ANNOTATION_FIELDS, build_training_rows, load_verified_translations, write_csv
from rakshaka.modeling import char_ngram_oov_ratio, make_classifier
from rakshaka.service import LANGUAGE_VALUES, predict


class MultilingualTests(unittest.TestCase):
    def test_all_five_languages_are_supported(self):
        self.assertTrue({"english", "sinhala", "singlish", "tamil", "tanglish"} <= LANGUAGE_VALUES)

    def test_verified_tamil_tanglish_and_pending_translation(self):
        original = [{"group_id": "a", "label": "scam", "text": "Example message"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            write_csv(path, ANNOTATION_FIELDS, [{
                "group_id": "a", "label": "scam", "english_text": "Example message",
                "tamil_text": "சோதனை செய்தி", "tamil_status": "verified", "tamil_reviewer": "reviewer-1",
                "tanglish_text": "sothanai seithi", "tanglish_status": "verified", "tanglish_reviewer": "reviewer-2",
                "sinhala_text": "pending example", "sinhala_status": "pending",
            }])
            translated = load_verified_translations(path, original)
            rows = build_training_rows(original, translated)
            self.assertEqual({r["language"] for r in rows}, {"english", "tamil", "tanglish"})

    def test_drift_measure_and_prediction_contract(self):
        texts = ["Free reward claim", "Claim your free prize", "Meeting at the office",
                 "Office meeting is scheduled", "Win a free reward", "Scheduled office meeting"]
        labels = ["scam", "scam", "legitimate", "legitimate", "scam", "legitimate"]
        model = make_classifier()
        model.fit(texts, labels)
        ratio = char_ngram_oov_ratio(model, "தமிழ் சோதனை")
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1)
        response = predict({"classifier": model}, "தமிழ் சோதனை", "tamil")
        self.assertIn("drift_alert", response)
        self.assertEqual(response["language"], "tamil")


if __name__ == "__main__":
    unittest.main()
