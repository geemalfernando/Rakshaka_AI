import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshaka.data import load_labelled_messages, normalize_label

import make_data


class LabelledMessageTests(unittest.TestCase):
    def test_accepts_scaffold_numeric_and_named_labels(self):
        self.assertEqual(normalize_label("1"), "scam")
        self.assertEqual(normalize_label(0), "legitimate")
        self.assertEqual(normalize_label("ham"), "legitimate")
        self.assertEqual(normalize_label("Smishing"), "scam")
        with self.assertRaises(ValueError):
            normalize_label("maybe")

    def test_loads_extra_csv_with_default_language_and_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra_train.csv"
            path.write_text("text,label\nPay now at x.example,1\n  ,0\nSee you at 5,0\n", encoding="utf-8")
            rows = load_labelled_messages(path)
        self.assertEqual([r["label"] for r in rows], ["scam", "legitimate"])
        self.assertEqual({r["language"] for r in rows}, {"unknown"})
        self.assertTrue(all(len(r["group_id"]) == 64 for r in rows))


class TemplateSplitTests(unittest.TestCase):
    def test_no_template_appears_in_more_than_one_split(self):
        rows = make_data.gen(make_data.SCAM, "scam", "scam", 10) + make_data.gen(
            make_data.HAM, "legitimate", "legit", 10
        )
        train, val, test = make_data.split_templates(rows)
        groups = [{r["group_id"] for r in split} for split in (train, val, test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual({r["label"] for r in test}, {"scam", "legitimate"})
        self.assertEqual(len(train) + len(val) + len(test), len(rows))


if __name__ == "__main__":
    unittest.main()
