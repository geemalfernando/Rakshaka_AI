import unittest

from rakshaka.modeling import retrain_gate


def metrics(balanced, scam_recall, legit_recall):
    return {"balanced_accuracy": balanced,
            "per_class": {"scam": {"recall": scam_recall}, "legitimate": {"recall": legit_recall}}}


class RetrainGateTests(unittest.TestCase):
    def test_candidate_that_is_as_good_or_better_is_deployed(self):
        ok, reasons = retrain_gate(metrics(0.90, 0.95, 0.85), metrics(0.905, 0.95, 0.86))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_small_noise_within_tolerance_is_allowed(self):
        ok, _ = retrain_gate(metrics(0.90, 0.95, 0.85), metrics(0.895, 0.945, 0.845), max_drop=0.01)
        self.assertTrue(ok)

    def test_mislabelled_legitimate_messages_are_caught(self):
        # reviewers ticking "scam" on legitimate messages lowers legitimate recall
        ok, reasons = retrain_gate(metrics(0.90, 0.95, 0.85), metrics(0.86, 0.97, 0.75))
        self.assertFalse(ok)
        self.assertTrue(any("legitimate_recall" in r for r in reasons))
        self.assertTrue(any("balanced_accuracy" in r for r in reasons))

    def test_missed_scams_are_caught_even_if_balanced_accuracy_holds(self):
        ok, reasons = retrain_gate(metrics(0.90, 0.98, 0.82), metrics(0.90, 0.90, 0.90))
        self.assertFalse(ok)
        self.assertEqual(len(reasons), 1)
        self.assertIn("scam_recall", reasons[0])


if __name__ == "__main__":
    unittest.main()
