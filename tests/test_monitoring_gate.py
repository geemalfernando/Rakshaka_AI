import unittest

from rakshaka.modeling import retrain_gate
from rakshaka.monitoring import drift_signals


def metrics(balanced=0.90, scam=0.95, legitimate=0.85):
    return {
        "balanced_accuracy": balanced,
        "per_class": {
            "scam": {"recall": scam},
            "legitimate": {"recall": legitimate},
        },
    }


class RetrainGateTests(unittest.TestCase):
    def test_accepts_identical_or_improved_candidate(self):
        accepted, reasons = retrain_gate(metrics(), metrics(0.94, 0.97, 0.88))
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_rejects_scam_recall_regression(self):
        accepted, reasons = retrain_gate(metrics(), metrics(0.91, 0.92, 0.90))
        self.assertFalse(accepted)
        self.assertTrue(any("scam_recall" in reason for reason in reasons))

    def test_rejects_legitimate_recall_regression(self):
        accepted, reasons = retrain_gate(metrics(), metrics(0.93, 0.96, 0.80))
        self.assertFalse(accepted)
        self.assertTrue(any("legitimate_recall" in reason for reason in reasons))

    def test_rejects_balanced_accuracy_regression(self):
        accepted, reasons = retrain_gate(metrics(), metrics(0.87, 0.95, 0.85))
        self.assertFalse(accepted)
        self.assertTrue(any("balanced_accuracy" in reason for reason in reasons))

    def test_accepts_drop_within_tolerance(self):
        accepted, reasons = retrain_gate(metrics(), metrics(0.895, 0.945, 0.845), 0.01)
        self.assertTrue(accepted)
        self.assertFalse(reasons)


class DriftSignalTests(unittest.TestCase):
    baseline = {"mean_conf": 0.82, "unk_rate": 0.18}

    def test_warmup_is_not_healthy(self):
        result = drift_signals([(0.2, 0.9, True)] * 5, self.baseline)
        self.assertEqual(result["status"], "WARMING_UP")
        self.assertEqual(result["n"], 5)

    def test_normal_window_stays_healthy(self):
        result = drift_signals([(0.80, 0.20, False)] * 25, self.baseline)
        self.assertEqual(result["status"], "OK")
        self.assertFalse(result["signals"])
        self.assertEqual(result["scam_rate"], 0.0)

    def test_confidence_drift_triggers(self):
        result = drift_signals([(0.65, 0.20, False)] * 20, self.baseline)
        self.assertEqual(result["status"], "DRIFT")
        self.assertTrue(any("confidence" in signal for signal in result["signals"]))

    def test_unknown_token_drift_triggers(self):
        result = drift_signals([(0.80, 0.45, True)] * 20, self.baseline)
        self.assertEqual(result["status"], "DRIFT")
        self.assertTrue(any("unknown-token" in signal for signal in result["signals"]))

    def test_combined_drift_and_summary(self):
        result = drift_signals([(0.6, 0.5, True)] * 20, self.baseline)
        self.assertEqual(result["status"], "DRIFT")
        self.assertEqual(len(result["signals"]), 2)
        self.assertEqual(result["scam_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
