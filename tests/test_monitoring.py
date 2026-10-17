import random
import unittest

from rakshaka.monitoring import drift_signals

BASE = {"mean_conf": 0.83, "unk_rate": 0.19}
ARGS = dict(conf_drop=0.10, unk_rise=0.12, min_window=20)


def traffic(n, conf, unk, seed=0, jitter=0.03):
    rnd = random.Random(seed)
    return [(min(1.0, conf + rnd.uniform(-jitter, jitter)), max(0.0, unk + rnd.uniform(-jitter, jitter)), rnd.random() < 0.5)
            for _ in range(n)]


class DriftScenarioTests(unittest.TestCase):
    def test_low_volume_is_never_judged(self):
        d = drift_signals(traffic(19, 0.5, 0.9), BASE, **ARGS)
        self.assertEqual(d["status"], "WARMING_UP")
        self.assertEqual(d["signals"], [])

    def test_normal_traffic_does_not_alert(self):
        for seed in range(20):   # many random windows around the baseline
            self.assertEqual(drift_signals(traffic(50, 0.83, 0.16, seed), BASE, **ARGS)["status"], "OK")

    def test_new_tactics_raise_unknown_token_alert(self):
        d = drift_signals(traffic(50, 0.80, 0.39), BASE, **ARGS)
        self.assertEqual(d["status"], "DRIFT")
        self.assertTrue(any("unknown-token" in s for s in d["signals"]))

    def test_falling_confidence_alone_raises_alert(self):
        d = drift_signals(traffic(50, 0.68, 0.19), BASE, **ARGS)
        self.assertEqual(d["status"], "DRIFT")
        self.assertTrue(any("confidence" in s for s in d["signals"]))

    def test_mixed_language_window_alerts_once_shift_dominates(self):
        normal, shifted = traffic(50, 0.83, 0.15, 1), traffic(50, 0.78, 0.45, 2)
        self.assertEqual(drift_signals(normal[:40] + shifted[:10], BASE, **ARGS)["status"], "OK")    # 20% new mix
        self.assertEqual(drift_signals(normal[:20] + shifted[:30], BASE, **ARGS)["status"], "DRIFT")  # 60% new mix

    def test_after_retraining_on_reviewed_shift_window_returns_to_ok(self):
        # once the new wording is in the vocabulary, its unknown-token rate falls back near the baseline
        d = drift_signals(traffic(50, 0.81, 0.05), BASE, **ARGS)
        self.assertEqual(d["status"], "OK")


if __name__ == "__main__":
    unittest.main()
