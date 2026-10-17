"""Runs the real FastAPI app against a temporary RAKSHAKA_HOME with a tiny model."""
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from rakshaka.modeling import make_classifier, preprocess_text

TOKEN = "test-admin-token"
TEXTS = ["Your DemoBank account is blocked verify now at x.example", "Send your OTP to 07X-XXXXXXX now",
         "Claim your prize fee at y.example", "Meeting moved to 3pm today", "Amma I will come home at 7",
         "Class is cancelled tomorrow"]
LABELS = ["scam", "scam", "scam", "legitimate", "legitimate", "legitimate"]


def write_model(folder, version):
    folder.mkdir(parents=True, exist_ok=True)
    model = make_classifier().set_params(features__char__min_df=1, features__word__min_df=1)
    model.fit([preprocess_text(t) for t in TEXTS], LABELS)
    joblib.dump({"classifier": model, "model_version": version, "languages": ["english"]}, folder / "model.joblib")
    (folder / "vocab.json").write_text(json.dumps(["your", "account", "meeting"]), encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"version": version, "mean_conf": 0.8, "unk_rate": 0.2}),
                                      encoding="utf-8")


class AdminApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = Path(tempfile.mkdtemp())
        write_model(cls.home / "models", "v1")
        (cls.home / "data").mkdir()
        header = "text,label,language,group_id\n"
        (cls.home / "data" / "test.csv").write_text(
            header + "".join(f"Meeting moved to {h}pm today,legitimate,english,t{h}\n" for h in range(1, 13)), encoding="utf-8")
        (cls.home / "data" / "shifted.csv").write_text(
            header + "".join(f"Qwyx zrrv plomb {i} kkvv nnqq,scam,tanglish,s{i}\n" for i in range(40)), encoding="utf-8")
        os.environ.update(RAKSHAKA_HOME=str(cls.home), ADMIN_TOKEN=TOKEN, RETENTION_DAYS="7", SIM_MIN_INTERVAL="1")
        sys.modules.pop("api", None)
        cls.api = importlib.import_module("api")
        cls.client = TestClient(cls.api.app)

    @classmethod
    def tearDownClass(cls):
        for name in ("RAKSHAKA_HOME", "ADMIN_TOKEN", "RETENTION_DAYS", "SIM_MIN_INTERVAL"):
            os.environ.pop(name, None)
        sys.modules.pop("api", None)
        shutil.rmtree(cls.home, ignore_errors=True)

    def queue_message(self, text="Unseen wording zzqx qqvv"):
        r = self.client.post("/predict", json={"text": text, "language": "tanglish"}).json()
        self.assertEqual(r["routed"], "human_review")   # mostly unseen words -> review
        rows = self.client.get("/review_queue", headers={"X-Admin-Token": TOKEN}).json()
        return next(row for row in rows if row["text"] == text)

    def test_public_endpoints_need_no_token(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.post("/predict", json={"text": "Meeting at 3pm"}).status_code, 200)
        self.assertNotIn("drift", self.client.post("/predict", json={"text": "Meeting at 3pm"}).json())

    def test_admin_endpoints_reject_missing_or_wrong_token(self):
        for method, path, body in (("get", "/review_queue", None), ("post", "/review", {"items": []}),
                                   ("get", "/predictions", None), ("get", "/alerts", None),
                                   ("get", "/monitor", None), ("post", "/simulate", {"dataset": "shifted", "n": 20}),
                                   ("post", "/simulate/reset", None),
                                   ("post", "/retrain", None), ("post", "/rollback", None),
                                   ("delete", "/review_queue/1", None)):
            for headers in ({}, {"X-Admin-Token": "wrong"}):
                kwargs = {"headers": headers}
                if body is not None:
                    kwargs["json"] = body
                r = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(r.status_code, 401, f"{method} {path} {headers}")

    def test_predictions_feed_never_returns_text(self):
        self.client.post("/predict", json={"text": "secret words zzqx", "language": "english"})
        rows = self.client.get("/predictions", headers={"X-Admin-Token": TOKEN}).json()
        self.assertTrue(rows)
        for row in rows:
            self.assertNotIn("text", row)

    def test_labelling_erases_text_from_the_log(self):
        row = self.queue_message("Label me zzqx qqvv kkww")
        r = self.client.post("/review", json={"items": [{"rowid": row["rowid"], "is_scam": True}]},
                             headers={"X-Admin-Token": TOKEN})
        self.assertEqual(r.json()["saved"], 1)
        con = sqlite3.connect(self.home / "logs" / "predictions.db")
        self.assertIsNone(con.execute("SELECT text FROM predictions WHERE rowid=?", (row["rowid"],)).fetchone()[0])
        con.close()
        self.assertIn("Label me zzqx qqvv kkww", (self.home / "data" / "reviewed.csv").read_text(encoding="utf-8"))

    def test_discard_erases_text_without_labelling(self):
        row = self.queue_message("Discard me zzqx qqvv")
        r = self.client.delete(f"/review_queue/{row['rowid']}", headers={"X-Admin-Token": TOKEN})
        self.assertEqual(r.status_code, 200)
        texts = [q["text"] for q in self.client.get("/review_queue", headers={"X-Admin-Token": TOKEN}).json()]
        self.assertNotIn("Discard me zzqx qqvv", texts)

    def test_retention_erases_old_unreviewed_text(self):
        row = self.queue_message("Old message zzqx qqvv")
        con = sqlite3.connect(self.home / "logs" / "predictions.db")
        con.execute("UPDATE predictions SET ts=? WHERE rowid=?", (time.time() - 8 * 86400, row["rowid"]))
        con.commit(); con.close()
        self.assertGreaterEqual(self.api.purge_expired_text(), 1)
        texts = [q["text"] for q in self.client.get("/review_queue", headers={"X-Admin-Token": TOKEN}).json()]
        self.assertNotIn("Old message zzqx qqvv", texts)

    def test_rollback_restores_the_previous_model_and_back(self):
        headers = {"X-Admin-Token": TOKEN}
        models = self.home / "models"
        shutil.rmtree(models / "previous", ignore_errors=True)
        self.assertEqual(self.client.post("/rollback", headers=headers).status_code, 409)   # nothing to restore
        write_model(models / "previous", "v0")
        r = self.client.post("/rollback", headers=headers).json()
        self.assertEqual((r["model_version"], r["previous_version"]), ("v0", "v1"))
        self.assertEqual(self.client.get("/health").json()["model_version"], "v0")
        r = self.client.post("/rollback", headers=headers).json()   # swap back
        self.assertEqual(r["model_version"], "v1")
        kinds = [a["kind"] for a in self.client.get("/alerts", headers=headers).json()]
        self.assertIn("ROLLBACK", kinds)

    def simulate(self, dataset="shifted", n=25):
        self.api.state["last_sim"] = 0   # skip the minimum interval between independent tests
        return self.client.post("/simulate", json={"dataset": dataset, "n": n}, headers={"X-Admin-Token": TOKEN})

    def test_simulation_has_its_own_window_and_never_touches_live_metrics(self):
        headers = {"X-Admin-Token": TOKEN}
        self.client.post("/simulate/reset", headers=headers)
        live_before = self.client.get("/monitor", headers=headers).json()["n"]
        r = self.simulate("shifted", 25)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["sent"], 25)
        self.assertEqual(r.json()["monitor"]["status"], "DRIFT")          # unseen wording -> simulated drift
        self.assertEqual(self.client.get("/monitor", headers=headers).json()["n"], live_before)   # live untouched
        sim = self.client.get("/monitor", params={"mode": "simulation"}, headers=headers).json()
        self.assertEqual((sim["mode"], sim["status"]), ("simulation", "DRIFT"))
        sources = {row["source"] for row in self.client.get("/predictions", params={"source": "simulation"},
                                                             headers=headers).json()}
        self.assertEqual(sources, {"simulation"})
        kinds = [a["kind"] for a in self.client.get("/alerts", headers=headers).json()]
        self.assertIn("SIMULATION", kinds)          # audit entry
        self.assertIn("DRIFT_SIMULATION", kinds)    # never logged as a live DRIFT
        self.client.post("/simulate/reset", headers=headers)
        self.assertEqual(self.client.get("/monitor", params={"mode": "simulation"}, headers=headers).json()["n"], 0)

    def test_simulation_is_bounded_and_spaced_out(self):
        headers = {"X-Admin-Token": TOKEN}
        self.api.state["last_sim"] = 0
        self.assertEqual(self.client.post("/simulate", json={"dataset": "shifted", "n": 1000}, headers=headers).status_code, 422)
        self.assertEqual(self.client.post("/simulate", json={"dataset": "train", "n": 10}, headers=headers).status_code, 400)
        self.assertEqual(self.simulate("test", 5).status_code, 200)
        again = self.client.post("/simulate", json={"dataset": "test", "n": 5}, headers=headers)   # immediately again
        self.assertEqual(again.status_code, 429)

    def test_public_predict_is_rate_limited_per_client(self):
        original = self.api.PREDICT_PER_MINUTE
        self.api.PREDICT_PER_MINUTE = 3
        self.api.predict_calls.clear()
        try:
            codes = [self.client.post("/predict", json={"text": f"Meeting {i}"}).status_code for i in range(5)]
        finally:
            self.api.PREDICT_PER_MINUTE = original
            self.api.predict_calls.clear()
        self.assertEqual(codes, [200, 200, 200, 429, 429])


if __name__ == "__main__":
    unittest.main()
