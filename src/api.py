"""FastAPI service: prediction + logging + drift monitoring + human-review queue."""
import csv, hmac, json, os, shutil, sqlite3, hashlib, time, pathlib, logging, subprocess, sys, threading
from collections import deque
from typing import List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rakshaka.data import load_labelled_messages
from rakshaka.modeling import evaluate_classifier, retrain_gate
from rakshaka.monitoring import drift_signals
from rakshaka.service import load_artifact, predict as rakshaka_predict
from features import unk_rate

# RAKSHAKA_HOME lets tests run the real app against a temporary models/ logs/ data/ folder
HOME = pathlib.Path(os.getenv("RAKSHAKA_HOME", ROOT))
MODELS, LOGS, DATA = HOME / "models", HOME / "logs", HOME / "data"
LOGS.mkdir(exist_ok=True)
DB = LOGS / "predictions.db"

# ---- drift config (calibrated on data/val.csv; see README) ----
WINDOW = 50              # rolling window of recent predictions
MIN_WINDOW = 20          # predictions needed before drift is judged
CONF_DROP = 0.10         # alert if mean conf drops > 0.10 below baseline
UNK_RISE = 0.12          # alert if unseen-token rate rises >12 points above baseline (normal traffic stays >=11 below)
LOW_CONF = 0.60          # below this a prediction goes to human review (~3% of normal traffic)
MSG_UNK = 0.50           # a message that is mostly unseen words goes to human review
# ---- retraining gate: a candidate replaces the live model only if it is not worse on a fixed regression set ----
GATE_SETS = (DATA / "val.csv", DATA / "test.csv")   # fixed regression sets, never trained on
# data/extra_test.csv is deliberately NOT in the gate: it stays an untouched final test for reported results.
GATE_MAX_DROP = 0.01          # allowed drop in balanced accuracy, scam recall or legitimate recall
MODEL_FILES = ("model.joblib", "vocab.json", "meta.json")
# ---- admin access and privacy ----
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")   # required for review, retrain and rollback; unset = admin disabled
RETENTION_DAYS = float(os.getenv("RETENTION_DAYS", "7"))   # unreviewed message text is erased after this

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / "service.log", encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("rakshaka")

app = FastAPI(title="Rakshaka AI")
state = {}
window = deque(maxlen=WINDOW)
retrain_lock = threading.Lock()

def db():
    con = sqlite3.connect(DB)
    # Privacy (PDPA No. 9 of 2022): raw text is stored ONLY for messages routed to human review,
    # because a reviewer has to read them. Auto-handled messages keep just a hash.
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(ts REAL, text_hash TEXT, text TEXT, language TEXT,
                   label TEXT, confidence REAL, unk REAL, oov REAL, model_version TEXT, routed TEXT,
                   reviewed INT DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS alerts(ts REAL, kind TEXT, detail TEXT)""")
    return con

def load():
    if not (MODELS / "model.joblib").exists():
        raise RuntimeError("models/model.joblib not found: run `python src/make_data.py && cd src && python train.py`")
    state["artifact"] = load_artifact(MODELS / "model.joblib")
    state["vocab"] = set(json.loads((MODELS / "vocab.json").read_text(encoding="utf-8")))
    state["meta"] = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
    state["last_status"] = None
    window.clear()
    log.info("loaded model %s", state["meta"]["version"])

def purge_expired_text():
    """Privacy: erase stored message text older than RETENTION_DAYS (metadata and hashes stay)."""
    con = db()
    n = con.execute("UPDATE predictions SET text=NULL WHERE text IS NOT NULL AND ts < ?",
                    (time.time() - RETENTION_DAYS * 86400,)).rowcount
    con.commit(); con.close()
    if n:
        log.info("retention: erased text of %d messages older than %g days", n, RETENTION_DAYS)
    return n

def require_admin(x_admin_token: Optional[str] = Header(default=None)):
    """Admin endpoints need the X-Admin-Token header to match ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="admin endpoints are disabled: start the API with ADMIN_TOKEN set")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="valid X-Admin-Token header required")

load()
purge_expired_text()

class Msg(BaseModel):
    text: str
    language: str = "unknown"   # english | sinhala | singlish | tamil | tanglish | unknown

class ReviewItem(BaseModel):
    rowid: int
    is_scam: bool

class Review(BaseModel):
    items: List[ReviewItem]

def drift_status():
    thresholds = {"conf_drop": CONF_DROP, "unk_rise": UNK_RISE, "min_window": MIN_WINDOW, "window": WINDOW}
    d = drift_signals(list(window), state["meta"], CONF_DROP, UNK_RISE, MIN_WINDOW)   # rakshaka/monitoring.py
    return {**d, "baseline": state["meta"], "thresholds": thresholds}

@app.get("/health")
def health():
    return {"ok": True, "model_version": state["meta"]["version"]}

@app.post("/predict")
def predict(m: Msg):
    try:
        r = rakshaka_predict(state["artifact"], m.text, m.language)   # shared with rakshaka/service.py
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    is_scam = r["prediction"] == "scam"
    conf = max(r["scam_probability"], 1 - r["scam_probability"])
    unk = unk_rate(m.text, state["vocab"])
    window.append((conf, unk, is_scam))
    d = drift_status()
    # safe fallback: low confidence, mostly-unseen wording, or window drift -> human review,
    # and warn the user rather than say "safe"
    routed = "human_review" if (conf < LOW_CONF or unk >= MSG_UNK or d["status"] == "DRIFT") else "auto"
    verdict = ("SCAM" if is_scam else "LIKELY SAFE") if routed == "auto" else "UNCERTAIN - be careful, sent for review"
    con = db()
    cur = con.execute("INSERT INTO predictions(ts,text_hash,text,language,label,confidence,unk,oov,model_version,routed)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (time.time(), hashlib.sha256(m.text.encode()).hexdigest()[:16],
                       m.text if routed == "human_review" else None, r["language"], r["prediction"], conf, unk,
                       r["drift_oov_ratio"], state["meta"]["version"], routed))
    if d["status"] == "DRIFT" and state.get("last_status") != "DRIFT":   # alert once per drift episode
        con.execute("INSERT INTO alerts VALUES (?,?,?)", (time.time(), "DRIFT", "; ".join(d["signals"])))
        log.warning("DRIFT ALERT: %s -> routing predictions to human review", d["signals"])
    state["last_status"] = d["status"]
    con.commit(); con.close()
    log.info("pred id=%s lang=%s label=%s conf=%.3f unk=%.2f oov=%.2f routed=%s",
             cur.lastrowid, r["language"], r["prediction"], conf, unk, r["drift_oov_ratio"], routed)
    return {"label": r["prediction"], "verdict": verdict, "confidence": round(conf, 3),
            "scam_probability": round(r["scam_probability"], 3), "language": r["language"],
            "unknown_token_rate": round(unk, 3), "char_oov_ratio": round(r["drift_oov_ratio"], 3),
            "routed": routed, "drift": d["status"], "model_version": state["meta"]["version"]}

@app.get("/monitor")
def monitor():
    return drift_status()

@app.get("/predictions", dependencies=[Depends(require_admin)])
def predictions(limit: int = 200):
    """Recent prediction metadata for the dashboard charts. Never returns message text."""
    con = db()
    rows = con.execute("SELECT rowid,ts,language,label,confidence,unk,oov,model_version,routed FROM predictions "
                       "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    keys = ["rowid", "ts", "language", "label", "confidence", "unk", "oov", "model_version", "routed"]
    return [dict(zip(keys, r)) for r in reversed(rows)]

@app.get("/review_queue", dependencies=[Depends(require_admin)])
def review_queue(limit: int = 50):
    purge_expired_text()
    con = db()
    rows = con.execute("SELECT rowid,ts,text,language,label,confidence,unk FROM predictions "
                       "WHERE routed='human_review' AND reviewed=0 AND text IS NOT NULL ORDER BY ts DESC LIMIT ?",
                       (limit,)).fetchall()
    con.close()
    return [dict(zip(["rowid", "ts", "text", "language", "label", "confidence", "unk"], r)) for r in rows]

@app.post("/review", dependencies=[Depends(require_admin)])
def review(rv: Review):
    """Human labels from the review queue -> data/reviewed.csv (used by the next /retrain)."""
    con = db()
    path = DATA / "reviewed.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["text", "label", "language"])
        saved = 0
        for item in rv.items:
            row = con.execute("SELECT text, language FROM predictions WHERE rowid=? AND reviewed=0",
                              (item.rowid,)).fetchone()
            if row and row[0]:
                w.writerow([row[0], "scam" if item.is_scam else "legitimate", row[1]])
                # privacy: the labelled text now lives only in the training file, not in the prediction log
                con.execute("UPDATE predictions SET reviewed=1, text=NULL WHERE rowid=?", (item.rowid,))
                saved += 1
    con.commit(); con.close()
    log.info("human review saved %d labels", saved)
    return {"saved": saved}

@app.get("/alerts", dependencies=[Depends(require_admin)])
def alerts(limit: int = 20):
    con = db()
    rows = con.execute("SELECT ts,kind,detail FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(zip(["ts", "kind", "detail"], r)) for r in rows]

@app.post("/retrain", dependencies=[Depends(require_admin)])
def retrain():
    """Retraining trigger: train a candidate on train.csv + data/reviewed.csv (human labels), compare it with the
    live model on the fixed regression set, and hot-reload it only if it passes the gate."""
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="retraining already running")
    try:
        cand_dir = MODELS / "candidate"
        prev_meta = MODELS / "previous" / "meta.json"   # never reuse a version number after a rollback
        versions = [state["meta"]["version"]] + ([json.loads(prev_meta.read_text())["version"]] if prev_meta.exists() else [])
        cand_version = f"v{max(int(v[1:]) for v in versions) + 1}"
        r = subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--out", str(cand_dir),
                            "--version", cand_version], capture_output=True, text=True, cwd=str(ROOT / "src"))
        if r.returncode != 0:
            log.error("retrain failed: %s", r.stderr[-500:])
            return {"ok": False, "deployed": False, "model_version": state["meta"]["version"],
                    "log": (r.stdout or r.stderr)[-500:]}
        # leave out any gate message a reviewer has labelled (demo traffic can send test messages to review)
        reviewed_path = DATA / "reviewed.csv"
        reviewed = {r["text"] for r in load_labelled_messages(reviewed_path)} if reviewed_path.exists() else set()
        gate_rows = [r for p in GATE_SETS for r in load_labelled_messages(p) if r["text"] not in reviewed]
        current = evaluate_classifier(state["artifact"]["classifier"], gate_rows)
        candidate = evaluate_classifier(load_artifact(cand_dir / "model.joblib")["classifier"], gate_rows)
        deployed, reasons = retrain_gate(current, candidate, GATE_MAX_DROP)
        summary = lambda m: {"balanced_accuracy": round(m["balanced_accuracy"], 4),
                             "scam_recall": round(m["per_class"]["scam"]["recall"], 4),
                             "legitimate_recall": round(m["per_class"]["legitimate"]["recall"], 4)}
        gate = {"set": " + ".join(p.name for p in GATE_SETS), "rows": len(gate_rows), "current": summary(current),
                "candidate": summary(candidate), "reasons": reasons}
        con = db()
        if deployed:
            prev = MODELS / "previous"   # keep the outgoing model so /rollback can restore it
            prev.mkdir(exist_ok=True)
            for name in MODEL_FILES:
                shutil.copy2(MODELS / name, prev / name)
                os.replace(cand_dir / name, MODELS / name)
            load()
            con.execute("INSERT INTO alerts VALUES (?,?,?)", (time.time(), "RETRAIN_DEPLOYED",
                        f"{cand_version} passed the regression gate"))
            log.info("retrain: %s deployed, gate %s", cand_version, gate)
        else:
            con.execute("INSERT INTO alerts VALUES (?,?,?)", (time.time(), "RETRAIN_REJECTED",
                        f"{cand_version} kept out: " + "; ".join(reasons)))
            log.warning("retrain: %s rejected, live model kept: %s", cand_version, reasons)
        con.commit(); con.close()
        return {"ok": True, "deployed": deployed, "model_version": state["meta"]["version"],
                "candidate_version": cand_version, "gate": gate, "log": r.stdout[-500:]}
    finally:
        retrain_lock.release()

@app.delete("/review_queue/{rowid}", dependencies=[Depends(require_admin)])
def discard(rowid: int):
    """Privacy: drop a queued message without labelling it; its text is erased immediately."""
    con = db()
    n = con.execute("UPDATE predictions SET reviewed=2, text=NULL WHERE rowid=? AND reviewed=0", (rowid,)).rowcount
    con.commit(); con.close()
    if not n:
        raise HTTPException(status_code=404, detail="no unreviewed message with that id")
    return {"discarded": rowid}

@app.post("/rollback", dependencies=[Depends(require_admin)])
def rollback():
    """Swap the live model with the one it replaced (models/previous). Calling it again swaps back."""
    prev = MODELS / "previous"
    if not all((prev / name).exists() for name in MODEL_FILES):
        raise HTTPException(status_code=409, detail="no previous model to roll back to")
    with retrain_lock:
        tmp = MODELS / "rollback_tmp"
        tmp.mkdir(exist_ok=True)
        from_version = state["meta"]["version"]
        for name in MODEL_FILES:
            shutil.copy2(MODELS / name, tmp / name)
            os.replace(prev / name, MODELS / name)
            os.replace(tmp / name, prev / name)
        load()
        con = db()
        con.execute("INSERT INTO alerts VALUES (?,?,?)", (time.time(), "ROLLBACK",
                    f"{from_version} -> {state['meta']['version']}"))
        con.commit(); con.close()
    log.warning("rollback: %s -> %s", from_version, state["meta"]["version"])
    return {"ok": True, "model_version": state["meta"]["version"], "previous_version": from_version}
