"""FastAPI service: prediction + logging + drift monitoring + human-review queue."""
import csv, json, sqlite3, hashlib, time, pathlib, logging, subprocess, sys, threading
from collections import deque
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rakshaka.service import load_artifact, predict as rakshaka_predict
from features import unk_rate

MODELS, LOGS, DATA = ROOT / "models", ROOT / "logs", ROOT / "data"
LOGS.mkdir(exist_ok=True)
DB = LOGS / "predictions.db"

# ---- drift config (calibrated on data/val.csv; see README) ----
WINDOW = 50              # rolling window of recent predictions
MIN_WINDOW = 20          # predictions needed before drift is judged
CONF_DROP = 0.10         # alert if mean conf drops > 0.10 below baseline
UNK_RISE = 0.15          # alert if unseen-token rate rises >15 points above baseline
LOW_CONF = 0.60          # below this a prediction goes to human review (~3% of normal traffic)
MSG_UNK = 0.50           # a message that is mostly unseen words goes to human review

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

load()

class Msg(BaseModel):
    text: str
    language: str = "unknown"   # english | sinhala | singlish | tamil | tanglish | unknown

class ReviewItem(BaseModel):
    rowid: int
    is_scam: bool

class Review(BaseModel):
    items: List[ReviewItem]

def drift_status():
    if len(window) < MIN_WINDOW:
        return {"status": "WARMING_UP", "n": len(window), "signals": [], "baseline": state["meta"]}
    base = state["meta"]
    mean_conf = sum(w[0] for w in window) / len(window)
    mean_unk = sum(w[1] for w in window) / len(window)
    scam_rate = sum(w[2] for w in window) / len(window)
    signals = []
    if base["mean_conf"] - mean_conf > CONF_DROP:
        signals.append(f"confidence dropped {base['mean_conf']:.2f} -> {mean_conf:.2f}")
    if mean_unk - base["unk_rate"] > UNK_RISE:
        signals.append(f"unknown-token rate {base['unk_rate']:.0%} -> {mean_unk:.0%}")
    return {"status": "DRIFT" if signals else "OK", "signals": signals, "n": len(window),
            "mean_conf": round(mean_conf, 3), "unk_rate": round(mean_unk, 3),
            "scam_rate": round(scam_rate, 3), "baseline": base}

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

@app.get("/review_queue")
def review_queue(limit: int = 50):
    con = db()
    rows = con.execute("SELECT rowid,ts,text,language,label,confidence,unk FROM predictions "
                       "WHERE routed='human_review' AND reviewed=0 ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(zip(["rowid", "ts", "text", "language", "label", "confidence", "unk"], r)) for r in rows]

@app.post("/review")
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
                con.execute("UPDATE predictions SET reviewed=1 WHERE rowid=?", (item.rowid,))
                saved += 1
    con.commit(); con.close()
    log.info("human review saved %d labels", saved)
    return {"saved": saved}

@app.get("/alerts")
def alerts(limit: int = 20):
    con = db()
    rows = con.execute("SELECT ts,kind,detail FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(zip(["ts", "kind", "detail"], r)) for r in rows]

@app.post("/retrain")
def retrain():
    """Retraining trigger: uses data/reviewed.csv (human labels) + train.csv, then hot-reloads."""
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="retraining already running")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], capture_output=True, text=True,
                           cwd=str(ROOT / "src"))
        if r.returncode == 0:
            load()
        else:
            log.error("retrain failed: %s", r.stderr[-500:])
        return {"ok": r.returncode == 0, "model_version": state["meta"]["version"],
                "log": (r.stdout or r.stderr)[-500:]}
    finally:
        retrain_lock.release()
