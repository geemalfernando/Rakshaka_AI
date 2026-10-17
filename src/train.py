"""Train the scam classifier + save baseline stats used for drift detection."""
import json, pathlib, sys
from datetime import datetime, timezone
import joblib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rakshaka.data import load_labelled_messages
from rakshaka.modeling import char_ngram_oov_ratio, make_classifier, preprocess_text
from features import tokens, unk_rate

DATA, MODELS = ROOT / "data", ROOT / "models"
MODELS.mkdir(exist_ok=True)

def build():
    # char n-grams handle Sinhala/Tamil script + Singlish spelling variants (shared with rakshaka/)
    return make_classifier(class_weight="balanced")

def main(train_path=DATA / "train.csv", val_path=DATA / "val.csv", version=None):
    rows = load_labelled_messages(train_path)
    reviewed = DATA / "reviewed.csv"          # human-reviewed labels from the review queue
    n_reviewed = 0
    if reviewed.exists():
        extra = load_labelled_messages(reviewed, default_language="reviewed")
        n_reviewed = len(extra)
        rows += extra
    model = build().fit([preprocess_text(r["text"]) for r in rows], [r["label"] for r in rows])
    vocab = set(t for r in rows for t in tokens(r["text"]))

    # drift baselines from validation templates the model never saw
    # (not from training data, which is overconfident and has no unknown tokens)
    val = load_labelled_messages(val_path)
    conf = model.predict_proba([preprocess_text(r["text"]) for r in val]).max(axis=1)
    unk = [unk_rate(r["text"], vocab) for r in val]
    oov = [char_ngram_oov_ratio(model, r["text"]) for r in val]

    meta_path = MODELS / "meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    version = version or f"v{int(old.get('version', 'v0')[1:]) + 1}"
    baseline = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(rows),
        "n_reviewed": n_reviewed,
        "n_val": len(val),
        "mean_conf": float(np.mean(conf)),
        "scam_rate": float(np.mean([r["label"] == "scam" for r in rows])),
        "unk_rate": float(np.mean(unk)),
        "oov_rate": float(np.mean(oov)),
    }
    # same artifact format as scripts/train_model.py, so rakshaka.service and scripts/ can load it
    joblib.dump({"classifier": model, "model_version": version,
                 "languages": sorted({r["language"] for r in rows})}, MODELS / "model.joblib")
    (MODELS / "vocab.json").write_text(json.dumps(sorted(vocab), ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print("trained", baseline)

if __name__ == "__main__":
    main()
