"""Evaluate on the held-out test set, the shifted set and the independent set. Writes reports/metrics.json."""
import json, pathlib, sys
import joblib, numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rakshaka.data import load_labelled_messages
from rakshaka.modeling import char_ngram_oov_ratio, evaluate_by_language, evaluate_classifier, preprocess_text
from features import unk_rate

# optional argument: a model folder other than models/ (e.g. a clean model trained with --no-reviewed)
MODEL_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
artifact = joblib.load(MODEL_DIR / "model.joblib")
model = artifact["classifier"]
vocab = set(json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8")))
meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
out = {"model_version": artifact["model_version"], "baseline": meta}
# "independent" = data/extra_test.csv: written separately from the templates, never trained on or used for
# retraining decisions, so it stays an untouched final check.
SETS = {"test": ROOT / "data" / "test.csv", "shifted": ROOT / "data" / "shifted.csv"}
if (ROOT / "data" / "extra_test.csv").exists():
    SETS["independent"] = ROOT / "data" / "extra_test.csv"
for name, path in SETS.items():
    rows = load_labelled_messages(path)
    texts = [preprocess_text(r["text"]) for r in rows]
    actual = [r["label"] for r in rows]
    pred = model.predict(texts)
    conf = model.predict_proba(texts).max(axis=1)
    out[name] = {
        **evaluate_classifier(model, rows),
        "f1_scam": round(float(f1_score(actual, pred, pos_label="scam")), 4),
        "mean_conf": round(float(conf.mean()), 4),
        "unk_rate": round(float(np.mean([unk_rate(r["text"], vocab) for r in rows])), 4),
        "oov_rate": round(float(np.mean([char_ngram_oov_ratio(model, r["text"]) for r in rows])), 4),
        "by_language": evaluate_by_language(model, rows),
    }
    print(f"\n=== {name} ({len(rows)} rows, {out[name]['groups']} templates) ===")
    print(classification_report(actual, pred, labels=["legitimate", "scam"], digits=3, zero_division=0))
    print(f"mean confidence: {out[name]['mean_conf']:.3f}   unknown-token rate: {out[name]['unk_rate']:.1%}"
          f"   unseen char-ngram rate: {out[name]['oov_rate']:.1%}")
print(f"\nbaseline (validation): mean confidence {meta['mean_conf']:.3f}, "
      f"unknown-token rate {meta['unk_rate']:.1%}, unseen char-ngram rate {meta['oov_rate']:.1%}")
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
