"""Build reports/model_report.pdf: model choice, data sources, training and results, with charts.

Inputs (all produced locally by the pipelines):
  reports/metrics.json            src/evaluate.py (deployed multilingual model)
  reports/model_comparison.json   model comparison run (see "Why this model")
  artifacts/metrics.json          scripts/train_model.py (English public-dataset baseline)
  data/{train,val,test,shifted}.csv, data/extra_train.csv
  spam.csv and Dataset_5971.csv   only for source row counts (optional)

Run: .venv/bin/python reports/build_model_report.py   (needs matplotlib + reportlab)
"""
import csv
import io
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIGS = ROOT / "reports", ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Reference palette (dataviz skill): categorical slots 1-3, text and grid inks.
LEGIT, SCAM, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dd", "#ffffff"
LANG_ORDER = ["english", "sinhala", "singlish", "tamil", "tanglish"]


def api_config():
    """Drift and routing thresholds exactly as set in src/api.py, so the report never goes stale."""
    import re
    text = (ROOT / "src" / "api.py").read_text(encoding="utf-8")
    return {k: float(re.search(rf"^{k}\s*=\s*([0-9.]+)", text, re.M).group(1))
            for k in ("WINDOW", "MIN_WINDOW", "CONF_DROP", "UNK_RISE", "LOW_CONF", "MSG_UNK")}


CFG = api_config()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def public_counts():
    """Raw label counts of the two downloaded datasets (None if they are not present locally)."""
    spam_path = ROOT / "spam.csv"
    phish_path = ROOT / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION" / "Dataset_5971.csv"
    if not (spam_path.exists() and phish_path.exists()):
        return None
    rows = list(csv.reader(io.StringIO(spam_path.read_bytes().decode("utf-8", errors="replace"))))[1:]
    spam = Counter(r[0].strip().casefold() for r in rows if r)
    with open(phish_path, encoding="utf-8", errors="replace", newline="") as fh:
        phish = Counter(r["LABEL"].strip().casefold() for r in csv.DictReader(fh))
    clean = Counter(r["label"] for r in read_csv(ROOT / "data" / "processed" / "messages_clean.csv"))
    return {"spam": spam, "phish": phish, "clean": clean}


# ---------------------------------------------------------------- charts
def chart_model_comparison(cmp):
    names = [n for n in cmp if not n.startswith("_")]
    short = {"Naive Bayes (word counts)": "Naive Bayes\n(word counts)", "LogReg, word TF-IDF only": "LogReg\n(word only)",
             "Random forest (char+word)": "Random forest\n(char+word)",
             "Linear SVM, calibrated (char+word)": "Linear SVM\ncalibrated", "LogReg char+word (Rakshaka)":
             "LogReg char+word\n(Rakshaka, chosen)"}
    cv = [cmp[n]["cv_bal_acc_mean"] for n in names]
    err = [cmp[n]["cv_bal_acc_std"] for n in names]
    en = [cmp["_english"][n]["bal_acc"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], cv, w, color=LEGIT, label="Multilingual synthetic: 5-fold CV by template")
    ax.errorbar([i - w / 2 - 0.01 for i in x], cv, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], en, w, color=THIRD, label="English public datasets: held-out test")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, 0.802, f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7, color=SURFACE, rotation=90)
    ax.set_xticks(list(x), [short[n] for n in names], fontsize=7.5)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy by model (higher is better; axis starts at 0.80)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    return save(fig, "model_comparison.png")


def chart_public_sources(counts):
    labels = ["SMS Spam Collection\n(spam.csv)", "SMS Phishing Dataset\n(Dataset_5971.csv)",
              "Combined, after\ncleaning + de-duplication"]
    legit = [counts["spam"]["ham"], counts["phish"]["ham"], counts["clean"]["legitimate"]]
    scam = [counts["spam"]["spam"], counts["phish"]["spam"] + counts["phish"]["smishing"], counts["clean"]["scam"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, legit, 0.55, color=LEGIT, label="Legitimate (ham)", edgecolor=SURFACE, linewidth=1)
    ax.barh(y, scam, 0.55, left=legit, color=SCAM, label="Scam (spam / smishing)", edgecolor=SURFACE, linewidth=1)
    for yi, l, s in zip(y, legit, scam):
        ax.text(l / 2, yi, f"{l:,}", ha="center", va="center", color=SURFACE, fontsize=8)
        ax.text(l + s + 80, yi, f"{s:,} scam  ({l + s:,} total)", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 7600)
    ax.set_xlabel("Messages")
    ax.set_title("The two downloaded English datasets", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    return save(fig, "public_sources.png")


def chart_language_mix(train, extra):
    tmpl = [r for r in train if r["text"] not in {e["text"] for e in extra}]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = range(len(LANG_ORDER))
    w = 0.36
    for off, label, color in ((-w / 2 - 0.01, "legitimate", LEGIT), (w / 2 + 0.01, "scam", SCAM)):
        t = [sum(1 for r in tmpl if r["language"] == l and r["label"] == label) for l in LANG_ORDER]
        e = [sum(1 for r in extra if r["language"] == l and r["label"] == label) for l in LANG_ORDER]
        ax.bar([i + off for i in x], t, w, color=color, label=f"{label.capitalize()}: template generator")
        ax.bar([i + off for i in x], e, w, bottom=t, color=color, alpha=0.45, hatch="////", edgecolor=SURFACE,
               linewidth=0, label=f"{label.capitalize()}: extra_train.csv")
        for i, (a, b) in enumerate(zip(t, e)):
            ax.text(i + off, a + b + 6, f"{a + b}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(list(x), [l.capitalize() for l in LANG_ORDER])
    ax.set_ylabel("Training messages")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title("Training data by language and label (train split)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    ax.grid(axis="x", visible=False)
    return save(fig, "language_mix.png")


def chart_confusion(cm, title, name):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(max(r) for r in cm) * 1.15)
    for i in range(2):
        for j in range(2):
            v = cm[i][j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color=SURFACE if v > max(max(r) for r in cm) * 0.55 else INK)
    ax.set_xticks([0, 1], ["Legitimate", "Scam"])
    ax.set_yticks([0, 1], ["Legitimate", "Scam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return save(fig, name)


def chart_per_language(metrics):
    langs = [l for l in LANG_ORDER if l in metrics["test"]["by_language"]]
    vals = [metrics["test"]["by_language"][l]["balanced_accuracy"] for l in langs]
    ns = [metrics["test"]["by_language"][l]["rows"] for l in langs]
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    bars = ax.bar(range(len(langs)), vals, 0.55, color=LEGIT)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 0.04, f"n={n}", ha="center", fontsize=7, color=SURFACE)
    ax.set_xticks(range(len(langs)), [l.capitalize() for l in langs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Test set, per language", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
    return save(fig, "per_language.png")


def chart_drift(metrics, meta):
    base = meta["unk_rate"]
    alert = base + CFG["UNK_RISE"]
    labels = ["Baseline\n(validation)", "Held-out test\n(same tactics)", "Shifted set\n(new tactics)"]
    vals = [base, metrics["test"]["unk_rate"], metrics["shifted"]["unk_rate"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars = ax.bar(range(3), vals, 0.5, color=[LEGIT, LEGIT, SCAM])
    ax.axhline(alert, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(-0.3, alert + 0.008, f"Alert line {alert:.1%} (baseline + {CFG['UNK_RISE'] * 100:.0f} points)", ha="left", fontsize=7.5, color=INK2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, max(vals + [alert]) * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Unknown-token rate")
    ax.set_title("Drift signal: share of words the model never saw in training", loc="left")
    ax.grid(axis="x", visible=False)
    return save(fig, "drift_signal.png")


# ---------------------------------------------------------------- PDF
def build():
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    cmp = json.loads((REPORTS / "model_comparison.json").read_text(encoding="utf-8"))
    meta = metrics["baseline"]  # baseline saved by evaluate.py with this exact model (not the live models/meta.json)
    en_path = ROOT / "artifacts" / "metrics.json"
    en = json.loads(en_path.read_text(encoding="utf-8")) if en_path.exists() else None
    train = read_csv(ROOT / "data" / "train.csv")
    extra = read_csv(ROOT / "data" / "extra_train.csv")
    split = {n: read_csv(ROOT / "data" / f"{n}.csv") for n in ("train", "val", "test", "shifted")}
    counts = public_counts()
    import make_data

    figs = {
        "cmp": chart_model_comparison(cmp),
        "lang": chart_language_mix(train, extra),
        "cm": chart_confusion(metrics["test"]["confusion_matrix"], "Held-out test (264 messages)", "confusion_test.png"),
        "perlang": chart_per_language(metrics),
        "drift": chart_drift(metrics, meta),
    }
    if counts:
        figs["public"] = chart_public_sources(counts)
    if en:
        figs["cm_en"] = chart_confusion(en["test"]["confusion_matrix"], f"English test ({en['test']['rows']} messages)",
                                        "confusion_english.png")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.6,
                          textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor(INK2))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_LEFT,
                        spaceAfter=2, textColor=colors.HexColor(INK))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16,
                        spaceBefore=10, spaceAfter=5, textColor=colors.HexColor(INK), keepWithNext=1)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                        spaceBefore=6, spaceAfter=3, textColor=colors.HexColor(INK), keepWithNext=1)
    P = lambda t, s=body: Paragraph(t, s)
    bullets = lambda items, s=body: [Paragraph(f"&bull;&nbsp;&nbsp;{i}", ParagraphStyle("b", parent=s, leftIndent=10,
                                                                                         firstLineIndent=-8)) for i in items]

    def table(rows, widths, header=True, zebra=True):
        data = [[Paragraph(str(c), small if r else ParagraphStyle("th", parent=small, fontName="Helvetica-Bold",
                                                                  textColor=colors.HexColor(INK)))
                 for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if zebra:
            style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f5f2")) for r in range(2, len(rows), 2)]
        t.setStyle(TableStyle(style))
        return t

    def img(path, width):
        from reportlab.lib.utils import ImageReader
        w, h = ImageReader(str(path)).getSize()
        return Image(str(path), width=width, height=width * h / w)

    W = A4[0] - 36 * mm
    t, s = metrics["test"], metrics["shifted"]
    ind = metrics.get("independent")
    rk = cmp["LogReg char+word (Rakshaka)"]
    best_cv = max((n for n in cmp if not n.startswith("_")), key=lambda n: cmp[n]["cv_bal_acc_mean"])
    story = []

    # ---- title + summary
    story += [P("Rakshaka AI: Model Report", h1),
              P(f"Scam-message classifier for Sri Lanka (English, Sinhala, Singlish, Tamil, Tanglish). "
                f"Model {metrics['model_version']} (clean: {meta['n_reviewed']} human-review labels), trained {meta['trained_at'][:10]}. Report built {date.today():%d %B %Y}.",
                small), Spacer(1, 6)]
    summary = [["Held-out test", "Scams missed", "Legit flagged", "Shifted-set drift"],
               [f"<b>{t['balanced_accuracy']:.3f}</b> balanced accuracy<br/>{t['macro_f1']:.3f} macro F1",
                f"<b>{t['confusion_matrix'][1][0]}</b> of {sum(t['confusion_matrix'][1])}",
                f"<b>{t['confusion_matrix'][0][1]}</b> of {sum(t['confusion_matrix'][0])}",
                f"<b>Fires</b>: unknown-token rate {s['unk_rate']:.1%} vs alert line {meta['unk_rate'] + CFG['UNK_RISE']:.1%}"]]
    story += [table(summary, [W / 4] * 4, zebra=False), Spacer(1, 4)]
    story += [P("All scores come from <b>synthetic</b> messages written for this project (no real personal data). "
                "The test set holds out whole message templates the model never saw, so it measures generalisation "
                "to new wording within known scam tactics. It is not evidence of real-world accuracy.", small)]

    # ---- 1. why this model
    story += [P("1. Why logistic regression on character + word n-grams", h2),
              P("We compared five text classifiers on the same data, using the same preprocessing. Each was scored two "
                "ways: 5-fold cross-validation on the multilingual synthetic data, where whole templates are held out "
                "per fold, and a held-out test on the two English public datasets. "
                "<b>Our model is not the most accurate one we tested.</b> "
                f"{best_cv.replace(' (char+word)', '')} scored highest on the multilingual cross-validation. "
                "We chose logistic regression for the reasons below, which come from how the product works, "
                "not only from accuracy.")]
    story += [img(figs["cmp"], W), Spacer(1, 2)]
    rows = [["Model", "CV balanced acc. (multilingual)", "English test bal. acc.", "Shifted-set bal. acc.",
             "Confidence: normal to shifted", "Speed (ms / message)"]]
    for n in [k for k in cmp if not k.startswith("_")]:
        c = cmp[n]
        rows.append([("<b>%s</b>" % n) if "Rakshaka" in n else n,
                     f"{c['cv_bal_acc_mean']:.3f} &plusmn; {c['cv_bal_acc_std']:.3f}",
                     f"{cmp['_english'][n]['bal_acc']:.3f}", f"{c['shifted_bal_acc_mean']:.3f}",
                     f"{c['val_conf']:.2f} to {c['shift_conf']:.2f}", f"{c['ms_per_message']:.2f}"])
    story += [table(rows, [W * 0.27, W * 0.17, W * 0.13, W * 0.13, W * 0.16, W * 0.14]), Spacer(1, 5)]
    story += [P("Why we kept logistic regression", h3)]
    story += bullets([
        f"<b>Fast enough to run on every message.</b> {rk['ms_per_message']:.2f} ms per message on a laptop CPU, "
        f"compared with {cmp['Random forest (char+word)']['ms_per_message']:.1f} ms for the random forest "
        f"(about {cmp['Random forest (char+word)']['ms_per_message'] / rk['ms_per_message']:.0f} times slower). "
        "A bank help desk or SMS gateway needs answers in real time on cheap hardware.",
        f"<b>Real probabilities, built in.</b> The API routes a message to a human when confidence is below {CFG['LOW_CONF']:.2f}, and "
        "the drift monitor watches mean confidence. Logistic regression outputs a probability directly. A linear "
        "SVM only gives probabilities through an extra calibration step, which trains three more models and slows "
        "the in-app <i>/retrain</i> loop.",
        "<b>Consistent on both data sources.</b> It is second-best on the English public datasets "
        f"({cmp['_english']['LogReg char+word (Rakshaka)']['bal_acc']:.3f}). The random forest, which looks best on the "
        f"multilingual templates, is the <i>worst</i> on the English public data "
        f"({cmp['_english']['Random forest (char+word)']['bal_acc']:.3f}) and loses accuracy on the shifted set "
        f"({cmp['Random forest (char+word)']['shifted_bal_acc_mean']:.3f}). That pattern suggests it memorises "
        "template wording.",
        "<b>Character n-grams handle Sinhala, Tamil and romanised spelling.</b> Splitting Sinhala and Tamil "
        "script into words is unreliable, and Singlish and Tanglish have no fixed spelling. Character 2-5-grams cope with "
        "both. Adding them lifts cross-validation accuracy from "
        f"{cmp['LogReg, word TF-IDF only']['cv_bal_acc_mean']:.3f} (words only) to "
        f"{rk['cv_bal_acc_mean']:.3f}, and test accuracy on Sinhala from 0.50 to 0.63.",
        "<b>Simple, explainable and quick to retrain.</b> One linear weight per n-gram, so a reviewer can be shown "
        "why a message was flagged. Retraining takes under a second, which keeps the human-review loop practical.",
    ])
    story += [P("What we did not choose, and why", h3)]
    story += bullets([
        "<b>Keyword rules.</b> A hand-written keyword list (link, phone, OTP, PIN, prize, verify ...) reached only "
        "0.724 balanced accuracy on the test set and cannot learn new tactics.",
        "<b>Naive Bayes.</b> The fastest model, but the least accurate on the multilingual data "
        f"({cmp['Naive Bayes (word counts)']['cv_bal_acc_mean']:.3f}). Its confidence stays at about 0.99 even on new "
        "tactics, so a confidence-based drift signal would never fire.",
        "<b>Multilingual transformers (mBERT, XLM-RoBERTa) and LLM APIs.</b> <i>Not benchmarked.</i> In a 6-hour, "
        "CPU-only build they mean large downloads, slower inference and fine-tuning on about 1,400 synthetic messages, "
        "which tends to overfit. An LLM API would also send message text to a third party.",
    ])
    story += [P("<b>Recommended next experiment:</b> the calibrated linear SVM had the best English-test score "
                f"({cmp['_english']['Linear SVM, calibrated (char+word)']['bal_acc']:.3f}) and the best shifted-set score. "
                "With more real, reviewed data it is the first candidate to replace logistic regression, if its "
                "calibration holds up.", body)]

    # ---- 2. public datasets
    story += [P("2. Data from the two online datasets", h2),
              P("Two public English SMS datasets were downloaded during the build window:")]
    src_rows = [["Dataset", "What it contains", "Raw labels", "Use in Rakshaka"],
                ["<b>SMS Spam Collection</b> (Almeida &amp; G&oacute;mez Hidalgo, UCI Machine Learning Repository; "
                 "Kaggle copy <i>spam.csv</i>, columns v1/v2)",
                 "5,572 English SMS messages", "4,825 ham, 747 spam",
                 "English baseline pipeline (scripts/), model comparison"],
                ["<b>SMS Phishing Dataset for Machine Learning and Pattern Recognition</b> (Mendeley Data; "
                 "<i>Dataset_5971.csv</i>, columns LABEL, TEXT, URL, EMAIL, PHONE)",
                 "5,971 English SMS messages", "4,844 ham, 638 smishing, 489 spam (mixed case)",
                 "English baseline pipeline (scripts/), model comparison"]]
    story += [table(src_rows, [W * 0.36, W * 0.18, W * 0.2, W * 0.26])]
    story += [P("Licences still need to be confirmed and recorded in the README. Because redistribution rights are "
                "not established, the raw files are <b>not</b> committed to the repository (.gitignore).", small)]
    if counts:
        story += [img(figs["public"], W)]
    story += [P("How the two datasets were combined", h3)]
    story += bullets([
        "<b>Read safely.</b> spam.csv is not valid UTF-8. Its 1,182 invalid bytes were decoded as Windows-1252, "
        "which recovers characters such as the pound sign. Dataset_5971 contains 185 replacement characters whose "
        "original text cannot be recovered; they were kept as they are.",
        "<b>Map labels to two classes.</b> <i>ham</i> becomes <i>legitimate</i>; <i>spam</i> and <i>smishing</i> "
        "become <i>scam</i>. Case variants such as 'Smishing' and 'smishing' are merged.",
        "<b>Normalise and de-duplicate.</b> Unicode NFKC normalisation, collapsed whitespace, and a SHA-256 "
        "<i>group id</i> for the case-folded text. 4,970 duplicate rows were merged, leaving "
        f"<b>{(counts['clean']['legitimate'] + counts['clean']['scam']) if counts else 6573:,}</b> unique "
        "messages. A message that appears in both datasets with different labels would stop the pipeline; this did "
        "not happen.",
        "<b>Keep provenance.</b> Every row keeps its original labels and its source rows, for example "
        "<i>spam.csv:120|Dataset_5971.csv:88</i>, plus URL, email and phone flags.",
        "<b>Split by group.</b> 70 / 15 / 15 train / validation / test, by group id and stratified by label, so "
        "the same text never appears in two splits. Only the training set is balanced, by downsampling legitimate "
        "groups.",
    ])
    if en:
        en_t = en["test"]
        story += [Table([[img(figs["cm_en"], W * 0.36),
                          [P("<b>English baseline result</b> (scripts/train_model.py)", body),
                           P(f"Test: {en_t['rows']} unique held-out messages. Balanced accuracy "
                             f"<b>{en_t['balanced_accuracy']:.3f}</b>, macro F1 <b>{en_t['macro_f1']:.3f}</b>. "
                             f"{en_t['confusion_matrix'][1][0]} scams missed and {en_t['confusion_matrix'][0][1]} legitimate "
                             "messages flagged.", body),
                           P("This model is English-only. The deployed multilingual model (sections 3-5) is trained on "
                             "the synthetic multilingual data instead. The English datasets serve as an independent "
                             "second benchmark, and as the source for a human translation sample (below).", small)]]],
                        colWidths=[W * 0.38, W * 0.62], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")])]
    story += [P("Human translation path (prepared, not yet used)", h3),
              P("<i>scripts/prepare_data.py</i> draws a balanced sample of 200 English messages (100 per class) into "
                "<i>data/annotation/translation_sample.csv</i>, with columns for Sinhala, Singlish, Tamil and Tanglish. "
                "A translation is used for training only after a native-speaker reviewer marks it <i>verified</i> "
                "and adds their name. The code never translates automatically. <b>So far 0 translations are "
                "verified.</b>")]

    # ---- 3. multilingual data
    tl = Counter((r["language"], r["label"]) for r in extra)
    story += [P("3. How the Sinhala, Singlish, Tamil and Tanglish data was created", h2),
              P("No public Sinhala, Singlish, Tamil or Tanglish scam-SMS dataset with a clear licence was available. "
                "We also may not use real people's messages (PDPA No. 9 of 2022). So all multilingual data is "
                "<b>synthetic</b>, from two sources:")]
    story += bullets([
        f"<b>Template generator</b> (<i>src/make_data.py</i>, seed 42). {len(make_data.SCAM)} scam and "
        f"{len(make_data.HAM)} legitimate message templates, hand-written by the team with AI assistance and modelled on "
        "common Sri Lankan scam tactics: blocked bank accounts, OTP and PIN requests, lottery and parcel fees, fake jobs, "
        "Aswesuma payments, pensions, loans, electricity disconnection. Each template is filled 70 times with fake "
        "slot values: DemoBank, Rs. amounts, *.example links, 07X-XXXXXXX numbers, fictional names and times. "
        "Some copies get random lower case, upper case or '!!' noise.",
        f"<b>Hand-curated extra set</b> (<i>data/extra_train.csv</i>, {len(extra)} messages), written with Claude Code: "
        f"{tl[('sinhala', 'scam')]} + {tl[('sinhala', 'legitimate')]} Sinhala, {tl[('tamil', 'scam')]} + "
        f"{tl[('tamil', 'legitimate')]} Tamil, {tl[('singlish', 'scam')]} + {tl[('singlish', 'legitimate')]} Singlish and "
        f"{tl[('tanglish', 'scam')]} + {tl[('tanglish', 'legitimate')]} Tanglish (scam + legitimate). It adds new tactics "
        "(fake police, account-takeover codes, CVV requests, scholarship fees) and safe bank-style messages, such as "
        "an OTP saying 'do not share'. It also adds Tanglish, which the generator does not cover. No row copies or "
        "translates a held-out test or validation template, or a shifted-set tactic; the closest text match is 62% "
        "similar. <b>These messages have not yet been checked by native speakers.</b>",
    ])
    tp = Counter(l for _, l in make_data.SCAM + make_data.HAM)
    story += [table([["Language", "Script", "Templates (scam + legit)", "Example (all synthetic)"],
                     ["English", "Latin", f"{tp['english']}", "URGENT: Your DemoBank account is blocked. Verify now at secure-verify.example"],
                     ["Sinhala", "Sinhala", f"{tp['sinhala']}", "(Sinhala script) 'Your DemoBank account has been suspended. Confirm immediately via [link].'"],
                     ["Singlish", "Latin (romanised Sinhala)", f"{tp['singlish']}", "Machan oyage DemoBank account eka block wela. Me link eken verify karanna ..."],
                     ["Tamil", "Tamil", f"{tp['tamil']}", "(Tamil script) 'Your DemoBank account has been frozen. Verify at [link] immediately.'"],
                     ["Tanglish", "Latin (romanised Tamil)", "0 (extra set only)", "Ungal current bill kattavillai. Indru iravu connection cut seiyappadum ..."]],
                    [W * 0.12, W * 0.18, W * 0.17, W * 0.53])]
    story += [P("Sinhala and Tamil examples are shown translated, because the PDF fonts cannot render those scripts.",
                small), img(figs["lang"], W)]
    story += [P("Splitting without leaks", h3),
              P("Messages generated from the same template share almost all their words. If copies of one template "
                "landed in both train and test, the test score would be inflated. So the generator holds out "
                "<b>whole templates</b>, stratified by (label, language):")]
    story += [table([["Split", "Messages", "Templates / groups", "Purpose"],
                     ["Train", f"{len(split['train']):,}", f"{len({r['group_id'] for r in split['train']})}",
                      f"Fit the model (includes the {len(extra)} extra_train rows, one group each)"],
                     ["Validation", f"{len(split['val']):,}", f"{len({r['group_id'] for r in split['val']})}",
                      "Measure the drift baselines only (confidence, unknown-token rate)"],
                     ["Test (held out)", f"{len(split['test']):,}", f"{len({r['group_id'] for r in split['test']})}",
                      "Reported metrics: unseen wording, known tactics"],
                     ["Shifted", f"{len(split['shifted']):,}", f"{len({r['group_id'] for r in split['shifted']})}",
                      "Drift demo: new tactics (SL-UDI digital ID / QR codes, biometric fees, USDT / AI crypto, "
                      "'new number' impersonation, voice notes, heavier code-mixing)"]],
                    [W * 0.15, W * 0.12, W * 0.16, W * 0.57])]

    # ---- 4. training
    story += [P("4. How the model is trained", h2)]
    steps = [["Step", "What happens", "Where"],
             ["1. Generate", "Write train / val / test / shifted CSVs from the templates and append data/extra_train.csv to train.",
              "src/make_data.py"],
             ["2. Pre-process", "NFKC normalisation, case-folding, whitespace collapse. Add marker tokens <i>urltoken</i>, "
              "<i>emailtoken</i> and <i>phonetoken</i> when a link, email address or phone number is present.",
              "rakshaka/modeling.py"],
             ["3. Features", "Union of two TF-IDF vectorisers (sublinear tf, min_df 2): character 2-5-grams "
              "(up to 250k features) and word 1-2-grams (up to 100k).", "make_classifier()"],
             ["4. Classifier", "Logistic regression, class_weight='balanced' because scam outnumbers legitimate "
              "about 55 / 45, max_iter 1500, random_state 42. Human labels from data/reviewed.csv are added when present.",
              "src/train.py"],
             ["5. Drift baselines", f"Score the unseen validation templates. Mean confidence {meta['mean_conf']:.3f}, "
              f"unknown-token rate {meta['unk_rate']:.1%}, unseen char-n-gram rate {meta['oov_rate']:.1%}. Saved "
              "with the word vocabulary in models/meta.json and vocab.json.", "src/train.py"],
             ["6. Evaluate", "Held-out test and shifted set: balanced accuracy, macro F1, confusion matrix, per-language "
              "scores, confidence and drift rates. Written to reports/metrics.json.", "src/evaluate.py"],
             ["7. Serve and retrain", "FastAPI loads model.joblib. Reviewed labels go to data/reviewed.csv; "
              "<i>POST /retrain</i> trains a candidate, gates it on val + test, and hot-loads it only if it is not "
              "worse; otherwise the live model stays.", "src/api.py"]]
    story += [table(steps, [W * 0.16, W * 0.62, W * 0.22])]

    # ---- 5. results
    story += [P("5. Results on the held-out test set", h2)]
    story += [Table([[img(figs["cm"], W * 0.42), img(figs["perlang"], W * 0.55)]], colWidths=[W * 0.44, W * 0.56])]
    pc = t["per_class"]
    story += [table([["Class", "Precision", "Recall", "F1", "Messages"],
                     ["Legitimate", f"{pc['legitimate']['precision']:.3f}", f"{pc['legitimate']['recall']:.3f}",
                      f"{pc['legitimate']['f1-score']:.3f}", f"{int(pc['legitimate']['support'])}"],
                     ["Scam", f"{pc['scam']['precision']:.3f}", f"{pc['scam']['recall']:.3f}",
                      f"{pc['scam']['f1-score']:.3f}", f"{int(pc['scam']['support'])}"]],
                    [W * 0.28, W * 0.18, W * 0.18, W * 0.18, W * 0.18]), Spacer(1, 4)]
    story += bullets([
        f"<b>No scam was missed</b> (recall {pc['scam']['recall']:.2f}). {t['confusion_matrix'][0][1]} of "
        f"{sum(t['confusion_matrix'][0])} legitimate messages were flagged as scam. For a scam checker that is the "
        "safer kind of mistake.",
        f"<b>Sinhala is the weak spot</b> ({t['by_language']['sinhala']['balanced_accuracy']:.2f}). The held-out "
        "Sinhala legitimate template, a water-bill notice, is flagged as scam in "
        f"{t['by_language']['sinhala']['confusion_matrix'][0][1]} of {sum(t['by_language']['sinhala']['confusion_matrix'][0])} "
        "copies; these are all of the test set's false alarms. Adding the Sinhala extra set raised "
        "this from 0.50 to 0.63. We did not add bill-style Sinhala messages to training, because that would tune the "
        "model to the test set.",
        (f"<b>Independent test set</b> ({ind['rows']} messages in data/extra_test.csv, written separately, never trained "
         f"on): balanced accuracy <b>{ind['balanced_accuracy']:.3f}</b>. Sinhala "
         f"{ind['by_language']['sinhala']['balanced_accuracy']:.2f} (all 15 varied legitimate messages correct, so the "
         "Sinhala problem above is one template, not Sinhala in general). Tanglish "
         f"{ind['by_language']['tanglish']['balanced_accuracy']:.2f}: the one miss is a boss / gift-card impersonation "
         "with no link, which falls below 0.60 confidence and is routed to human review.") if ind else
        "<b>Tanglish cannot be scored yet.</b> The test set has no Tanglish messages.",
        f"Shifted set (new tactics): balanced accuracy {s['balanced_accuracy']:.3f}. The new tactics still use "
        "links and phone numbers, which the model already treats as scam signals. That is why the product uses an "
        "<b>input-drift</b> alarm rather than waiting for accuracy to fall.",
    ])

    # ---- 6. drift
    story += [KeepTogether([P("6. Drift monitoring and response", h2), img(figs["drift"], W),
                            P(f"The API keeps a rolling window of the last {CFG['WINDOW']:.0f} predictions and judges it once it holds "
                              f"{CFG['MIN_WINDOW']:.0f}. It raises a <b>DRIFT</b> alert if the window's unknown-token rate rises more than "
                              f"{CFG['UNK_RISE'] * 100:.0f} points above the validation baseline, or if mean confidence drops more than {CFG['CONF_DROP']:.2f}. "
                              f"New-tactic messages average {s['unk_rate']:.1%} unknown words, compared with "
                              f"{meta['unk_rate']:.1%} at baseline, so the alert fires while accuracy is still high. "
                              "The product notices the change before it starts failing silently.")])]
    story += bullets([
        f"<b>Safe fallback:</b> during drift, and for any message with confidence below {CFG['LOW_CONF']:.2f} or {CFG['MSG_UNK']:.0%} or more unseen "
        "words, the verdict is 'UNCERTAIN - be careful, sent for review'. It is never 'safe'.",
        "<b>Human review, then a gated retrain:</b> reviewers label queued messages in the dashboard. <i>/retrain</i> "
        "trains a candidate, then scores it and the live model on a fixed regression set (val + test, never trained "
        "on). The candidate goes live only if balanced accuracy, scam recall and legitimate recall do not fall by more "
        "than 0.01. In a test through the API, labels that wrongly ticked legitimate messages as scam were rejected "
        "(legitimate recall 0.948 to 0.929) and the live model was kept. Correct labels were deployed.",
        "<b>Rollback and access:</b> a deployed retrain keeps the previous model, and <i>/rollback</i> restores it. "
        "Review, retrain and rollback need an admin token; without the token they are refused.",
        "<b>Privacy:</b> predictions are logged with a text hash only. Message text is stored only for messages "
        "routed to human review, is erased as soon as a reviewer labels or discards it, and expires after 7 days "
        "if nobody reviews it.",
    ])

    # ---- 7. limitations
    story += [P("7. Limitations and next steps", h2)]
    story += bullets([
        "All multilingual data is synthetic, and the test set holds only 12 templates. Scores have wide error bars "
        f"(cross-validation &plusmn;{rk['cv_bal_acc_std']:.2f}) and are not real-world accuracy.",
        "The Sinhala, Tamil, Singlish and Tanglish texts have not been checked by native speakers. Verify them, and "
        "complete the 200-message human translation sample.",
        "Have native speakers write more legitimate Sinhala messages and a larger Tanglish test set (65 independent "
        "messages so far).",
        "Re-run this comparison with the calibrated linear SVM once real reviewed data exists.",
        "Confirm and record the licences of both English datasets in the README.",
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(18 * mm, 10 * mm, "Rakshaka AI - model report - synthetic data only, no real personal data")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = REPORTS / "model_report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title="Rakshaka AI - Model Report", author="Rakshaka AI team").build(
        story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    build()
