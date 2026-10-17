"""Streamlit UI: check a message + live monitoring + drift charts + review/retrain. All data comes from the API."""
import html
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per request
API = os.getenv("API_URL", "http://127.0.0.1:8000")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGUAGES = ["unknown", "english", "sinhala", "singlish", "tamil", "tanglish"]
LABELS = {
    "unknown": "As written",
    "english": "English",
    "sinhala": "Sinhala",
    "singlish": "Singlish",
    "tamil": "Tamil",
    "tanglish": "Tanglish",
}
SAMPLES = [
    ("Tax refund", "english", "You have a pending tax refund of Rs. 1,000,000. Submit your card details at lk-refund.example"),
    ("Parcel fee", "sinhala", "ඔබගේ පාර්සලය රේගුවේ රඳවා ඇත. Rs. 2,500 ගෙවන්න tinyurl.example/abc"),
    ("Bank alert", "english", "DemoBank security: a payment of Rs. 48,500 was blocked. Share your PIN at pay-now.example to release it."),
]
CHART = {
    "background": "transparent",
    "axis": {
        "labelColor": "#55675b",
        "titleColor": "#55675b",
        "gridColor": "rgba(31,95,58,0.10)",
        "domainColor": "rgba(16,38,26,0.20)",
        "labelFont": "Inter, sans-serif",
        "titleFont": "Inter, sans-serif",
    },
    "legend": {"labelColor": "#10261a", "titleColor": "#55675b", "labelFont": "Inter, sans-serif"},
    "view": {"stroke": "transparent"},
    "font": "Inter, sans-serif",
}

def apply_theme():
    st.set_page_config(page_title="Rakshaka AI", page_icon="◈", layout="wide")
    st.markdown(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.html(
        """
        <script>
        (() => {
          if (window.__rkSpot) return;
          window.__rkSpot = true;
          const spot = document.createElement("div");
          spot.id = "rk-spot";
          document.body.appendChild(spot);
          document.addEventListener("pointermove", (event) => {
            spot.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
            const card = event.target.closest?.(".rk-card, [data-testid='stMetric'], .rk-banner, [data-testid='stTab']");
            if (!card) return;
            const box = card.getBoundingClientRect();
            card.style.setProperty("--lx", `${event.clientX - box.left}px`);
            card.style.setProperty("--ly", `${event.clientY - box.top}px`);
          }, {passive: true});
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def use_sample(text, language):
    st.session_state.draft = text
    st.session_state.lang = language
    st.session_state.result = None


def result_of(row):
    return "Sent for review" if row["routed"] == "human_review" else row["label"].capitalize()


def counts(series, column):
    c = series.value_counts()
    return pd.DataFrame({column: c.index, "Checks": c.values})


def kind_of(verdict):
    if verdict == "SCAM":
        return "scam"
    if "UNCERTAIN" in verdict:
        return "hold"
    return "safe"


def explain(kind):
    if kind == "scam":
        return "Treat this as a fraudulent money claim. Do not pay, share a PIN, or open the link."
    if kind == "hold":
        return "The claim is unfamiliar, or the live ledger has shifted. A reviewer reads it before anyone is told it is safe."
    return "The wording matches legitimate notices this desk has seen. Still confirm any payment with the bank directly."


def verdict_card(result):
    kind = kind_of(result["verdict"])
    probability = float(result["scam_probability"])
    stats = (
        ("Unseen words", f"{result['unknown_token_rate']:.0%}"),
        ("New letter patterns", f"{result['char_oov_ratio']:.0%}"),
        ("Model", html.escape(str(result["model_version"]))),
    )
    tiles = "".join(f'<div class="rk-stat"><b>{value}</b><span>{label}</span></div>' for label, value in stats)
    return f"""
    <div class="rk-card {kind}">
      <div class="rk-score">
        <strong>{probability:.0%}</strong>
        <span>Fraud score</span>
        <div class="rk-meter"><i style="width:{probability * 100:.1f}%"></i></div>
      </div>
      <div>
        <div class="rk-kicker">{html.escape(LABELS.get(result["language"], result["language"]))}</div>
        <div class="rk-verdict">{html.escape(result["verdict"])}</div>
        <p>{explain(kind)}</p>
        <div class="rk-stats">{tiles}</div>
      </div>
    </div>
    """


def bar_chart(frame, x, y, colors, percent=False):
    y_axis = alt.Axis(title=None, format=".0%" if percent else ".2f")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=36)
        .encode(
            x=alt.X(f"{x}:N", sort=list(frame[x]), axis=alt.Axis(title=None, labelAngle=0)),
            y=alt.Y(f"{y}:Q", axis=y_axis),
            color=alt.Color(f"{x}:N", legend=None, scale=alt.Scale(domain=list(frame[x]), range=colors)),
            tooltip=[x, alt.Tooltip(y, format=".1%" if percent else ".3f")],
        )
        .properties(height=240)
        .configure(**CHART)
    )


def line_chart(frame):
    long = frame.melt("Check", var_name="Series", value_name="Value")
    domain = ["Unknown-token rate", "Confidence", "Unknown-token alert line"]
    colors = ["#1f5f3a", "#2f7d5c", "#b3261e"]
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("Check:Q", axis=alt.Axis(title=None)),
            y=alt.Y("Value:Q", axis=alt.Axis(title=None, format=".0%")),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["Check", "Series", alt.Tooltip("Value:Q", format=".1%")],
        )
        .properties(height=280)
        .configure(**CHART)
    )


def offline_banner():
    st.markdown(
        '<div class="rk-banner bad"><span class="rk-dot"></span><div><strong>The checker is offline</strong>'
        f"<p>Start the API, then refresh. Expected at {html.escape(API)}.</p></div></div>",
        unsafe_allow_html=True,
    )


def load_health():
    try:
        health = requests.get(f"{API}/health", timeout=3)
        health.raise_for_status()
        return health.json().get("model_version", "v1")
    except requests.RequestException:
        offline_banner()
        return None


def render_hero(model_version, status_kind=None, status_label=None):
    pills = f'<span class="rk-pill">Model {html.escape(str(model_version))}</span>'
    if status_kind and status_label:
        pills += f'<span class="rk-pill {status_kind}"><span class="rk-dot"></span>{html.escape(status_label)}</span>'
    pills += '<span class="rk-pill">Synthetic claims only</span>'
    st.markdown(
        f'<section class="rk-hero"><div class="rk-brand"><span class="rk-hex"></span><span>Scam-proof by design</span><span class="rk-dim">// money claims</span></div>'
        f'<div class="rk-title">Rakshaka</div>'
        f'<p class="rk-deck">A fraud desk for SMS and WhatsApp money claims. It flags fake refunds, fee demands and bank impersonation in English, Sinhala, Tamil, Singlish and Tanglish, and holds a check when the wording starts to shift.</p>'
        f'<div class="rk-pills">{pills}</div></section>',
        unsafe_allow_html=True,
    )


def load_metrics():
    """Held-out results written by src/evaluate.py (synthetic data). None if the report is missing."""
    try:
        return json.loads((Path(ROOT) / "reports" / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_facts():
    metrics = load_metrics()
    facts = [("5", "Languages", "English, Sinhala, Tamil, Singlish, Tanglish")]
    if metrics:
        cm = metrics["test"]["confusion_matrix"]
        facts.append((f"{cm[1][1]}/{sum(cm[1])}", "Test scams caught", "held-out synthetic templates"))
        if "independent" in metrics:
            facts.append((f"{metrics['independent']['balanced_accuracy']:.0%}", "Independent test",
                          "Sinhala + Tanglish, never trained on"))
    facts.append(("24/7", "Drift watch", "new wording is held for human review"))
    tiles = "".join(f'<div class="rk-fact"><b>{html.escape(v)}</b><span>{html.escape(t)}</span>'
                    f'<small>{html.escape(d)}</small></div>' for v, t, d in facts)
    st.markdown(f'<div class="rk-strip">{tiles}</div>', unsafe_allow_html=True)


SIGNS = (
    ("!", "warn", "Asks for an OTP, PIN or CVV", "No bank or government office asks for these by SMS or WhatsApp."),
    ("Rs", "warn", "A fee to release money", "Prize, refund, loan or parcel 'release fees' are the most common trick."),
    ("⏱", "", "Urgency or a threat", "'Account blocked in 24 hours' or 'power cut tonight' is meant to rush you."),
    ("↗", "", "Unknown link or new number", "Open your bank's own app or call the number on your card instead."),
)


def signs_panel():
    items = "".join(f'<li class="{cls}"><i>{icon}</i><div><strong>{html.escape(title)}</strong>'
                    f'<p>{html.escape(body)}</p></div></li>' for icon, cls, title, body in SIGNS)
    st.markdown(f'<div class="rk-panel"><p class="rk-kicker">Before you reply</p><h3>Warning signs</h3>'
                f'<ul class="rk-signs">{items}</ul></div>', unsafe_allow_html=True)


NEXT_STEPS = {
    "scam": ("Do not pay, reply or open the link", "Block the sender and delete the message",
             "If you already shared a PIN or OTP, call your bank's official number now"),
    "hold": ("Do not act on it yet", "Check with the bank or office through its official number",
             "Ask a family member or the bank before paying anything"),
    "safe": ("Still confirm any payment in your bank's own app", "Never share an OTP or PIN, even with bank staff",
             "If anything feels wrong, check with the official number"),
}


def next_steps_panel(kind):
    items = "".join(f'<li><i>{n}</i><div><strong>{html.escape(t)}</strong></div></li>'
                    for n, t in enumerate(NEXT_STEPS[kind], start=1))
    st.markdown(f'<div class="rk-panel" style="margin-top:0.9rem"><p class="rk-kicker">What to do now</p>'
                f'<ul class="rk-signs">{items}</ul></div>', unsafe_allow_html=True)


def render_steps():
    steps = (
        ("01", "Paste the message", "Any SMS or WhatsApp money message, in five languages or a mix."),
        ("02", "The model scores it", "Character and word patterns, trained on synthetic Sri Lankan scam and safe messages."),
        ("03", "Unfamiliar wording is held", "When wording looks new, the answer is 'uncertain' and a person reviews it."),
        ("04", "Reviewed, then safely retrained", "Reviewers label new tactics; a new model goes live only if it is not worse."),
    )
    cards = "".join(f'<div class="rk-step"><em>{n}</em><strong>{html.escape(t)}</strong><p>{html.escape(d)}</p></div>'
                    for n, t, d in steps)
    st.markdown('<div class="rk-section" style="margin-top:2rem"><p class="rk-kicker">How it works</p>'
                f'<h2>From message to safe answer</h2></div><div class="rk-steps">{cards}</div>', unsafe_allow_html=True)
    st.markdown('<div class="rk-foot"><b>Rakshaka</b> is a demo built on synthetic messages only: no real personal '
                'data is used. Message text is stored only when a check is sent for human review. In an emergency, '
                'call your bank on the number printed on your card.</div>', unsafe_allow_html=True)


def render_check():
    for key, value in (("draft", ""), ("lang", "unknown"), ("result", None)):
        st.session_state.setdefault(key, value)
    model_version = load_health()
    if model_version is None:
        return
    render_hero(model_version)
    render_facts()
    composer, stage = st.columns([1.1, 0.9], gap="large")
    with composer:
        st.markdown('<div class="rk-section"><p class="rk-kicker">01 · Claim</p><h2>Paste the money message</h2></div>', unsafe_allow_html=True)
        sample_cols = st.columns(3)
        for column, (name, language, text) in zip(sample_cols, SAMPLES):
            column.button(name, key=f"sample-{language}-{name}", on_click=use_sample, args=(text, language), width="stretch")
        st.text_area(
            "SMS or WhatsApp message",
            key="draft",
            height=190,
            placeholder="Paste a refund, fee or bank message. Demo text only — do not forward a real scam.",
        )
        st.selectbox("Language", LANGUAGES, format_func=lambda code: LABELS[code], key="lang")
        if st.button("Assess claim", type="primary", width="stretch"):
            text = st.session_state.draft.strip()
            if not text:
                st.session_state.result = {"error": "Paste a message first."}
            else:
                try:
                    response = requests.post(
                        f"{API}/predict",
                        json={"text": text, "language": st.session_state.lang},
                        timeout=30,
                    )
                    st.session_state.result = response.json() if response.ok else {"error": response.json().get("detail", response.text)}
                except requests.RequestException as exc:
                    st.session_state.result = {"error": str(exc)}
    with stage:
        result = st.session_state.result
        if result and result.get("error"):
            st.error(result["error"] if isinstance(result["error"], str) else str(result["error"]))
        elif result:
            st.markdown(verdict_card(result), unsafe_allow_html=True)
            next_steps_panel(kind_of(result["verdict"]))
            with st.expander("Full API response"):
                st.json(result)
        else:
            signs_panel()
    render_steps()


def render_window(monitor, live=True):
    """Status banner, live-window metrics and baseline/alert-line bars for one drift window."""
    status = monitor.get("status", "WARMING_UP")
    status_kind = {"OK": "ok", "DRIFT": "bad"}.get(status, "warn")
    base, thresholds = monitor["baseline"], monitor["thresholds"]
    if status == "DRIFT":
        detail = " · ".join(monitor["signals"]) + (". New checks are routed to human review." if live else
                                                    ". Simulated checks are routed to human review.")
        title = "Drift alert" if live else "Drift alert (simulation)"
    elif status == "OK":
        title = "Ledger healthy" if live else "Simulation within baseline"
        detail = "Recent checks still sit close to the validation baseline."
    else:
        title = "Warming up"
        detail = f"{monitor['n']} of {thresholds['min_window']} checks so far. Drift is judged after {thresholds['min_window']}."
    st.markdown(
        f'<div class="rk-banner {status_kind}"><span class="rk-dot"></span><div><strong>{html.escape(title)}</strong><p>{html.escape(detail)}</p></div></div>',
        unsafe_allow_html=True,
    )
    if "mean_conf" in monitor:
        confidence, unknown, scam_rate, version = st.columns(4)
        confidence.metric("Mean confidence", f"{monitor['mean_conf']:.3f}", round(monitor["mean_conf"] - base["mean_conf"], 3))
        unknown.metric("Unknown-token rate", f"{monitor['unk_rate']:.0%}", f"{monitor['unk_rate'] - base['unk_rate']:+.0%}", delta_color="inverse")
        scam_rate.metric("Scam rate", f"{monitor['scam_rate']:.0%}")
        version.metric("Model", base["version"])
        left, right = st.columns(2)
        token_frame = pd.DataFrame({
            "Measure": ["Baseline", "Window", "Alert line"],
            "Rate": [base["unk_rate"], monitor["unk_rate"], base["unk_rate"] + thresholds["unk_rise"]],
        })
        confidence_frame = pd.DataFrame({
            "Measure": ["Baseline", "Window", "Alert line"],
            "Confidence": [base["mean_conf"], monitor["mean_conf"], base["mean_conf"] - thresholds["conf_drop"]],
        })
        with left:
            st.markdown("**Unknown-token rate**")
            st.caption("An alert fires when the window rises above the line.")
            st.altair_chart(bar_chart(token_frame, "Measure", "Rate", ["#9aa89e", "#1f5f3a", "#b3261e"], percent=True), width="stretch")
        with right:
            st.markdown("**Mean confidence**")
            st.caption("An alert fires when the window falls below the line.")
            st.altair_chart(bar_chart(confidence_frame, "Measure", "Confidence", ["#9aa89e", "#2f7d5c", "#b3261e"]), width="stretch")
    else:
        st.caption(
            f"Baseline from validation: mean confidence {base['mean_conf']:.3f}, "
            f"unknown-token rate {base['unk_rate']:.0%} (model {base['version']})."
        )


def render_history(source, base, thresholds):
    """Per-check line chart and language / result bars for live or simulated predictions (no message text)."""
    history = requests.get(f"{API}/predictions", params={"limit": 500, "source": source}, headers=admin_headers(), timeout=5)
    if not history.ok:
        st.warning("Prediction history needs a valid admin token. Sign in again.")
        return
    predictions = pd.DataFrame(history.json())
    if predictions.empty:
        st.markdown('<div class="rk-empty">No checks yet.</div>', unsafe_allow_html=True)
        return
    predictions["Result"] = predictions.apply(result_of, axis=1)
    logged, scam, legitimate, review = st.columns(4)
    logged.metric("Checks logged", len(predictions))
    scam.metric("Scam", int((predictions["Result"] == "Scam").sum()))
    legitimate.metric("Legitimate", int((predictions["Result"] == "Legitimate").sum()))
    review.metric("Sent for review", int((predictions["Result"] == "Sent for review").sum()))
    series = pd.DataFrame({
        "Check": range(1, len(predictions) + 1),
        "Unknown-token rate": predictions["unk"],
        "Confidence": predictions["confidence"],
        "Unknown-token alert line": base["unk_rate"] + thresholds["unk_rise"],
    })
    st.altair_chart(line_chart(series), width="stretch")
    st.caption("Each point is one check. Drift uses the rolling average, and message text is never shown here.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Language**")
        language_counts = counts(predictions["language"], "Language")
        st.altair_chart(bar_chart(language_counts, "Language", "Checks", ["#1f5f3a", "#2f7d5c", "#154a2c", "#a8650a", "#b3261e", "#55675b"]), width="stretch")
    with right:
        st.markdown("**Result**")
        result_counts = counts(predictions["Result"], "Result")
        st.altair_chart(bar_chart(result_counts, "Result", "Checks", ["#b3261e", "#1f5f3a", "#a8650a"]), width="stretch")


def get_monitor(mode):
    response = requests.get(f"{API}/monitor", params={"mode": mode}, headers=admin_headers(), timeout=5)
    if not response.ok:
        st.warning("Drift statistics need a valid admin token. Sign in again.")
        return None
    return response.json()


def render_monitor():
    monitor = get_monitor("live")
    if monitor is None:
        return
    st.markdown(
        f'<div class="rk-section"><p class="rk-kicker">Live traffic</p><h2>Last {monitor["thresholds"]["window"]} real checks</h2></div>',
        unsafe_allow_html=True,
    )
    render_window(monitor, live=True)
    st.markdown('<div class="rk-section"><p class="rk-kicker">History</p><h2>Checks over time</h2></div>', unsafe_allow_html=True)
    render_history("live", monitor["baseline"], monitor["thresholds"])
    st.markdown('<div class="rk-section"><p class="rk-kicker">Log</p><h2>Recent alerts</h2></div>', unsafe_allow_html=True)
    alert_response = requests.get(f"{API}/alerts", headers=admin_headers(), timeout=5)
    alerts = pd.DataFrame(alert_response.json() if alert_response.ok else [])
    if alerts.empty:
        st.markdown('<div class="rk-empty">No alerts yet. A drift episode will be recorded here once.</div>', unsafe_allow_html=True)
    else:
        alerts["When"] = pd.to_datetime(alerts["ts"], unit="s", utc=True).dt.tz_convert("Asia/Colombo").dt.strftime("%d %b %H:%M")
        st.dataframe(
            alerts.rename(columns={"kind": "Kind", "detail": "Detail"})[["When", "Kind", "Detail"]],
            hide_index=True,
            width="stretch",
        )


def render_simulation():
    st.markdown(
        '<div class="rk-section"><p class="rk-kicker">Demo</p><h2>Drift simulation</h2></div>'
        '<p class="rk-note">Sends a bounded sample of a fixed synthetic dataset through the model on the server. It uses its '
        'own drift window, so live monitoring is never affected. Uncertain simulated checks still go to the review queue, '
        'and every run is recorded in the alert log.</p>',
        unsafe_allow_html=True,
    )
    batch_size = st.slider("Messages per run", 10, 100, 50)
    test_col, shifted_col, reset_col = st.columns(3)
    for column, name, label in ((test_col, "test", "Send normal wording"), (shifted_col, "shifted", "Send new scam tactics")):
        if column.button(f"{label} ({batch_size})", key=f"sim-{name}", width="stretch"):
            with st.spinner(f"Simulating {batch_size} {name} messages…"):
                response = requests.post(f"{API}/simulate", json={"dataset": name, "n": batch_size},
                                         headers=admin_headers(), timeout=120)
            if response.ok:
                st.toast(f"{response.json()['sent']} sent, {response.json()['sent_to_review']} to review")
            else:
                st.warning(response.json().get("detail", response.text))
    if reset_col.button("Reset simulation window", width="stretch"):
        requests.post(f"{API}/simulate/reset", headers=admin_headers(), timeout=10)
    monitor = get_monitor("simulation")
    if monitor is None:
        return
    render_window(monitor, live=False)
    st.markdown('<div class="rk-section"><p class="rk-kicker">History</p><h2>Simulated checks</h2></div>', unsafe_allow_html=True)
    render_history("simulation", monitor["baseline"], monitor["thresholds"])


def admin_headers():
    """The token typed at sign-in, kept only in this browser session and sent only to admin endpoints."""
    return {"X-Admin-Token": st.session_state.get("admin_token", "")}


def render_review():
    response = requests.get(f"{API}/review_queue", headers=admin_headers(), timeout=5)
    if not response.ok:
        st.error(response.json().get("detail", "The review queue needs a valid admin token."))
        return
    queue = pd.DataFrame(response.json())
    waiting = len(queue)
    st.markdown(
        f"""
        <div class="rk-section">
          <p class="rk-kicker">Review</p>
          <h2>{waiting} message{"s" if waiting != 1 else ""} waiting</h2>
        </div>
        <p class="rk-note">Choose scam or legitimate for each message you have checked. Rows left blank are not saved,
        and every saved label is used the next time the model trains.</p>
        """,
        unsafe_allow_html=True,
    )
    if waiting:
        # no pre-filled answer: the model's own guess is never saved as a human label by default
        queue["your_label"] = None
        edited = st.data_editor(
            queue[["rowid", "text", "language", "label", "confidence", "your_label"]],
            disabled=["rowid", "text", "language", "label", "confidence"],
            column_config={
                "rowid": "ID",
                "text": st.column_config.TextColumn("Message", width="large"),
                "language": "Language",
                "label": "Model said",
                "confidence": st.column_config.NumberColumn("Confidence", format="percent"),
                "your_label": st.column_config.SelectboxColumn("Your label", options=["scam", "legitimate"]),
            },
            width="stretch",
            hide_index=True,
        )
        chosen = edited[edited["your_label"].notna()]
        st.caption(f"{len(chosen)} of {len(edited)} labelled.")
        save_col, discard_col = st.columns(2)
        if save_col.button("Save labels and retrain", type="primary", disabled=chosen.empty, width="stretch"):
            items = [{"rowid": int(row["rowid"]), "is_scam": row["your_label"] == "scam"} for _, row in chosen.iterrows()]
            with st.spinner("Saving labels and training a candidate model…"):
                saved = requests.post(f"{API}/review", json={"items": items}, headers=admin_headers(), timeout=30).json()["saved"]
                trained = requests.post(f"{API}/retrain", headers=admin_headers(), timeout=180).json()
            if not trained.get("ok"):
                st.error("Retraining failed: " + trained.get("log", "unknown error"))
            elif trained.get("deployed"):
                st.success(f"Saved {saved} labels. {trained['candidate_version']} passed the regression check and is now live.")
            else:
                st.warning(
                    f"Saved {saved} labels, but {trained['candidate_version']} was NOT deployed: "
                    + "; ".join(trained["gate"]["reasons"]) + f". {trained['model_version']} stays live. Check the labels."
                )
            gate = trained.get("gate")
            if gate:
                st.dataframe(pd.DataFrame([{"model": "live", **gate["current"]}, {"model": "candidate", **gate["candidate"]}]),
                             hide_index=True, width="stretch")
                st.caption(f"Regression check on {gate['set']} ({gate['rows']} messages, never trained on).")
        if discard_col.button("Discard unlabelled rows", width="stretch",
                              help="Removes them from the queue and erases their text"):
            for _, row in edited[edited["your_label"].isna()].iterrows():
                requests.delete(f"{API}/review_queue/{int(row['rowid'])}", headers=admin_headers(), timeout=10)
            st.rerun()
    else:
        st.markdown(
            '<div class="rk-empty">The queue is clear. Uncertain checks, and every check during a drift alert, will land here.</div>',
            unsafe_allow_html=True,
        )


def render_models():
    health = requests.get(f"{API}/health", timeout=5).json()
    monitor = get_monitor("live")
    st.markdown(
        f'<div class="rk-section"><p class="rk-kicker">Models</p><h2>Live model {html.escape(str(health["model_version"]))}</h2></div>'
        '<p class="rk-note">Retraining happens from the Human review tab. Each retrain trains a candidate, compares it with '
        'the live model on a fixed regression set (val + test, never trained on), and deploys it only if balanced accuracy, '
        'scam recall and legitimate recall do not drop. The replaced model is kept for rollback.</p>',
        unsafe_allow_html=True,
    )
    if monitor:
        base = monitor["baseline"]
        trained, rows, reviewed = st.columns(3)
        trained.metric("Trained", str(base.get("trained_at", "?"))[:16].replace("T", " "))
        rows.metric("Training messages", base.get("n_train", "?"))
        reviewed.metric("Human labels used", base.get("n_reviewed", 0))
    st.markdown('<div class="rk-section"><p class="rk-kicker">Safety</p><h2>Model rollback</h2></div>', unsafe_allow_html=True)
    st.caption("Restores the model that the last deployed retrain replaced. Press again to swap back.")
    if st.button("Roll back to previous model"):
        rolled = requests.post(f"{API}/rollback", headers=admin_headers(), timeout=60)
        if rolled.ok:
            st.success(f"Rolled back {rolled.json()['previous_version']} → {rolled.json()['model_version']}")
        else:
            st.warning(rolled.json().get("detail", rolled.text))


def render_login():
    if st.session_state.get("admin"):
        st.switch_page(ADMIN)
    st.markdown(
        """
        <section class="rk-hero">
          <div class="rk-brand"><span class="rk-hex"></span><span>Admin desk</span><span class="rk-dim">// admin token</span></div>
          <div class="rk-title">Sign in</div>
          <p class="rk-deck">Monitoring and human review stay on this side of the desk. The public checker does not use a password.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    _, column, _ = st.columns([1, 1.05, 1])
    with column:
        with st.form("admin-login"):
            secret = st.text_input("Admin token", type="password",
                                   help="The API's ADMIN_TOKEN (printed by ./run.sh, or set in .env for Docker).")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            try:   # the API checks the token; the dashboard stores no secret of its own
                check = requests.get(f"{API}/alerts", params={"limit": 1}, headers={"X-Admin-Token": secret}, timeout=5)
            except requests.RequestException:
                st.error("The API cannot be reached.")
            else:
                if check.ok:
                    st.session_state.admin = True
                    st.session_state.admin_token = secret
                    st.switch_page(ADMIN)
                elif check.status_code == 503:
                    st.error("Admin access is disabled: start the API with ADMIN_TOKEN set.")
                else:
                    st.error("That token does not open the admin desk.")


def render_admin():
    if not st.session_state.get("admin"):
        st.switch_page(LOGIN)
    model_version = load_health()
    if model_version is None:
        return
    heading, sign_out = st.columns([4, 1])
    with heading:
        st.markdown(
            f'<div class="rk-brand"><span class="rk-hex"></span><span>Admin desk</span><span class="rk-dim">// model {html.escape(str(model_version))}</span></div>',
            unsafe_allow_html=True,
        )
    with sign_out:
        if st.button("Sign out", width="stretch"):
            st.session_state.admin = False
            st.session_state.pop("admin_token", None)
            st.switch_page(CHECK)
    watch_tab, sim_tab, review_tab, models_tab = st.tabs(["Monitoring", "Drift simulation", "Human review", "Models"])
    with watch_tab:
        render_monitor()
    with sim_tab:
        render_simulation()
    with review_tab:
        render_review()
    with models_tab:
        render_models()


apply_theme()
CHECK = st.Page(render_check, title="Check a message", url_path="", default=True)
LOGIN = st.Page(render_login, title="Admin login", url_path="login")
ADMIN = st.Page(render_admin, title="Admin desk", url_path="admin")
st.navigation([CHECK, LOGIN, ADMIN], position="hidden").run()
