"""Streamlit UI: check a message + live monitoring + drift charts + review/retrain. All data comes from the API."""
import hmac
import html
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per request
API = os.getenv("API_URL", "http://127.0.0.1:8000")
ADMIN_PASSWORD = "scamproof"
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
        "labelColor": "#8ea396",
        "titleColor": "#8ea396",
        "gridColor": "rgba(156,255,46,0.08)",
        "domainColor": "rgba(232,243,234,0.16)",
        "labelFont": "Inter, sans-serif",
        "titleFont": "Inter, sans-serif",
    },
    "legend": {"labelColor": "#e8f3ea", "titleColor": "#8ea396", "labelFont": "Inter, sans-serif"},
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
    colors = ["#9cff2e", "#2bf5c8", "#ff5a4a"]
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


def render_check():
    for key, value in (("draft", ""), ("lang", "unknown"), ("result", None)):
        st.session_state.setdefault(key, value)
    model_version = load_health()
    if model_version is None:
        return
    render_hero(model_version)
    composer, stage = st.columns([1.12, 0.88], gap="large")
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
        st.markdown('<div class="rk-section"><p class="rk-kicker">02 · Desk</p><h2>Claim assessment</h2></div>', unsafe_allow_html=True)
        result = st.session_state.result
        if result and result.get("error"):
            st.error(result["error"] if isinstance(result["error"], str) else str(result["error"]))
        elif result:
            st.markdown(verdict_card(result), unsafe_allow_html=True)
            with st.expander("Full API response"):
                st.json(result)
        else:
            st.markdown(
                """
                <div class="rk-card rk-idle">
                  <div class="rk-docket" aria-hidden="true"><span></span><span></span><span></span><i></i></div>
                  <div>
                    <p class="rk-kicker">Claim desk</p>
                    <div class="rk-stage-title">No claim on file</div>
                    <p>Try a synthetic refund, parcel fee or bank alert. Message text is stored only when a check is sent for human review.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_monitor():
    monitor = requests.get(f"{API}/monitor", timeout=5).json()
    status = monitor.get("status", "WARMING_UP")
    status_kind = {"OK": "ok", "DRIFT": "bad"}.get(status, "warn")
    base, thresholds = monitor["baseline"], monitor["thresholds"]
    if status == "DRIFT":
        detail = " · ".join(monitor["signals"]) + ". New checks are routed to human review."
        title = "Drift alert"
    elif status == "OK":
        title, detail = "Ledger healthy", "Recent money claims still sit close to the validation baseline."
    else:
        title = "Warming up"
        detail = f"{monitor['n']} of {thresholds['min_window']} checks so far. Drift is judged after {thresholds['min_window']}."
    st.markdown(
        f'<div class="rk-banner {status_kind}"><span class="rk-dot"></span><div><strong>{html.escape(title)}</strong><p>{html.escape(detail)}</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="rk-section"><p class="rk-kicker">Live window</p><h2>Last {thresholds["window"]} predictions</h2></div>',
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
            "Measure": ["Baseline", "Live window", "Alert line"],
            "Rate": [base["unk_rate"], monitor["unk_rate"], base["unk_rate"] + thresholds["unk_rise"]],
        })
        confidence_frame = pd.DataFrame({
            "Measure": ["Baseline", "Live window", "Alert line"],
            "Confidence": [base["mean_conf"], monitor["mean_conf"], base["mean_conf"] - thresholds["conf_drop"]],
        })
        with left:
            st.markdown("**Unknown-token rate**")
            st.caption("An alert fires when the live window rises above the line.")
            st.altair_chart(bar_chart(token_frame, "Measure", "Rate", ["#6d7a70", "#9cff2e", "#ff5a4a"], percent=True), width="stretch")
        with right:
            st.markdown("**Mean confidence**")
            st.caption("An alert fires when the live window falls below the line.")
            st.altair_chart(bar_chart(confidence_frame, "Measure", "Confidence", ["#6d7a70", "#2bf5c8", "#ff5a4a"]), width="stretch")
    else:
        st.caption(
            f"Baseline from validation: mean confidence {base['mean_conf']:.3f}, "
            f"unknown-token rate {base['unk_rate']:.0%} (model {base['version']})."
        )

    st.subheader("Checks over time")
    monitor_token = st.text_input("Admin token for prediction history and alerts", type="password", key="monitor_token")
    if monitor_token:
        history_response = requests.get(f"{API}/predictions", params={"limit": 500}, headers={"X-Admin-Token": monitor_token}, timeout=10)
        preds = pd.DataFrame(history_response.json()) if history_response.ok else pd.DataFrame()
        if not history_response.ok:
            st.warning("Prediction history is unavailable: verify the admin token.")
    else:
        preds = pd.DataFrame()
        st.info("Enter an admin token to view prediction history.")
    if preds.empty:
        st.info("No checks yet. Check a message or simulate traffic below.")
    else:
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
            st.altair_chart(bar_chart(language_counts, "Language", "Checks", ["#9cff2e", "#2bf5c8", "#7ad117", "#f5a623", "#ff5a4a", "#8ea396"]), width="stretch")
        with right:
            st.markdown("**Result**")
            result_counts = counts(predictions["Result"], "Result")
            st.altair_chart(bar_chart(result_counts, "Result", "Checks", ["#ff5a4a", "#9cff2e", "#f5a623"]), width="stretch")

    st.markdown('<div class="rk-section"><p class="rk-kicker">Demo</p><h2>Simulate traffic</h2></div>', unsafe_allow_html=True)
    test_col, shifted_col = st.columns(2)
    batch_size = st.slider("Messages to send", 10, 100, 50)
    for column, name in ((test_col, "test"), (shifted_col, "shifted")):
        if column.button(f"Send {batch_size} from {name} set", key=f"sim-{name}", width="stretch"):
            frame = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"))
            batch = frame.sample(min(batch_size, len(frame)))
            with st.spinner(f"Sending {len(batch)} {name} messages…"):
                for text, language in zip(batch.text, batch.language):
                    requests.post(f"{API}/predict", json={"text": text, "language": language}, timeout=30)
            st.rerun()
    st.subheader("Recent alerts")
    if monitor_token:
        alert_response = requests.get(f"{API}/alerts", headers={"X-Admin-Token": monitor_token}, timeout=10)
        if alert_response.ok:
            st.dataframe(pd.DataFrame(alert_response.json()))
        else:
            st.warning("Alert history requires a valid admin token.")

with tab3:
    st.info("Reviewer access requires the API admin token. The token is never written to a file.")
    reviewer_token = st.text_input("Admin token", type="password", key="reviewer_token")
    if not reviewer_token:
        st.warning("Enter an admin token to view and manage the review queue.")
    else:
        headers = {"X-Admin-Token": reviewer_token}
        try:
            queue_response = requests.get(f"{API}/review_queue", headers=headers, timeout=10)
            queue_response.raise_for_status()
        except requests.RequestException as exc:
            st.error("Could not access the review queue. Check the admin token and API settings.")
        else:
            q = pd.DataFrame(queue_response.json())
            st.write(f"{len(q)} messages waiting for review")
            if len(q):
                q["is_scam"] = q["label"] == "scam"
                edited = st.data_editor(
                    q[["rowid", "text", "language", "confidence", "is_scam"]],
                    disabled=["rowid", "text", "language", "confidence"],
                    width="stretch",
                )
                if st.button("Save labels + retrain"):
                    items = [
                        {"rowid": int(r.rowid), "is_scam": bool(r.is_scam)}
                        for r in edited.itertuples()
                    ]
                    try:
                        review_response = requests.post(
                            f"{API}/review", json={"items": items}, headers=headers, timeout=30
                        )
                        review_response.raise_for_status()
                        saved = review_response.json()["saved"]
                        retrain_response = requests.post(
                            f"{API}/retrain", headers=headers, timeout=180
                        )
                        retrain_response.raise_for_status()
                        result = retrain_response.json()
                    except requests.RequestException:
                        st.error("Review/retraining request failed. Check API logs and retry safely.")
                    else:
                        if result.get("ok") and result.get("deployed"):
                            st.success(f"Saved {saved} labels. Deployed {result['model_version']}.")
                        elif result.get("ok"):
                            st.warning(f"Saved {saved} labels, but the candidate was not deployed. Review the acceptance gate.")
                            st.json(result.get("gate", {}))
                        else:
                            st.error("Retraining failed: " + result.get("log", "unknown error"))
