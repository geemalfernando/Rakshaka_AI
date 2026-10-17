"""Rakshaka in the browser.

A person pastes a message they already received and presses Check this message.
Scoring starts then. Drift watch shows how those submissions, and a shifted
sample, sit against the wording the model learned.
"""
import html

import pandas as pd
import streamlit as st

from rakshaka.service import (
    DRIFT_OOV_THRESHOLD,
    REVIEW_PROBABILITY_HIGH,
    REVIEW_PROBABILITY_LOW,
)
from rakshaka.ui_data import (
    SAMPLE_MESSAGES,
    app_paths,
    compare_drift,
    load_model,
    read_jsonl,
    read_reviews,
    submit_check,
    submit_review,
    summarise_predictions,
)


# Team estimates for the impact page. Replace the value lines with the team's
# own figures. The page labels every figure as a team estimate.
IMPACT_ESTIMATES = (
    {
        "label": "First pilot",
        "value": "One staffed desk",
        "detail": (
            "Team estimate. A community or bank help desk sits with someone "
            "who brings a message they are unsure about."
        ),
    },
    {
        "label": "Who that desk can reach",
        "value": "Households in one district",
        "detail": (
            "Team estimate. The same form works for phone texts and emails "
            "in Sinhala, Tamil, English, Singlish, and Tanglish."
        ),
    },
    {
        "label": "Harm one check can prevent",
        "value": "A shared code or a rushed payment",
        "detail": (
            "Team estimate. The result tells the person to stop before they "
            "reply, share a PIN, or send money."
        ),
    },
)

PAGES = ("Check a message", "Drift watch", "Who this protects")
SOURCE_CHOICES = (
    ("Phone text", "phone"),
    ("Email", "email"),
    ("Other", "other"),
)
LANGUAGE_CHOICES = (
    ("I'm not sure", "unknown"),
    ("English", "english"),
    ("Sinhala", "sinhala"),
    ("Singlish", "singlish"),
    ("Tamil", "tamil"),
    ("Tanglish", "tanglish"),
)
SOURCE_LABELS = {code: label for label, code in SOURCE_CHOICES}
LANGUAGE_LABELS = {code: label for label, code in LANGUAGE_CHOICES}
SOURCE_LABELS["unrecorded"] = "Not recorded"

RESPONSE_COPY = {
    "human_review_and_retraining_recommended": (
        "A reviewer should check the new wording and confirm the labels. "
        "Retraining is recommended. Rakshaka will not retrain by itself."
    ),
    "continue_monitoring": (
        "Keep watching. New wording still sits near what the model learned."
    ),
}


def main():
    st.set_page_config(
        page_title="Rakshaka",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME, unsafe_allow_html=True)
    paths = app_paths()
    model = current_model(paths["model"])
    page = sidebar(model)
    if page == "Check a message":
        page_check(model, paths)
    elif page == "Drift watch":
        page_drift(model, paths)
    else:
        page_impact()


def sidebar(model):
    st.sidebar.markdown(
        '<p class="mark">Rakshaka</p>'
        '<p class="mark-sub">Check a message before you act.</p>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown(
        '<p class="privacy">Message text is saved only if you send it to a '
        "reviewer. A normal check stores the result, the language, and where "
        "you said the message came from.</p>",
        unsafe_allow_html=True,
    )
    if model is None:
        st.sidebar.caption("Checker is waiting for a trained model.")
    else:
        st.sidebar.caption("Checker ready · {}".format(model.get("model_version", "model")))
    return page


def page_check(model, paths):
    ensure_form_state()
    st.markdown(
        """
        <p class="hero-kicker">For someone who just received a message</p>
        <h1 class="hero-title">Paste the message you received.</h1>
        <p class="hero-copy">It may have arrived as a phone text, an email, or a chat.
        Copy it into the form. Rakshaka reads it only after you press
        Check this message.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sample-note">Samples only. Not real bank messages.</p>', unsafe_allow_html=True)
    columns = st.columns(3)
    for column, sample in zip(columns, SAMPLE_MESSAGES):
        if column.button(sample["label"], key="sample-{}".format(sample["language"]), width="stretch"):
            st.session_state.message_text = sample["text"]
            st.session_state.language_label = LANGUAGE_LABELS[sample["language"]]
            st.rerun()

    with st.form("check-message", clear_on_submit=False):
        st.radio(
            "Where did you get it?",
            [label for label, _code in SOURCE_CHOICES],
            horizontal=True,
            key="source_label",
            help="This labels the message for the admin view. Rakshaka does not open that app.",
        )
        st.selectbox("Language", [label for label, _code in LANGUAGE_CHOICES], key="language_label")
        st.text_area(
            "Message",
            key="message_text",
            height=180,
            placeholder="Paste the full message here",
        )
        submitted = st.form_submit_button("Check this message", type="primary", width="stretch")

    if submitted:
        run_check(model, paths)

    result = st.session_state.get("last_result")
    if result:
        show_result(result, paths["review"])


def run_check(model, paths):
    message = st.session_state.message_text
    source = dict(SOURCE_CHOICES)[st.session_state.source_label]
    language = dict(LANGUAGE_CHOICES)[st.session_state.language_label]
    if model is None:
        st.session_state.last_result = None
        st.error(
            "The checker is not ready yet. Train a model with "
            "`python scripts/train_model.py`, then submit the form again."
        )
        return
    try:
        scored = submit_check(model, message, language, source, paths["log"])
    except ValueError as error:
        st.session_state.last_result = None
        if "non-empty" in str(error):
            st.warning("Paste the message first, then press Check this message.")
        else:
            st.error(str(error))
        return
    st.session_state.last_result = scored
    st.session_state.last_message = message
    st.session_state.reviewed_stamp = None


def show_result(result, review_path):
    kind, title, body = verdict_view(result)
    st.markdown(
        '<div class="verdict verdict-{}">'
        "<h2>{}</h2><p>{}</p></div>".format(kind, html.escape(title), html.escape(body)),
        unsafe_allow_html=True,
    )
    with st.expander("Why this result"):
        left, right = st.columns(2)
        left.metric("Chance this is a scam", percent(result["scam_probability"]))
        right.metric("Unfamiliar letter patterns", percent(result["drift_oov_ratio"]))
        watch = (
            "This is above the 25% watch line, so the wording looks new."
            if result["drift_alert"]
            else "This is under the 25% watch line."
        )
        st.write(
            "Unfamiliar letter patterns are the drift signal: the share of "
            "character patterns this message uses that the model did not learn. "
            + watch
        )
        st.write(
            "You said this came from {}, in {}.".format(
                SOURCE_LABELS.get(result["source"], result["source"]),
                LANGUAGE_LABELS.get(result["language"], result["language"]),
            )
        )
        st.caption(
            "Checked at {} · model {}".format(result["timestamp"], result["model_version"])
        )
    st.markdown("##### Send this to a reviewer")
    st.caption(
        "This is the step that saves the words. It sends the message you just "
        "checked, even if you edit the box afterwards."
    )
    if st.session_state.get("reviewed_stamp") == result["timestamp"]:
        st.success("Sent. A reviewer can open it on Drift watch.")
        return
    with st.form("send-review", clear_on_submit=False):
        note = st.text_input("Note for the reviewer", placeholder="Optional. What felt wrong?")
        send = st.form_submit_button("Send this to a reviewer", width="stretch")
    if send:
        submit_review(st.session_state.last_message, result, note, review_path)
        st.session_state.reviewed_stamp = result["timestamp"]
        st.rerun()


def verdict_view(result):
    reason = review_reason(result)
    if result["prediction"] == "scam":
        body = "Do not reply. Do not share a code or PIN."
        if reason:
            body = "{} {}".format(body, reason)
        return "scam", "This looks like a scam", body
    if result["human_review_recommended"]:
        return (
            "review",
            "A person should look at this",
            reason or "Ask someone you trust before you act on it.",
        )
    return (
        "safe",
        "Looks legitimate",
        "This looks like an ordinary message. If something still feels wrong, "
        "ask someone you trust before you act.",
    )


def review_reason(result):
    parts = []
    if result["drift_alert"]:
        parts.append(
            "The wording is unfamiliar, so a person should look before you trust this result."
        )
    probability = result["scam_probability"]
    if REVIEW_PROBABILITY_LOW <= probability <= REVIEW_PROBABILITY_HIGH:
        parts.append("The score is too close to call on its own.")
    return " ".join(parts)


def page_drift(model, paths):
    st.markdown(
        """
        <p class="hero-kicker">For an admin</p>
        <h1 class="hero-title">See when the wording moves.</h1>
        <p class="hero-copy">Each point here comes from a message someone submitted
        on Check a message, or from the shifted sample used to rehearse a change
        in scam wording.</p>
        """,
        unsafe_allow_html=True,
    )
    summary = summarise_predictions(read_jsonl(paths["log"]))
    comparison = None
    if model is not None:
        try:
            comparison = compare_drift(model, paths["baseline"], paths["shifted"])
        except (OSError, ValueError) as error:
            st.error("The drift comparison could not be read. {}".format(error))
    render_banner(summary, comparison)
    st.subheader("What the model learned, and what is new")
    render_comparison(comparison)
    st.subheader("Checks people have submitted")
    render_recent(summary)
    st.subheader("Review inbox")
    render_inbox(paths["review"])


def render_banner(summary, comparison):
    shifted_alert = bool(comparison and comparison.get("available") and comparison["drift_alert"])
    live_alert = summary["drift_alert"]
    if shifted_alert or live_alert:
        reasons = []
        if shifted_alert:
            reasons.append("The shifted sample crossed the watch line.")
        if live_alert:
            reasons.append("Submitted checks crossed the watch line.")
        st.markdown(
            '<div class="banner banner-shift"><h2>Wording has shifted</h2>'
            "<p>{} {}</p></div>".format(
                html.escape(" ".join(reasons)),
                html.escape(RESPONSE_COPY["human_review_and_retraining_recommended"]),
            ),
            unsafe_allow_html=True,
        )
        return
    if summary["count"] == 0 and not (comparison and comparison.get("available")):
        st.markdown(
            '<div class="banner banner-wait"><h2>Waiting for a comparison</h2>'
            "<p>Submit a message on Check a message, or add cleaned data, and "
            "this page will show whether the wording still matches what the "
            "model learned.</p></div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="banner banner-watch"><h2>Watching</h2><p>{}</p></div>'.format(
            html.escape(RESPONSE_COPY["continue_monitoring"])
        ),
        unsafe_allow_html=True,
    )


def render_comparison(comparison):
    if comparison is None:
        st.info("Train a model with `python scripts/train_model.py` to draw this comparison.")
        return
    if not comparison["available"]:
        st.info(comparison["reason"])
        return
    learned = comparison["baseline"]["mean_unseen_char_ngram_ratio"]
    shifted = comparison["shifted"]["mean_unseen_char_ngram_ratio"]
    first, second, third = st.columns(3)
    first.metric("What the model learned", percent(learned))
    second.metric("New wording", percent(shifted))
    third.metric("Watch line", percent(comparison["threshold"]))
    frame = pd.DataFrame(
        {
            "Wording": ["What the model learned", "Watch line", "New wording"],
            "Unfamiliar patterns": [learned, comparison["threshold"], shifted],
        }
    )
    st.bar_chart(frame, x="Wording", y="Unfamiliar patterns", height=280)
    if comparison["shifted_source"] == "reviewed_file":
        caption = "New wording is data/drift/drift_samples.csv, a reviewed shift file."
    else:
        caption = (
            "New wording is the built-in sample shift: Sinhala, Tamil, Singlish, "
            "Tanglish, and English lines. Each one is a sample, not a real bank message."
        )
    st.caption(
        "{} Learned wording uses up to 200 cleaned messages. "
        "The watch line is {:.0f}%.".format(caption, comparison["threshold"] * 100)
    )


def render_recent(summary):
    if summary["count"] == 0:
        st.info(
            "No one has submitted a message yet. Each check on the first screen "
            "adds a point here. The words themselves stay off this log."
        )
        return
    scam = summary["by_prediction"].get("scam", 0)
    legitimate = summary["by_prediction"].get("legitimate", 0)
    mean = summary["mean_drift_oov_ratio"]
    first, second, third, fourth = st.columns(4)
    first.metric("Checks submitted", summary["count"])
    second.metric("Scam", scam)
    third.metric("Legitimate", legitimate)
    fourth.metric("Mean unfamiliar patterns", percent(mean))
    series = pd.DataFrame(
        {
            "Check": list(range(1, summary["count"] + 1)),
            "Unfamiliar patterns": [row["drift_oov_ratio"] for row in summary["series"]],
        }
    )
    st.line_chart(series, x="Check", y="Unfamiliar patterns", height=280)
    st.caption(
        "{} of these checks sit on or above the {:.0f}% watch line.".format(
            summary["messages_over_threshold"],
            summary["threshold"] * 100,
        )
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Language**")
        st.bar_chart(
            count_frame(summary["by_language"], LANGUAGE_LABELS, "Language"),
            x="Language",
            y="Checks",
            height=240,
        )
    with right:
        st.markdown("**Where it came from**")
        st.bar_chart(
            count_frame(summary["by_source"], SOURCE_LABELS, "Source"),
            x="Source",
            y="Checks",
            height=240,
        )


def render_inbox(review_path):
    reviews = read_reviews(review_path)
    if not reviews:
        st.info("No one has asked for a person to look yet.")
        return
    rows = []
    for item in reviews:
        rows.append(
            {
                "When": item.get("timestamp", ""),
                "Source": SOURCE_LABELS.get(item.get("source"), item.get("source", "")),
                "Language": LANGUAGE_LABELS.get(item.get("language"), item.get("language", "")),
                "Result": item.get("prediction", ""),
                "Unfamiliar patterns": percent(item.get("drift_oov_ratio") or 0),
                "Drift": "Yes" if item.get("drift_alert") else "No",
                "Message": item.get("message", ""),
                "Note": item.get("note", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_impact():
    st.markdown(
        """
        <p class="hero-kicker">Who this protects</p>
        <h1 class="hero-title">A second look before someone acts on a message.</h1>
        <p class="hero-copy">Rakshaka is for people in Sri Lanka who receive banking,
        delivery, or prize texts and are new to spotting a scam in Sinhala, Tamil,
        English, or a romanized mix.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        In June 2026 the Central Bank of Sri Lanka launched the nationwide
        “Be Scam Proof” campaign, because digital fraud keeps changing shape.
        A checker that is right on the first day can go quiet later if scammers
        switch wording. Drift watch is how a desk notices that change and sends
        the new messages to a person.
        """
    )
    st.subheader("Reach and harm")
    st.caption("These figures are team estimates. They are not official statistics.")
    columns = st.columns(3)
    for column, estimate in zip(columns, IMPACT_ESTIMATES):
        with column:
            st.markdown(
                '<div class="estimate"><p class="estimate-label">{}</p>'
                '<p class="estimate-value">{}</p><p>{}</p></div>'.format(
                    html.escape(estimate["label"]),
                    html.escape(estimate["value"]),
                    html.escape(estimate["detail"]),
                ),
                unsafe_allow_html=True,
            )
    st.subheader("What a check changes")
    st.markdown(
        """
        The person pastes the message and gets a plain result: ordinary, scam,
        or ask a person. A scam result says to stop, with no reply and no shared
        code. When the wording is new, the same screen asks for a human look, so
        the model does not fail in silence.
        """
    )
    st.subheader("A path to a pilot")
    st.markdown(
        """
        1. A community desk or a bank help desk opens **Check a message** with the person who received the text.
        2. Staff use **Drift watch** through the week. When the banner says the wording has shifted, they review the inbox and confirm labels before anyone retrains.
        3. The next step is the same form in one district, with a named reviewer, still storing message text only when someone asks for that review.
        """
    )


def count_frame(counts, labels, column):
    names = [labels.get(key, key) for key in counts]
    return pd.DataFrame({column: names, "Checks": [counts[key] for key in counts]})


def percent(value):
    return "{:.0f}%".format(float(value) * 100)


def ensure_form_state():
    if "source_label" not in st.session_state:
        st.session_state.source_label = SOURCE_CHOICES[0][0]
    if "language_label" not in st.session_state:
        st.session_state.language_label = LANGUAGE_CHOICES[0][0]
    if "message_text" not in st.session_state:
        st.session_state.message_text = ""


@st.cache_resource
def load_cached_model(path_text, stamp):
    del stamp
    return load_model(path_text)


def current_model(model_path):
    if not model_path.exists():
        return None
    try:
        return load_cached_model(str(model_path), model_path.stat().st_mtime)
    except (OSError, ValueError) as error:
        st.error("The model file could not be read. {}".format(error))
        return None


THEME = """
<style>
.stApp { background: #f4efe6; color: #1c1915; }
.block-container { padding-top: 1.6rem; max-width: 1080px; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp label { color: #1c1915 !important; }
.stApp [data-baseweb="select"] > div,
.stApp textarea,
.stApp [data-baseweb="input"] > div {
  background-color: #fffdf8 !important;
  color: #1c1915 !important;
  border-color: #d9d0c3 !important;
}
.stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-secondary"] {
  background: #fffdf8;
  color: #1c1915;
  border: 1px solid #d9d0c3;
}
h1, h2, h3, .hero-title {
  font-family: Georgia, "Iowan Old Style", Palatino, serif;
  letter-spacing: -0.03em;
  color: #1c1915;
}
.hero-kicker {
  color: #9a4e12;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.78rem;
  margin: 0 0 0.35rem 0;
}
.hero-title {
  font-size: 2.7rem;
  line-height: 1.05;
  margin: 0 0 0.7rem 0;
}
.hero-copy {
  font-size: 1.12rem;
  line-height: 1.45;
  color: #3f3933;
  max-width: 42rem;
  margin-bottom: 0.8rem;
}
.sample-note { color: #6d665e; font-size: 0.92rem; margin: 0.2rem 0 0.4rem 0; }
.verdict, .banner, .estimate {
  border-radius: 18px;
  padding: 1.15rem 1.3rem;
  margin: 0.4rem 0 0.8rem 0;
}
.verdict h2, .banner h2 { margin: 0 0 0.35rem 0; font-size: 1.7rem; }
.verdict p, .banner p, .estimate p { margin: 0; font-size: 1.05rem; line-height: 1.45; }
.verdict-safe { background: #e5f2ea; border: 1px solid #1f6b4a; }
.verdict-scam { background: #f8e6e3; border: 1px solid #9c2f2f; }
.verdict-review { background: #f8ecd9; border: 1px solid #a15c12; }
.banner-watch { background: #e7eef6; border: 1px solid #24527a; }
.banner-shift { background: #f8ecd9; border: 1px solid #a15c12; }
.banner-wait { background: #fffdf8; border: 1px solid #d9d0c3; }
.estimate { background: #fffdf8; border: 1px solid #e0d5c6; min-height: 11rem; }
.estimate-label {
  color: #9a4e12;
  font-size: 0.78rem !important;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.35rem !important;
}
.estimate-value {
  font-family: Georgia, Palatino, serif;
  font-size: 1.45rem !important;
  line-height: 1.2;
  margin-bottom: 0.45rem !important;
}
section[data-testid="stSidebar"] { background: #1c1915; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #f6f1e7 !important; }
section[data-testid="stSidebar"] .mark {
  color: #f0b27a !important;
  font-family: Georgia, Palatino, serif;
  font-size: 1.7rem;
  line-height: 1;
  margin: 0.2rem 0 0.3rem 0;
}
section[data-testid="stSidebar"] .mark-sub {
  color: #d9cbb8 !important;
  margin: 0 0 1rem 0;
}
section[data-testid="stSidebar"] .privacy {
  color: #cbbba6 !important;
  font-size: 0.92rem;
  line-height: 1.4;
  margin-top: 1.2rem;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
</style>
"""


if __name__ == "__main__":
    main()
