"""Streamlit UI: check a message + live monitoring + drift demo + review/retrain."""
import os, requests, pandas as pd, streamlit as st

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per request
API = os.getenv("API_URL", "http://127.0.0.1:8000")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGUAGES = ["unknown", "english", "sinhala", "singlish", "tamil", "tanglish"]
st.set_page_config(page_title="Rakshaka AI", layout="wide")
st.title("Rakshaka AI - scam message checker")
st.caption("All demo messages are synthetic. Never forward real scam content.")

try:
    requests.get(f"{API}/health", timeout=3).raise_for_status()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API}. Start it with: cd src && uvicorn api:app --port 8000")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Check a message", "Monitoring & drift", "Human review"])

with tab1:
    txt = st.text_area("Paste an SMS / WhatsApp message (English, Sinhala, Tamil or Singlish)")
    lang = st.selectbox("Language (optional)", LANGUAGES)
    if st.button("Check") and txt.strip():
        resp = requests.post(f"{API}/predict", json={"text": txt, "language": lang})
        if resp.ok:
            r = resp.json()
            color = "red" if r["verdict"] == "SCAM" else ("orange" if "UNCERTAIN" in r["verdict"] else "green")
            st.markdown(f"### :{color}[{r['verdict']}]")
            st.json(r)
        else:
            st.error(resp.json().get("detail", resp.text))

with tab2:
    m = requests.get(f"{API}/monitor").json()
    if m["status"] == "DRIFT":
        st.error("DRIFT ALERT: " + " | ".join(m["signals"]) + " -> predictions routed to human review")
    elif m["status"] == "OK":
        st.success("Model healthy")
    else:
        st.info(f"Warming up ({m['n']} predictions)")
    if "mean_conf" in m:
        base = m["baseline"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean confidence", m["mean_conf"], round(m["mean_conf"] - base["mean_conf"], 3))
        c2.metric("Unknown-token rate", f"{m['unk_rate']:.0%}", f"{m['unk_rate'] - base['unk_rate']:+.0%}",
                  delta_color="inverse")
        c3.metric("Scam rate", f"{m['scam_rate']:.0%}")
        c4.metric("Model", base["version"])
    st.subheader("Simulate traffic")
    a, b = st.columns(2)
    n = st.slider("messages", 10, 100, 50)
    for col, name in [(a, "test"), (b, "shifted")]:
        if col.button(f"Send {n} from {name} set"):
            df = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"))
            df = df.sample(min(n, len(df)))
            for t, lang in zip(df.text, df.language):
                requests.post(f"{API}/predict", json={"text": t, "language": lang})
            st.rerun()
    st.subheader("Recent alerts")
    st.dataframe(pd.DataFrame(requests.get(f"{API}/alerts").json()))

with tab3:
    q = pd.DataFrame(requests.get(f"{API}/review_queue").json())
    st.write(f"{len(q)} messages waiting for review")
    if len(q):
        q["is_scam"] = q["label"] == "scam"
        edited = st.data_editor(q[["rowid", "text", "language", "confidence", "is_scam"]],
                                disabled=["rowid", "text", "language", "confidence"], width="stretch")
        if st.button("Save labels + retrain"):
            items = [{"rowid": int(r.rowid), "is_scam": bool(r.is_scam)} for r in edited.itertuples()]
            saved = requests.post(f"{API}/review", json={"items": items}).json()["saved"]
            r = requests.post(f"{API}/retrain").json()
            if r["ok"]:
                st.success(f"Saved {saved} labels. Retrained -> {r['model_version']}")
            else:
                st.error("Retraining failed: " + r["log"])
