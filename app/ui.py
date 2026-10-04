"""Streamlit front end for the contract-clause demo (talks to app/api.py).

    streamlit run app/ui.py

Env var LEGAL_FT_API_URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("LEGAL_FT_API_URL", "http://127.0.0.1:8000")
DISCLAIMER = ("**Not legal advice.** This is a research model: it can miss clauses, mislabel "
              "them or misquote. Always read the contract and consult a qualified lawyer.")
SAMPLE = (
    "7. RESTRICTIVE COVENANTS\n"
    "7.1 During the Term and for a period of eighteen (18) months thereafter, the Distributor "
    "shall not, directly or indirectly, manufacture, distribute or sell within the Territory any "
    "product that competes with the Products.\n"
    "7.2 Nothing in Section 7.1 shall prevent the Distributor from holding up to five percent (5%) "
    "of the publicly traded shares of any company.\n"
    "8. GOVERNING LAW\n"
    "8.1 This Agreement shall be governed by the laws of the State of Delaware."
)
LABEL_COLOURS = {"high": "green", "medium": "orange", "low": "red"}

st.set_page_config(page_title="Contract clause finder", page_icon="⚖️", layout="centered")
st.title("Contract clause finder")
st.caption("Qwen2.5-7B + QLoRA adapter fine-tuned on CUAD · answers only from the text you paste")
st.warning(DISCLAIMER, icon="⚠️")


@st.cache_data(ttl=300, show_spinner=False)
def clause_types() -> list[str]:
    r = requests.get(f"{API_URL}/clause-types", timeout=10)
    r.raise_for_status()
    return r.json()["clause_types"]


try:
    types = clause_types()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn app.api:app --port 8000` "
             "and wait for the model to load (about a minute).")
    st.stop()

if "excerpt" not in st.session_state:
    st.session_state.excerpt = ""
if st.button("Load a sample excerpt"):
    st.session_state.excerpt = SAMPLE

excerpt = st.text_area("Contract excerpt (up to ~3–4 pages)", key="excerpt", height=260)
default = types.index("Non-Compete") if "Non-Compete" in types else 0
clause_type = st.selectbox("Clause type to look for", types, index=default)

if st.button("Find clause", type="primary", disabled=not excerpt.strip()):
    with st.spinner("Reading the excerpt…"):
        try:
            r = requests.post(f"{API_URL}/ask", json={"excerpt": excerpt, "clause_type": clause_type},
                              timeout=300)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()
    if r.status_code == 422:
        detail = r.json().get("detail")
        st.error(detail if isinstance(detail, str) else "Invalid input.")
        st.stop()
    r.raise_for_status()
    a = r.json()

    if a["present"] is None:
        st.subheader("Could not read the model's answer")
    elif a["present"]:
        st.subheader(f"✅ {clause_type} clause found")
    else:
        st.subheader(f"➖ No {clause_type} clause in this excerpt")

    colour = LABEL_COLOURS[a["confidence_label"]]
    st.markdown(f"**Confidence:** :{colour}[{a['confidence_label']} · {a['confidence']:.0%}]")
    st.progress(a["confidence"])
    st.caption("Confidence is the model's probability for its own answer. On held-out contracts "
               "these probabilities were well calibrated (calibration error 0.026), so ~90% "
               "confidence means right about 9 times in 10. It is not a legal opinion.")

    if a["quotes"]:
        st.markdown("**Evidence quoted from your excerpt**")
        for q in a["quotes"]:
            if q["verified"]:
                st.success(f"“{q['text']}”", icon="✅")
            else:
                st.error(f"“{q['text']}”\n\nThis quote was **not found** in the excerpt; "
                         "do not rely on it.", icon="⚠️")
    elif a["present"]:
        st.info("The model said the clause is present but gave no quote; treat this answer "
                "with caution.")

    if not a["valid_output"]:
        st.caption("The model's output did not follow the expected format exactly.")
    st.caption(f"{a['model']} · {a['latency_s']:.1f}s")
