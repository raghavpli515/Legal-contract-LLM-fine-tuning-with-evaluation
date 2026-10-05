"""Streamlit front end for the contract-clause demo (talks to app/api.py).

    streamlit run app/ui.py

Three tabs: grounded clause Q&A, clause-type classification (the two tasks the adapter was
trained on) and the before/after evaluation, read from results/ so the page cannot drift
from the README. Env var LEGAL_FT_API_URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("LEGAL_FT_API_URL", "http://127.0.0.1:8000")
RESULTS = Path(__file__).resolve().parents[1] / "results"
GITHUB_URL = "https://github.com/raghavpli515/Legal-contract-LLM-fine-tuning-with-evaluation"
HUB_URL = "https://huggingface.co/PimoLee5/qwen2.5-7b-cuad-qlora"
DISCLAIMER = ("**Not legal advice.** This is a research model: it can miss clauses, mislabel "
              "them or misquote. Always read the contract and consult a qualified lawyer.")
SAMPLE_EXCERPT = (
    "7. RESTRICTIVE COVENANTS\n"
    "7.1 During the Term and for a period of eighteen (18) months thereafter, the Distributor "
    "shall not, directly or indirectly, manufacture, distribute or sell within the Territory any "
    "product that competes with the Products.\n"
    "7.2 Nothing in Section 7.1 shall prevent the Distributor from holding up to five percent (5%) "
    "of the publicly traded shares of any company.\n"
    "8. GOVERNING LAW\n"
    "8.1 This Agreement shall be governed by the laws of the State of Delaware."
)
SAMPLE_CLAUSE = ("Either party may terminate this Agreement at any time, without cause, upon "
                 "thirty (30) days' prior written notice to the other party.")
LABEL_COLOURS = {"high": "green", "medium": "orange", "low": "red"}


def load_metrics() -> dict | None:
    try:
        return {run: json.loads((RESULTS / "metrics" / f"{run}.json").read_text(encoding="utf-8"))
                for run in ("base", "finetuned")}
    except (OSError, ValueError):
        return None


def post(path: str, payload: dict) -> dict | None:
    """POST to the API; show a readable error and return None on failure."""
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    if r.status_code == 422:
        detail = r.json().get("detail")
        st.error(detail if isinstance(detail, str) else "Invalid input.")
        return None
    if not r.ok:
        st.error(f"The API returned an error ({r.status_code}).")
        return None
    return r.json()


def format_confidence(p: float) -> str:
    """A probabilistic model is never certain: show ">99%" rather than a rounded "100%"."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def show_confidence(answer: dict, ece: float | None) -> None:
    colour = LABEL_COLOURS[answer["confidence_label"]]
    st.markdown(f"**Confidence:** :{colour}[{answer['confidence_label']} · "
                f"{format_confidence(answer['confidence'])}]")
    st.progress(answer["confidence"])
    measured = f" (calibration error {ece:.3f} on held-out contracts)" if ece is not None else ""
    st.caption(f"Confidence is the model's probability for its own answer{measured}. "
               "It is not a legal opinion.")


st.set_page_config(page_title="Contract clause finder", page_icon="⚖️", layout="centered")
st.title("Contract clause finder")
st.caption("Qwen2.5-7B-Instruct + a QLoRA adapter fine-tuned on CUAD · "
           f"[code]({GITHUB_URL}) · [adapter]({HUB_URL})")
st.warning(DISCLAIMER, icon="⚠️")

metrics = load_metrics()


@st.cache_data(ttl=300, show_spinner=False)
def clause_types() -> list[str]:
    r = requests.get(f"{API_URL}/clause-types", timeout=10)
    r.raise_for_status()
    return r.json()["clause_types"]


try:
    types = clause_types()
    api_error = None
except requests.RequestException:
    types = []
    api_error = (f"Cannot reach the API at {API_URL}. Start it with "
                 "`uvicorn app.api:app --port 8000` and wait about a minute for the model to load.")

tab_find, tab_classify, tab_results = st.tabs(
    ["Find a clause", "Classify a clause", "How well does it work?"])

# ---- Tab 1: grounded Q&A -----------------------------------------------------------
with tab_find:
    st.markdown("Paste part of a contract and pick a clause type. The model says whether that "
                "clause is in the text and quotes it; every quote is checked against your text.")
    if api_error:
        st.error(api_error)
    else:
        if "excerpt" not in st.session_state:
            st.session_state.excerpt = ""
        if st.button("Load a sample excerpt"):
            st.session_state.excerpt = SAMPLE_EXCERPT
        excerpt = st.text_area("Contract excerpt (up to about 3–4 pages)", key="excerpt", height=240)
        default = types.index("Non-Compete") if "Non-Compete" in types else 0
        clause_type = st.selectbox("Clause type to look for", types, index=default)

        if st.button("Find clause", type="primary", disabled=not excerpt.strip()):
            with st.spinner("Reading the excerpt…"):
                st.session_state.find_result = post(
                    "/ask", {"excerpt": excerpt, "clause_type": clause_type})

        # kept in session state so the answer survives reruns (e.g. using the other tab)
        a = st.session_state.get("find_result")
        if a:
            if a["present"] is None:
                st.subheader("Could not read the model's answer")
            elif a["present"]:
                st.subheader(f"✅ {a['clause_type']} clause found")
            else:
                st.subheader(f"➖ No {a['clause_type']} clause in this excerpt")
            show_confidence(a, metrics["finetuned"]["qa"]["ece"] if metrics else None)

            if a["quotes"]:
                st.markdown("**Evidence quoted from your excerpt**")
                for q in a["quotes"]:
                    if q["verified"]:
                        st.success(f"“{q['text']}”", icon="✅")
                    else:
                        st.error(f"“{q['text']}”\n\nThis quote was **not found** in the "
                                 "excerpt; do not rely on it.", icon="⚠️")
            elif a["present"]:
                st.info("The model said the clause is present but gave no quote; treat this "
                        "answer with caution.")
            if not a["valid_output"]:
                st.caption("The model's output did not follow the expected format exactly.")
            st.caption(f"{a['model']} · {a['latency_s']:.1f}s")

        st.caption("Limits: one excerpt at a time (about 1,300 tokens), English commercial "
                   "contracts, 35 clause types. Whole-contract review is not supported.")

# ---- Tab 2: classification ---------------------------------------------------------
with tab_classify:
    st.markdown("Paste a single clause. The model names its type, out of the 35 CUAD clause types.")
    if api_error:
        st.error(api_error)
    else:
        if "clause" not in st.session_state:
            st.session_state.clause = ""
        if st.button("Load a sample clause"):
            st.session_state.clause = SAMPLE_CLAUSE
        clause = st.text_area("Contract clause", key="clause", height=140)

        if st.button("Classify clause", type="primary", disabled=not clause.strip()):
            with st.spinner("Classifying…"):
                st.session_state.classify_result = post("/classify", {"clause": clause})

        c = st.session_state.get("classify_result")
        if c:
            if c["label"]:
                st.subheader(f"🏷️ {c['label']}")
            else:
                st.subheader("The model did not name a known clause type")
            show_confidence(c, metrics["finetuned"]["classification"]["ece"] if metrics else None)
            if not c["valid_output"]:
                st.caption("The model's output did not follow the expected format exactly.")
            st.caption(f"{c['model']} · {c['latency_s']:.1f}s")

        with st.expander(f"The {len(types)} clause types"):
            st.markdown(" · ".join(types))

# ---- Tab 3: evaluation -------------------------------------------------------------
with tab_results:
    st.markdown("The same base model **before and after** fine-tuning, on 400 held-out items from "
                "50 contracts never seen in training. Same quantization, prompts and decoding.")
    if not metrics:
        st.info("Results files not found. Run `python -m legal_ft.eval.report base finetuned`.")
    else:
        b, f = metrics["base"], metrics["finetuned"]
        cols = st.columns(4)

        def pct(x: float) -> str:
            return f"{100 * x:.1f}%"

        def delta_pts(new: float, old: float) -> str:
            return f"{100 * (new - old):+.1f} pts vs base"

        cols[0].metric("Classification accuracy", pct(f["classification"]["accuracy"]),
                       delta_pts(f["classification"]["accuracy"], b["classification"]["accuracy"]))
        cols[1].metric("Missed clauses", pct(f["qa"]["missed_rate"]),
                       delta_pts(f["qa"]["missed_rate"], b["qa"]["missed_rate"]),
                       delta_color="inverse")
        cols[2].metric("Calibration error (ECE)", f"{f['qa']['ece']:.3f}",
                       f"{f['qa']['ece'] - b['qa']['ece']:+.3f} vs base", delta_color="inverse")
        cols[3].metric("Hallucinations (audited)", pct(f["qa"]["hallucination_rate_audited"]),
                       delta_pts(f["qa"]["hallucination_rate_audited"],
                                 b["qa"]["hallucination_rate_audited"]),
                       delta_color="inverse")

        st.markdown(
            "**What improved:** the base model misses most clauses, usually with near-certain "
            "confidence. Fine-tuning fixed that and made its confidence trustworthy.\n\n"
            "**What got worse:** hallucinations rose slightly. The fine-tuned model sometimes "
            "labels a real passage as the wrong clause type. It quotes the contract verbatim "
            "when it does, so the quote can be checked. Every flagged case in both runs was "
            "audited by hand.")
        table = RESULTS / "comparison.md"
        if table.exists():
            with st.expander("Full results table (95% confidence intervals in brackets)"):
                st.markdown(table.read_text(encoding="utf-8"))
        st.markdown(f"Method, audit and limitations: [GitHub]({GITHUB_URL}) · "
                    f"adapter and model card: [Hugging Face]({HUB_URL})")
