"""
ReconAI dashboard. Run: streamlit run app.py

Shows the reconciliation summary, the exception table (LLM cause + confidence +
suggested resolution), and a per-record audit viewer. Reads from the audit DB
written by the pipeline; a button runs a fresh reconciliation.
"""

import pandas as pd
import streamlit as st

from reconai import audit
from reconai.pipeline import run

st.set_page_config(page_title="ReconAI", layout="wide")
st.title("ReconAI — reconciliation + exception resolution")
st.caption("Deterministic matching first. The LLM only ever sees the unmatched residue.")

with st.sidebar:
    st.header("Run")
    use_llm = st.checkbox("Classify exceptions with LLM", value=True,
                          help="Unchecked = matcher only, no API calls.")
    if st.button("Run reconciliation", type="primary"):
        with st.spinner("Reconciling..."):
            summary, matches, exceptions = run(use_llm=use_llm)
        st.session_state["last_run"] = summary["run_id"]
        st.success(f"Done: {summary['run_id']}")

conn = audit.init_db()
run_id = st.session_state.get("last_run") or audit.latest_run_id(conn)

if not run_id:
    st.info("No runs yet. Click **Run reconciliation** in the sidebar.")
    st.stop()

rows = audit.fetch_run(conn, run_id)
df = pd.DataFrame(rows)
matched = df[df["decision"] == "matched"]
exceptions = df[df["decision"] == "exception"]

st.subheader(f"Run {run_id}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ledger records", len(df))
c2.metric("Auto-matched", len(matched))
c3.metric("Exceptions", len(exceptions))
match_rate = len(matched) / len(df) * 100 if len(df) else 0
c4.metric("Match rate", f"{match_rate:.1f}%")

left, right = st.columns(2)
with left:
    st.markdown("**Matched — by rule**")
    if not matched.empty:
        st.bar_chart(matched["rule_or_cause"].value_counts())
with right:
    st.markdown("**Exceptions — by cause**")
    if not exceptions.empty:
        st.bar_chart(exceptions["rule_or_cause"].value_counts())

st.subheader("Exceptions")
if exceptions.empty:
    st.write("None.")
else:
    view = exceptions[["ledger_id", "rule_or_cause", "confidence",
                       "suggested_resolution", "rationale"]].rename(columns={
        "rule_or_cause": "cause",
    })
    st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("Audit viewer")
selected = st.selectbox("Inspect a record", df["ledger_id"].tolist())
if selected:
    record = df[df["ledger_id"] == selected].iloc[0].to_dict()
    st.json({k: v for k, v in record.items() if k != "id"})
