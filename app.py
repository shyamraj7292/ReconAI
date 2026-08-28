"""
ReconAI dashboard. Run: streamlit run app.py

Three things, in order of what a reviewer cares about:
  1. The loop-closing outcome — money reconciled / gated / flagged for recovery.
  2. The exception + action table, where gated actions can be approved live
     (approving executes the action and posts to the mock ledger).
  3. A per-record audit viewer.
"""

import pandas as pd
import streamlit as st

from reconai import audit, actions
from reconai.pipeline import run

st.set_page_config(page_title="ReconAI", layout="wide")
st.title("ReconAI — reconciliation agent that closes the loop")
st.caption("Deterministic matching first. LLM only on the residue. A deterministic "
           "policy — never the LLM — decides every money action, and gates it.")

with st.sidebar:
    st.header("Run")
    mode = st.radio("Classifier", ["LLM (needs API key)", "Simulated (offline)", "Matcher only"],
                    index=1, help="Simulated maps ground truth -> cause so you can demo "
                                  "the full loop with no API key.")
    if st.button("Run reconciliation", type="primary"):
        with st.spinner("Reconciling..."):
            summary, _, _ = run(
                use_llm=(mode == "LLM (needs API key)"),
                sim=(mode == "Simulated (offline)"),
            )
        st.session_state["last_run"] = summary["run_id"]
        st.success(f"Done: {summary['run_id']}")

conn = audit.init_db()
actions.init_actions(conn)
run_id = st.session_state.get("last_run") or audit.latest_run_id(conn)

if not run_id:
    st.info("No runs yet. Click **Run reconciliation** in the sidebar.")
    st.stop()

decisions = pd.DataFrame(audit.fetch_run(conn, run_id))
resolutions = actions.fetch_resolutions(conn, run_id)
outcome = actions.batch_outcome(conn, run_id)
matched = decisions[decisions["decision"] == "matched"] if not decisions.empty else decisions

st.subheader(f"Run {run_id}")

# --- headline: the loop-closing outcome ---
o1, o2, o3, o4 = st.columns(4)
o1.metric("Auto-matched", len(matched))
o2.metric("Money reconciled", f"Rs {outcome['money_reconciled']:,.0f}", help="Booked by executed actions")
o3.metric("Pending approval", f"Rs {outcome['money_pending_approval']:,.0f}",
          help=f"{outcome['pending_approval']} gated actions awaiting a human")
o4.metric("Flagged for recovery", f"Rs {outcome['money_flagged_for_recovery']:,.0f}",
          help="Settlement queries for money recorded but not received")

# --- exceptions + gated action approval ---
st.subheader("Exceptions & resolution actions")
if not resolutions:
    st.write("No exceptions in this run.")
else:
    for r in resolutions:
        gate = "AUTO" if not r["requires_approval"] else "GATED"
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            c1.markdown(
                f"**{r['ledger_id']}** — `{r['cause']}`  \n"
                f"{r['description']}"
            )
            c2.markdown(
                f"Action: `{r['action_type']}`  \n"
                f"Amount: **Rs {r['amount']:,.2f}**  \n"
                f"Gate: `{gate}` — {r['gate_reason']}"
            )
            status = r["status"]
            if status == actions.STATUS_PENDING:
                if c3.button("Approve & execute", key=f"approve_{r['id']}"):
                    actions.approve_action(conn, r["id"])
                    st.rerun()
                c3.caption("awaiting approval")
            elif status == actions.STATUS_EXECUTED:
                c3.success("executed")
            else:
                c3.info("manual review")

# --- audit viewer ---
st.subheader("Audit viewer")
if not decisions.empty:
    selected = st.selectbox("Inspect a record", decisions["ledger_id"].tolist())
    if selected:
        record = decisions[decisions["ledger_id"] == selected].iloc[0].to_dict()
        st.json({k: v for k, v in record.items() if k != "id"})
