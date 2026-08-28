"""
End-to-end orchestration: load -> deterministic match -> LLM-classify the
residue -> write audit trail -> return a summary. This is the one loop the whole
project closes.

Usage:
  python -m reconai.pipeline           # full run (matcher + LLM on exceptions)
  python -m reconai.pipeline --no-llm  # matcher only, skip API calls
"""

import argparse
from collections import Counter

from reconai.loader import load_all
from reconai.matcher import reconcile
from reconai import audit, actions
from reconai.resolver import plan_action


def _sim_classify(exception):
    """Offline stand-in for the LLM: reads the injected scenario from ground
    truth so the full loop (planning + gated execution) can be demoed and
    verified without an API key. NOT used in a real run — see --sim."""
    import json
    from reconai.loader import DATA_DIR
    gt = json.loads((DATA_DIR / "ground_truth.json").read_text())
    scenario = gt.get(exception["ledger_id"], {}).get("scenario", "cannot_determine")
    scenario_to_cause = {
        "missing": "missing_record", "partial_refund": "partial_refund",
        "fee_deduction": "fee_deduction", "timing_lag": "timing_lag",
        "duplicate": "duplicate", "rounding": "rounding_difference",
        "exact": "cannot_determine",
    }
    return {"cause": scenario_to_cause.get(scenario, "cannot_determine"),
            "confidence": 0.95, "rationale": "simulated from ground truth (offline demo)",
            "suggested_resolution": ""}


def run(use_llm: bool = True, db_path=None, sim: bool = False):
    ledger, bank, gateway = load_all()
    matches, unmatched, leftover_bank, leftover_gateway = reconcile(ledger, bank, gateway)

    conn = audit.init_db(db_path) if db_path else audit.init_db()
    actions.init_actions(conn)
    run_id = audit.new_run_id()

    for m in matches:
        audit.record_match(conn, run_id, m)

    exceptions = []
    if sim and unmatched:
        for u in unmatched:
            exceptions.append({**u, **_sim_classify(u)})
    elif use_llm and unmatched:
        from reconai.classifier import classify_exception
        for u in unmatched:
            classification = classify_exception(u)
            exceptions.append({**u, **classification})
    else:
        # skip the LLM but still plan actions off a placeholder classification,
        # so the loop-closing path is exercisable without an API key
        for u in unmatched:
            placeholder = {"cause": "cannot_determine", "confidence": 0.0,
                           "rationale": "LLM classification skipped", "suggested_resolution": ""}
            exceptions.append({**u, **placeholder})

    # plan + (auto-)execute a bounded, gated resolution action for each exception
    resolutions = []
    for e in exceptions:
        action = plan_action(e, e)
        audit.record_exception(conn, run_id, e, e)
        resolution_id, status = actions.apply_action(conn, run_id, action)
        resolutions.append({**e, "action_type": action.action_type,
                            "action_amount": action.amount, "action_status": status,
                            "requires_approval": action.requires_approval,
                            "gate_reason": action.gate_reason,
                            "action_description": action.description,
                            "resolution_id": resolution_id})

    conn.commit()
    outcome = actions.batch_outcome(conn, run_id)

    summary = {
        "run_id": run_id,
        "total_ledger": len(ledger),
        "matched": len(matches),
        "exceptions": len(unmatched),
        "leftover_bank": len(leftover_bank),
        "leftover_gateway": len(leftover_gateway),
        "match_rate": len(matches) / len(ledger) if ledger else 0.0,
        "rule_breakdown": dict(Counter(m.rule for m in matches)),
        "cause_breakdown": dict(Counter(e["cause"] for e in exceptions)),
        "action_breakdown": dict(Counter(r["action_type"] for r in resolutions)),
        "outcome": outcome,
    }
    conn.close()
    return summary, matches, resolutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip LLM classification")
    parser.add_argument("--sim", action="store_true",
                        help="offline demo: simulate the classifier from ground truth (no API key)")
    args = parser.parse_args()

    summary, matches, resolutions = run(use_llm=not args.no_llm, sim=args.sim)
    o = summary["outcome"]

    print(f"run: {summary['run_id']}")
    print(f"ledger records : {summary['total_ledger']}")
    print(f"matched        : {summary['matched']} ({summary['match_rate']*100:.1f}%)")
    print(f"  by rule      : {summary['rule_breakdown']}")
    print(f"exceptions     : {summary['exceptions']}")
    print(f"  by cause     : {summary['cause_breakdown']}")
    print(f"  by action    : {summary['action_breakdown']}")
    print("\nloop closed:")
    print(f"  auto-executed        : {o['auto_executed']}")
    print(f"  pending approval     : {o['pending_approval']}")
    print(f"  manual review        : {o['manual_review']}")
    print(f"  money reconciled     : Rs {o['money_reconciled']:,.2f}  (booked automatically)")
    print(f"  pending approval     : Rs {o['money_pending_approval']:,.2f}  (gated, awaiting human)")
    print(f"  flagged for recovery : Rs {o['money_flagged_for_recovery']:,.2f}  (settlement queries)")
    if resolutions:
        print("\nexception detail:")
        for r in resolutions:
            gate = "AUTO" if not r["requires_approval"] else "GATED"
            print(f"  {r['ledger_id']}  {r['cause']:<18} {r['action_type']:<16} "
                  f"Rs {r['action_amount']:>10,.2f}  [{gate}] {r['action_status']}")


if __name__ == "__main__":
    main()
