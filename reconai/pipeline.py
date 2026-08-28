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
from reconai import audit


def run(use_llm: bool = True, db_path=None):
    ledger, bank, gateway = load_all()
    matches, unmatched, leftover_bank, leftover_gateway = reconcile(ledger, bank, gateway)

    conn = audit.init_db(db_path) if db_path else audit.init_db()
    run_id = audit.new_run_id()

    for m in matches:
        audit.record_match(conn, run_id, m)

    exceptions = []
    if use_llm and unmatched:
        from reconai.classifier import classify_exception
        for u in unmatched:
            classification = classify_exception(u)
            audit.record_exception(conn, run_id, u, classification)
            exceptions.append({**u, **classification})
    else:
        # record exceptions without a cause so the audit trail is still complete
        for u in unmatched:
            placeholder = {"cause": "unclassified", "confidence": 0.0,
                           "rationale": "LLM classification skipped", "suggested_resolution": ""}
            audit.record_exception(conn, run_id, u, placeholder)
            exceptions.append({**u, **placeholder})

    conn.commit()

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
    }
    conn.close()
    return summary, matches, exceptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip LLM classification")
    args = parser.parse_args()

    summary, matches, exceptions = run(use_llm=not args.no_llm)

    print(f"run: {summary['run_id']}")
    print(f"ledger records : {summary['total_ledger']}")
    print(f"matched        : {summary['matched']} ({summary['match_rate']*100:.1f}%)")
    print(f"  by rule      : {summary['rule_breakdown']}")
    print(f"exceptions     : {summary['exceptions']}")
    print(f"  by cause     : {summary['cause_breakdown']}")
    if exceptions:
        print("\nexception detail:")
        for e in exceptions:
            print(f"  {e['ledger_id']}  {e['cause']:<18} conf={e['confidence']:.2f}  {e['suggested_resolution']}")


if __name__ == "__main__":
    main()
