"""
Scores the pipeline against ground truth. Because the synthetic data knows the
true scenario for every ledger record, we can report real precision/recall on
matching and real accuracy on the LLM's exception classification — no eyeballing.

Usage:
  python eval.py            # matcher metrics only (no API calls)
  python eval.py --llm      # also score the LLM exception classifier (uses API)
"""

import argparse
import json
import time
from pathlib import Path

from reconai.loader import load_all, DATA_DIR
from reconai.matcher import reconcile

MATCHABLE = {"exact", "timing_lag", "fee_deduction", "duplicate", "rounding"}
# scenarios that should surface as exceptions (one leg missing / partial)
EXCEPTION_SCENARIOS = {"missing", "partial_refund"}


def evaluate_matcher(matches, unmatched, ground_truth):
    matched_ids = {m.ledger_id for m in matches}
    exception_ids = {u["ledger_id"] for u in unmatched}

    true_matchable = {lid for lid, v in ground_truth.items() if v["scenario"] in MATCHABLE}
    true_exception = {lid for lid, v in ground_truth.items() if v["scenario"] in EXCEPTION_SCENARIOS}

    tp = len(matched_ids & true_matchable)
    fp = len(matched_ids - true_matchable)
    fn = len(true_matchable - matched_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    exc_caught = len(exception_ids & true_exception)
    exc_total = len(true_exception)
    exc_recall = exc_caught / exc_total if exc_total else 0.0

    return {
        "total_ledger": len(ground_truth),
        "matched": len(matched_ids),
        "exceptions": len(exception_ids),
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": f1,
        "exception_recall": exc_recall,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_classifier(unmatched, ground_truth):
    """Scores LLM cause labels against injected scenarios. Imports lazily so the
    matcher-only path never needs the API key or the google package."""
    from reconai.classifier import classify_exception

    # map LLM cause vocabulary -> ground-truth scenario names
    cause_to_scenario = {
        "missing_record": "missing",
        "partial_refund": "partial_refund",
        "fee_deduction": "fee_deduction",
        "timing_lag": "timing_lag",
        "duplicate": "duplicate",
        "rounding_difference": "rounding",
    }

    correct, total, cannot_determine = 0, 0, 0
    latencies = []
    rows = []
    for u in unmatched:
        lid = u["ledger_id"]
        true_scenario = ground_truth[lid]["scenario"]
        t0 = time.time()
        result = classify_exception(u)
        latencies.append(time.time() - t0)
        predicted = cause_to_scenario.get(result["cause"], result["cause"])
        if result["cause"] == "cannot_determine":
            cannot_determine += 1
        else:
            total += 1
            if predicted == true_scenario:
                correct += 1
        rows.append((lid, true_scenario, result["cause"], result["confidence"]))

    accuracy = correct / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "classified": total,
        "cannot_determine": cannot_determine,
        "classification_accuracy": accuracy,
        "avg_latency_s": avg_latency,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also score the LLM classifier (uses API)")
    args = parser.parse_args()

    if not (DATA_DIR / "ledger.csv").exists():
        raise SystemExit("No data found. Run: python data/generate_synthetic_data.py")

    ledger, bank, gateway = load_all()
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text())

    t0 = time.time()
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    match_secs = time.time() - t0

    m = evaluate_matcher(matches, unmatched, ground_truth)
    print("=" * 56)
    print("  MATCHER  (deterministic, no AI)")
    print("=" * 56)
    print(f"  ledger records      : {m['total_ledger']}")
    print(f"  auto-matched        : {m['matched']}")
    print(f"  exceptions surfaced : {m['exceptions']}")
    print(f"  throughput          : {m['total_ledger']/match_secs:,.0f} records/sec")
    print(f"  match precision     : {m['match_precision']:.3f}")
    print(f"  match recall        : {m['match_recall']:.3f}")
    print(f"  match F1            : {m['match_f1']:.3f}")
    print(f"  exception recall    : {m['exception_recall']:.3f}  (true exceptions caught)")
    print(f"  tp/fp/fn            : {m['tp']}/{m['fp']}/{m['fn']}")

    if args.llm:
        print("\n" + "=" * 56)
        print("  CLASSIFIER  (LLM, exception residue only)")
        print("=" * 56)
        c = evaluate_classifier(unmatched, ground_truth)
        print(f"  exceptions sent to LLM : {len(unmatched)}")
        print(f"  confidently classified : {c['classified']}")
        print(f"  cannot_determine       : {c['cannot_determine']}")
        print(f"  classification accuracy: {c['classification_accuracy']:.3f}")
        print(f"  avg latency / call     : {c['avg_latency_s']:.2f}s")
        print("\n  ledger_id   true_scenario     predicted_cause    conf")
        print("  " + "-" * 54)
        for lid, true_s, cause, conf in c["rows"]:
            flag = " " if cause.replace("_record", "").replace("_difference", "") in true_s or true_s in cause else "x"
            print(f"  {lid}   {true_s:<16}  {cause:<18} {conf:.2f} {flag}")


if __name__ == "__main__":
    main()
