"""
Matcher correctness against hand-built cases and against the generated
ground truth. Run: pytest tests/
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reconai.matcher import reconcile  # noqa: E402
from reconai.loader import load_all, DATA_DIR  # noqa: E402


def _row(prefix, i, ref, date, amount):
    key = {"L": "ledger_id", "B": "bank_id", "G": "gateway_id"}[prefix]
    return {key: f"{prefix}{i:05d}", "ref": ref, "date": date, "amount": amount}


def test_exact_match_both_legs():
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 100.0)]
    bank = [_row("B", 1, "PAY1", "2026-07-01", 100.0)]
    gateway = [_row("G", 1, "PAY1", "2026-07-01", 100.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert len(matches) == 1
    assert matches[0].rule == "exact_match"
    assert matches[0].bank_ids == ["B00001"]
    assert matches[0].gateway_ids == ["G00001"]
    assert unmatched == [] and lb == [] and lg == []


def test_single_leg_is_exception_not_match():
    # gateway has it, bank does not -> must NOT be a full match
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 100.0)]
    bank = []
    gateway = [_row("G", 1, "PAY1", "2026-07-01", 100.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert matches == []
    assert len(unmatched) == 1
    assert unmatched[0]["_partial_gateway_id"] == "G00001"
    assert unmatched[0]["_partial_bank_id"] is None


def test_fee_deduction_fuzzy_matches():
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 1000.0)]
    bank = [_row("B", 1, "PAY1", "2026-07-01", 980.0)]      # 2% fee
    gateway = [_row("G", 1, "PAY1", "2026-07-01", 980.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert len(matches) == 1
    assert matches[0].rule == "fuzzy_match"


def test_timing_lag_within_window():
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 500.0)]
    bank = [_row("B", 1, "PAY1", "2026-07-04", 500.0)]      # 3-day lag
    gateway = [_row("G", 1, "PAY1", "2026-07-04", 500.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert len(matches) == 1
    assert matches[0].rule == "fuzzy_match"


def test_timing_lag_outside_window_is_exception():
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 500.0)]
    bank = [_row("B", 1, "PAY1", "2026-07-20", 500.0)]      # 19-day gap, too far
    gateway = [_row("G", 1, "PAY1", "2026-07-20", 500.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert matches == []
    assert len(unmatched) == 1


def test_duplicate_bank_rows_swept_into_match():
    ledger = [_row("L", 1, "PAY1", "2026-07-01", 100.0)]
    bank = [_row("B", 1, "PAY1", "2026-07-01", 100.0), _row("B", 2, "PAY1", "2026-07-01", 100.0)]
    gateway = [_row("G", 1, "PAY1", "2026-07-01", 100.0)]
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)
    assert len(matches) == 1
    assert set(matches[0].bank_ids) == {"B00001", "B00002"}
    assert lb == []  # both duplicates consumed


@pytest.mark.skipif(not (DATA_DIR / "ledger.csv").exists(),
                    reason="run data/generate_synthetic_data.py first")
def test_against_ground_truth_precision():
    ledger, bank, gateway = load_all()
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text())
    matches, unmatched, lb, lg = reconcile(ledger, bank, gateway)

    # every full match must correspond to a ground-truth record whose true
    # scenario is a matchable one (not 'missing')
    matchable = {"exact", "timing_lag", "fee_deduction", "duplicate", "rounding"}
    false_matches = 0
    for m in matches:
        true_scenario = ground_truth[m.ledger_id]["scenario"]
        if true_scenario not in matchable:
            false_matches += 1

    precision = 1 - (false_matches / len(matches)) if matches else 0
    assert precision >= 0.95, f"precision {precision:.3f} too low ({false_matches} false matches)"

    # all 'missing' records must end up as exceptions, never as matches
    missing_ids = {lid for lid, v in ground_truth.items() if v["scenario"] == "missing"}
    matched_ids = {m.ledger_id for m in matches}
    assert not (missing_ids & matched_ids), "a 'missing' record was wrongly matched"
