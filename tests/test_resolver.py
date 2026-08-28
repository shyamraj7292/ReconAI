"""
Resolution-action policy: proves each cause maps to the right bounded, gated
action independent of the LLM. Run: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reconai.resolver import plan_action, AUTO_APPROVE_LIMIT  # noqa: E402


def _exc(ledger_id="LDG1", amount=1000.0):
    return {"ledger_id": ledger_id, "amount": amount, "ref": "PAY1", "date": "2026-07-01"}


def _cls(cause, confidence=0.9):
    return {"cause": cause, "confidence": confidence}


def test_missing_record_flags_recovery_and_is_gated():
    a = plan_action(_exc(amount=5000.0), _cls("missing_record"))
    assert a.action_type == "RAISE_QUERY"
    assert a.kind == "recover"
    assert a.amount == 5000.0            # full exposure flagged
    assert a.requires_approval is True   # always gated


def test_duplicate_reversal_always_gated_even_if_small():
    a = plan_action(_exc(amount=10.0), _cls("duplicate"))
    assert a.action_type == "REVERSE_DUPLICATE"
    assert a.requires_approval is True   # reversals move money -> always human


def test_small_rounding_writeoff_auto_approves():
    a = plan_action(_exc(amount=0.5), _cls("rounding_difference"))
    assert a.action_type == "WRITE_OFF"
    assert a.requires_approval is False  # below the limit, safe to auto-apply


def test_fee_adjustment_gated_above_limit():
    big = AUTO_APPROVE_LIMIT + 1
    a = plan_action(_exc(amount=big), _cls("fee_deduction"))
    assert a.action_type == "POST_ADJUSTMENT"
    assert a.requires_approval is True   # amount over cap


def test_fee_adjustment_auto_below_limit():
    small = AUTO_APPROVE_LIMIT - 1
    a = plan_action(_exc(amount=small), _cls("fee_deduction"))
    assert a.requires_approval is False


def test_low_confidence_forces_human_even_when_cheap():
    a = plan_action(_exc(amount=1.0), _cls("rounding_difference", confidence=0.2))
    assert a.requires_approval is True   # confidence < 0.5 escalates
    assert "confidence" in a.gate_reason


def test_cannot_determine_routes_to_manual():
    a = plan_action(_exc(), _cls("cannot_determine", confidence=0.0))
    assert a.action_type == "MANUAL_REVIEW"
    assert a.amount == 0.0
    assert a.requires_approval is True


def test_timing_lag_no_money_movement_auto():
    a = plan_action(_exc(amount=9999.0), _cls("timing_lag"))
    assert a.moves_money is False
    assert a.amount == 0.0
    assert a.requires_approval is False  # nothing moves, safe to auto-apply
