"""
Explains a single unmatched (exception) record: why did deterministic matching
fail, and what should a human do about it? This is the ONLY place an LLM is used
in the pipeline — the matcher handled everything explainable with cheap rules,
so the model only sees the genuinely ambiguous residue.

The prompt forces a fixed cause vocabulary and requires 'cannot_determine' when
evidence is weak, which is what keeps the exception list honest instead of the
model inventing a confident label for everything.
"""

from reconai.llm_client import call_json

VALID_CAUSES = {
    "missing_record",
    "partial_refund",
    "fee_deduction",
    "timing_lag",
    "duplicate",
    "rounding_difference",
    "cannot_determine",
}

PROMPT_TEMPLATE = """You are a finance reconciliation analyst. A payment recorded in the internal \
ledger could not be automatically matched to both the bank statement and the payment \
gateway settlement report. Deterministic rules (exact match, then fuzzy match within a \
3% amount tolerance and a 4-day settlement window) already ran and failed.

Classify the single most likely cause and propose a resolution.

Ledger record (source of truth for what was expected):
  ref: {ref}
  date: {date}
  amount: {amount}
  merchant: {merchant}

What deterministic matching found:
  bank leg matched: {bank_status}
  gateway leg matched: {gateway_status}

Cause vocabulary — you MUST pick exactly one of these for "cause":
  - missing_record: the payment is absent from a source entirely (money may not have settled)
  - partial_refund: a smaller amount appears, consistent with a partial refund against the payment
  - fee_deduction: a slightly smaller amount consistent with a platform fee
  - timing_lag: the record likely exists but settled outside the matching window
  - duplicate: the same payment appears more than once in a source
  - rounding_difference: a sub-rupee discrepancy
  - cannot_determine: evidence is insufficient to choose confidently

Rules:
  - If evidence does not clearly point to one cause, you MUST answer "cannot_determine". Do not guess.
  - confidence is your genuine certainty from 0.0 to 1.0.

Respond with ONLY a JSON object of this exact shape:
{{"cause": "<one of the vocabulary values>", "confidence": <float 0-1>, \
"suggested_resolution": "<one concrete next step a human should take>", \
"rationale": "<one sentence explaining the classification>"}}"""


def build_prompt(exception: dict) -> str:
    bank_status = (
        f"yes (partial, id {exception['_partial_bank_id']})"
        if exception.get("_partial_bank_id")
        else "no"
    )
    gateway_status = (
        f"yes (partial, id {exception['_partial_gateway_id']})"
        if exception.get("_partial_gateway_id")
        else "no"
    )
    return PROMPT_TEMPLATE.format(
        ref=exception["ref"],
        date=exception["date"],
        amount=exception["amount"],
        merchant=exception.get("merchant", "unknown"),
        bank_status=bank_status,
        gateway_status=gateway_status,
    )


def _validate(result: dict) -> dict:
    cause = result.get("cause", "cannot_determine")
    if cause not in VALID_CAUSES:
        # model went off-vocabulary -> treat as undetermined rather than trust it
        cause = "cannot_determine"
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "cause": cause,
        "confidence": confidence,
        "suggested_resolution": str(result.get("suggested_resolution", "")).strip(),
        "rationale": str(result.get("rationale", "")).strip(),
    }


def classify_exception(exception: dict) -> dict:
    """Returns {cause, confidence, suggested_resolution, rationale}. On LLM
    failure, degrades to a cannot_determine result rather than crashing the
    whole batch — one bad record must not sink the run."""
    prompt = build_prompt(exception)
    try:
        raw = call_json(prompt)
    except Exception as e:  # noqa: BLE001
        return {
            "cause": "cannot_determine",
            "confidence": 0.0,
            "suggested_resolution": "Manual review required (automated classification failed).",
            "rationale": f"LLM error: {e}",
        }
    return _validate(raw)
