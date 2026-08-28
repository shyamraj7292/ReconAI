"""
Deterministic, rule-based reconciliation between ledger, bank and gateway
records. No AI here on purpose: exact/fuzzy matching is cheap, fast and fully
explainable, so it should not go anywhere near an LLM call.

Two passes:
  1. Exact match: same ref, same amount, same date.
  2. Fuzzy match: same ref, amount within fee/rounding tolerance, date within
     a lag window.

Everything that survives both passes is unmatched residue, handed off to the
LLM classifier (see classifier.py) to explain *why*.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher

AMOUNT_TOLERANCE_PCT = 0.03   # covers typical platform fees (1.5-2.5%) and rounding
DATE_WINDOW_DAYS = 4          # covers observed settlement lag
REF_SIMILARITY_MIN = 0.85     # for fuzzy ref matching if refs are ever mangled


@dataclass
class MatchResult:
    ledger_id: str
    bank_ids: list = field(default_factory=list)
    gateway_ids: list = field(default_factory=list)
    rule: str = ""          # which rule fired, for the audit trail
    confidence: float = 1.0


def _parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _ref_similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _amount_close(a: float, b: float, base: float) -> bool:
    return abs(a - b) <= max(base * AMOUNT_TOLERANCE_PCT, 1.0)


def _date_close(a: date, b: date) -> bool:
    return abs((a - b).days) <= DATE_WINDOW_DAYS


def reconcile(ledger_rows, bank_rows, gateway_rows):
    """
    ledger_rows / bank_rows / gateway_rows: list[dict] as read from the CSVs
    (all values already parsed: amount -> float, date kept as 'YYYY-MM-DD' str).

    A ledger record only counts as a full match when BOTH its bank leg and its
    gateway leg are found — a match on only one side (fee deduction, partial
    refund, a leg missing entirely) is exactly the kind of thing that should
    not be silently accepted, so it is routed to unmatched_ledger with context
    on which side matched, for the LLM classifier to explain.

    Returns (matches: list[MatchResult], unmatched_ledger: list[dict],
             leftover_bank: list[dict], leftover_gateway: list[dict])
    """
    bank_pool = {r["bank_id"]: dict(r, _date=_parse_date(r["date"])) for r in bank_rows}
    gateway_pool = {r["gateway_id"]: dict(r, _date=_parse_date(r["date"])) for r in gateway_rows}

    matches = []
    unmatched_ledger = []

    for ledger in ledger_rows:
        ref = ledger["ref"]
        amount = ledger["amount"]
        ldate = _parse_date(ledger["date"])

        bank_hit = _find_exact(bank_pool, ref, amount, ldate)
        gw_hit = _find_exact(gateway_pool, ref, amount, ldate)
        rule = "exact_match"

        if not (bank_hit and gw_hit):
            fuzzy_bank_hit = _find_fuzzy(bank_pool, ref, amount, ldate)
            fuzzy_gw_hit = _find_fuzzy(gateway_pool, ref, amount, ldate)
            if fuzzy_bank_hit or fuzzy_gw_hit:
                bank_hit = bank_hit or fuzzy_bank_hit
                gw_hit = gw_hit or fuzzy_gw_hit
                rule = "fuzzy_match"

        if bank_hit and gw_hit:
            bank_ids = [bank_hit]
            gw_ids = [gw_hit]
            # sweep for duplicates: any other bank rows with the same ref+amount+date
            bank_ids += _find_duplicates(bank_pool, ref, amount, ldate, exclude=bank_ids)
            for bid in bank_ids:
                bank_pool.pop(bid, None)
            for gid in gw_ids:
                gateway_pool.pop(gid, None)
            confidence = 1.0 if rule == "exact_match" else 0.85
            matches.append(MatchResult(ledger["ledger_id"], bank_ids, gw_ids, rule=rule, confidence=confidence))
            continue

        # only one leg found, or neither -> genuine exception, hand to classifier.
        # Note: we deliberately do NOT consume the partially-found leg here, so
        # it also shows up in leftover_bank/leftover_gateway for the residue report.
        unmatched_ledger.append(dict(
            ledger,
            _partial_bank_id=bank_hit,
            _partial_gateway_id=gw_hit,
        ))

    return matches, unmatched_ledger, list(bank_pool.values()), list(gateway_pool.values())


def _find_exact(pool, ref, amount, ldate):
    for rid, row in pool.items():
        if row["ref"] == ref and row["amount"] == amount and row["_date"] == ldate:
            return rid
    return None


def _find_duplicates(pool, ref, amount, ldate, exclude):
    found = []
    for rid, row in pool.items():
        if rid in exclude:
            continue
        if row["ref"] == ref and row["amount"] == amount and row["_date"] == ldate:
            found.append(rid)
    return found


def _find_fuzzy(pool, ref, amount, ldate):
    best_rid, best_score = None, float("-inf")
    for rid, row in pool.items():
        if row["ref"] != ref:
            continue
        if not _date_close(row["_date"], ldate):
            continue
        if not _amount_close(row["amount"], amount, amount):
            continue
        # score: closer amount and closer date wins if multiple candidates
        score = -(abs(row["amount"] - amount) + abs((row["_date"] - ldate).days))
        if score > best_score:
            best_rid, best_score = rid, score
    return best_rid
