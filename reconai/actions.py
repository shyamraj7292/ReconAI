"""
Executes resolution actions against a MOCK finance system and records every
action to SQLite. This is what makes ReconAI an agent that closes the loop
rather than a report that stops at "here's a problem":

  - auto-approved actions are executed immediately (bounded, low-risk)
  - gated actions are recorded as PENDING_APPROVAL and NOT executed until a
    human approves them (see approve_action)

Executing an action posts an adjusting entry to a mock `mock_ledger` table and
records the action in `resolutions`, so the batch-level financial outcome
(money reconciled, money flagged for recovery) is a query, not a claim.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "audit.db"

STATUS_EXECUTED = "EXECUTED"
STATUS_PENDING = "PENDING_APPROVAL"
STATUS_MANUAL = "MANUAL_REVIEW"


def init_actions(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ledger_id TEXT NOT NULL,
            cause TEXT NOT NULL,
            action_type TEXT NOT NULL,
            amount REAL NOT NULL,
            kind TEXT NOT NULL,              -- reconcile | recover | review
            status TEXT NOT NULL,            -- EXECUTED | PENDING_APPROVAL | MANUAL_REVIEW
            requires_approval INTEGER NOT NULL,
            gate_reason TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mock_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ledger_id TEXT NOT NULL,
            entry_type TEXT NOT NULL,        -- the action_type that posted it
            amount REAL NOT NULL,
            memo TEXT,
            posted_at TEXT NOT NULL
        )
    """)
    conn.commit()


def apply_action(conn, run_id, action):
    """Records the action and, if it does not require approval, executes it.
    Returns the resolution status."""
    if action.action_type == "MANUAL_REVIEW":
        status = STATUS_MANUAL
    elif action.requires_approval:
        status = STATUS_PENDING
    else:
        status = STATUS_EXECUTED

    cur = conn.execute(
        """INSERT INTO resolutions
           (run_id, ledger_id, cause, action_type, amount, kind, status,
            requires_approval, gate_reason, description, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, action.ledger_id, action.cause, action.action_type, action.amount,
         action.kind, status, int(action.requires_approval), action.gate_reason,
         action.description, _now()),
    )
    resolution_id = cur.lastrowid

    if status == STATUS_EXECUTED and action.moves_money:
        _post_to_ledger(conn, run_id, action)

    conn.commit()
    return resolution_id, status


def approve_action(conn, resolution_id):
    """Human approves a PENDING action: execute it now (post to mock ledger) and
    flip status to EXECUTED. This is the gate being cleared."""
    cur = conn.execute("SELECT * FROM resolutions WHERE id = ?", (resolution_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if not row:
        raise ValueError(f"no resolution {resolution_id}")
    res = dict(zip(cols, row))
    if res["status"] != STATUS_PENDING:
        return res["status"]

    if res["kind"] != "review":
        # recover-type actions (RAISE_QUERY) are logged as executed but post no
        # ledger entry — the money isn't ours to book yet, it's being chased.
        if res["action_type"] not in ("RAISE_QUERY",):
            _post_to_ledger_raw(conn, res["run_id"], res["ledger_id"],
                                res["action_type"], res["amount"], res["description"])
    conn.execute("UPDATE resolutions SET status = ? WHERE id = ?", (STATUS_EXECUTED, resolution_id))
    conn.commit()
    return STATUS_EXECUTED


def batch_outcome(conn, run_id):
    """The headline numbers for the run: how much the agent actually closed."""
    cur = conn.execute(
        "SELECT status, kind, amount, action_type FROM resolutions WHERE run_id = ?",
        (run_id,),
    )
    reconciled = 0.0      # money booked by executed reconcile actions
    pending_amount = 0.0  # money waiting on human approval
    recover_flagged = 0.0 # money flagged to chase (missing settlements)
    counts = {STATUS_EXECUTED: 0, STATUS_PENDING: 0, STATUS_MANUAL: 0}
    for status, kind, amount, action_type in cur.fetchall():
        counts[status] = counts.get(status, 0) + 1
        if status == STATUS_EXECUTED and kind == "reconcile":
            reconciled += amount
        elif status == STATUS_PENDING:
            pending_amount += amount
            if kind == "recover":
                recover_flagged += amount
        elif kind == "recover":
            recover_flagged += amount
    return {
        "money_reconciled": round(reconciled, 2),
        "money_pending_approval": round(pending_amount, 2),
        "money_flagged_for_recovery": round(recover_flagged, 2),
        "auto_executed": counts.get(STATUS_EXECUTED, 0),
        "pending_approval": counts.get(STATUS_PENDING, 0),
        "manual_review": counts.get(STATUS_MANUAL, 0),
    }


def fetch_resolutions(conn, run_id):
    cur = conn.execute("SELECT * FROM resolutions WHERE run_id = ? ORDER BY id", (run_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _post_to_ledger(conn, run_id, action):
    _post_to_ledger_raw(conn, run_id, action.ledger_id, action.action_type,
                        action.amount, action.description)


def _post_to_ledger_raw(conn, run_id, ledger_id, entry_type, amount, memo):
    conn.execute(
        """INSERT INTO mock_ledger (run_id, ledger_id, entry_type, amount, memo, posted_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, ledger_id, entry_type, amount, memo, _now()),
    )


def _now():
    return datetime.now(timezone.utc).isoformat()
