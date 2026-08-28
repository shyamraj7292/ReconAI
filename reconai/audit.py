"""
SQLite audit trail. Every decision — matched by which rule, or flagged as an
exception with the LLM's cause and rationale — gets one row here. This is the
"show the audit trail" requirement: when a judge asks why a record was decided
a certain way, the answer is a query, not a guess.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "audit.db"


def init_db(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ledger_id TEXT NOT NULL,
            decision TEXT NOT NULL,          -- 'matched' | 'exception'
            rule_or_cause TEXT NOT NULL,     -- rule name, or LLM cause
            confidence REAL,
            bank_ids TEXT,
            gateway_ids TEXT,
            rationale TEXT,
            suggested_resolution TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def record_match(conn, run_id, match):
    conn.execute(
        """INSERT INTO decisions
           (run_id, ledger_id, decision, rule_or_cause, confidence,
            bank_ids, gateway_ids, rationale, suggested_resolution, created_at)
           VALUES (?, ?, 'matched', ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, match.ledger_id, match.rule, match.confidence,
            ",".join(match.bank_ids), ",".join(match.gateway_ids),
            "", "", _now(),
        ),
    )


def record_exception(conn, run_id, exception, classification):
    conn.execute(
        """INSERT INTO decisions
           (run_id, ledger_id, decision, rule_or_cause, confidence,
            bank_ids, gateway_ids, rationale, suggested_resolution, created_at)
           VALUES (?, ?, 'exception', ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, exception["ledger_id"], classification["cause"],
            classification["confidence"],
            exception.get("_partial_bank_id") or "",
            exception.get("_partial_gateway_id") or "",
            classification["rationale"], classification["suggested_resolution"],
            _now(),
        ),
    )


def fetch_run(conn, run_id):
    cur = conn.execute(
        "SELECT * FROM decisions WHERE run_id = ? ORDER BY id", (run_id,)
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def latest_run_id(conn):
    cur = conn.execute("SELECT run_id FROM decisions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
