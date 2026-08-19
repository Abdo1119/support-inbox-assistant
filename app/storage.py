"""SQLite persistence for triaged tickets and their review state.

Standard library only. The schema is one table, and an ORM would be a
dependency earning nothing against `sqlite3` plus seven functions.

Two decisions worth knowing before reading the code:

**`suggested_reply` and `edited_reply` are separate columns and stay that way.**
The model's draft is never overwritten by the human's edit. The pair -- what the
model proposed next to what a person actually sent -- is a feedback dataset that
normal use produces for free, and it is only available if both survive.

**A row a human has touched is never overwritten by a re-run.** See
`save_record`: losing a reviewer's edit to a batch job would be worse than
serving a stale prediction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import Field

from app.schemas import TriageRecord

# Generated output, not source. Matches the `*.db` line already in .gitignore:
# the repository should be reproducible by running scripts/run_triage.py, not by
# shipping a binary.
DEFAULT_DB_PATH = Path("triage.db")

TABLE = "triage"


class Status(str, Enum):
    """Where a ticket is in the human review queue."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RecordNotFound(LookupError):
    """No row with that ticket_id."""


class InvalidTransition(ValueError):
    """A status change that the review workflow does not permit."""


class StoredRecord(TriageRecord):
    """A TriageRecord plus the review state that only the database owns.

    Subclassing rather than duplicating: everything the triage layer decided is
    inherited unchanged, and validation runs again on the way out of SQLite,
    which is where a hand-edited database or a schema drift would otherwise slip
    through unnoticed.
    """

    status: Status = Status.PENDING
    edited_reply: str | None = None
    created_at: str
    updated_at: str


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    ticket_id          TEXT    PRIMARY KEY,

    -- The model's judgement, as validated by app/schemas.py.
    category           TEXT    NOT NULL,
    priority           TEXT    NOT NULL,
    summary            TEXT    NOT NULL,
    suggested_reply    TEXT    NOT NULL,
    model_confidence   REAL    NOT NULL,
    final_confidence   REAL    NOT NULL,
    -- JSON array. SQLite has no list type, and a second table for at most a
    -- couple of strings per row would buy a join and nothing else.
    confidence_signals TEXT    NOT NULL DEFAULT '[]',
    -- SQLite has no boolean either; 0/1 with the conversion done on read.
    escalate           INTEGER NOT NULL,
    escalation_reason  TEXT,
    fallback_used      INTEGER NOT NULL,
    retries            INTEGER NOT NULL,

    -- Review state, owned entirely by the human side.
    status             TEXT    NOT NULL DEFAULT 'pending'
                               CHECK (status IN ('pending', 'approved', 'rejected')),
    -- NULL until someone edits. Never merged into suggested_reply.
    edited_reply       TEXT,

    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL
);
"""
# No indexes: the table holds 30 rows. An index on `status` would be cargo cult
# at this size, and SQLite scans it faster than it would traverse a B-tree.
#
# CHECK constraints cover `status` because nothing else validates it. `category`
# and `priority` deliberately have none -- Pydantic owns those enums, and a
# second copy of the allowed values in SQL is a source of drift, not of safety.


def _now() -> str:
    """UTC, ISO 8601. Sorts lexicographically, which is why it is stored as text."""
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection and ensure the schema exists.

    The path is a parameter rather than a module global so tests can pass
    ":memory:" and the eval can point at a copy without touching the real file.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(_SCHEMA)
    return conn


def _row_to_record(row: sqlite3.Row) -> StoredRecord:
    """Rebuild a validated record from a row, undoing the SQLite type squeeze."""
    data = dict(row)
    data["confidence_signals"] = json.loads(data["confidence_signals"])
    data["escalate"] = bool(data["escalate"])
    data["fallback_used"] = bool(data["fallback_used"])
    return StoredRecord(**data)


def save_record(conn: sqlite3.Connection, record: TriageRecord) -> str:
    """Write a triage result. Returns "inserted", "updated", or "preserved".

    A row that a human has touched -- status moved off pending, or a reply
    edited -- is left exactly as it is and reported as "preserved". Re-running
    the batch must never cost a reviewer their work, and a prediction that is
    one prompt revision out of date is a far smaller loss than an edit that is
    gone.

    An untouched pending row is refreshed with the new prediction and keeps its
    original created_at, so the row's age still means what it says.
    """
    existing = conn.execute(
        f"SELECT status, edited_reply, created_at FROM {TABLE} WHERE ticket_id = ?",
        (record.ticket_id,),
    ).fetchone()

    if existing is not None:
        touched = existing["status"] != Status.PENDING.value or existing["edited_reply"] is not None
        if touched:
            return "preserved"

    now = _now()
    created = existing["created_at"] if existing is not None else now
    with conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE} (
                ticket_id, category, priority, summary, suggested_reply,
                model_confidence, final_confidence, confidence_signals,
                escalate, escalation_reason, fallback_used, retries,
                status, edited_reply, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                category           = excluded.category,
                priority           = excluded.priority,
                summary            = excluded.summary,
                suggested_reply    = excluded.suggested_reply,
                model_confidence   = excluded.model_confidence,
                final_confidence   = excluded.final_confidence,
                confidence_signals = excluded.confidence_signals,
                escalate           = excluded.escalate,
                escalation_reason  = excluded.escalation_reason,
                fallback_used      = excluded.fallback_used,
                retries            = excluded.retries,
                updated_at         = excluded.updated_at
            """,
            (
                record.ticket_id,
                record.category.value,
                record.priority.value,
                record.summary,
                record.suggested_reply,
                record.model_confidence,
                record.final_confidence,
                json.dumps(record.confidence_signals),
                int(record.escalate),
                record.escalation_reason,
                int(record.fallback_used),
                record.retries,
                created,
                now,
            ),
        )
    return "updated" if existing is not None else "inserted"


def get_record(conn: sqlite3.Connection, ticket_id: str) -> StoredRecord | None:
    """Return one record, or None if there is no such ticket."""
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def list_records(
    conn: sqlite3.Connection,
    *,
    status: Status | str | None = None,
    escalate: bool | None = None,
) -> list[StoredRecord]:
    """List records, optionally filtered by review status and escalation.

    Both filters are optional and independent. Ordering puts escalated tickets
    first and then lowest confidence first, which is the order a reviewer
    working a queue actually wants.
    """
    clauses: list[str] = []
    params: list[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(Status(status).value)
    if escalate is not None:
        clauses.append("escalate = ?")
        params.append(int(escalate))

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM {TABLE}{where} ORDER BY escalate DESC, final_confidence ASC, ticket_id ASC",
        params,
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def set_edited_reply(conn: sqlite3.Connection, ticket_id: str, reply: str) -> None:
    """Store the human's edit. `suggested_reply` is untouched, always."""
    with conn:
        cur = conn.execute(
            f"UPDATE {TABLE} SET edited_reply = ?, updated_at = ? WHERE ticket_id = ?",
            (reply, _now(), ticket_id),
        )
    if cur.rowcount == 0:
        raise RecordNotFound(ticket_id)


def set_status(conn: sqlite3.Connection, ticket_id: str, status: Status | str) -> None:
    """Move a ticket to approved or rejected.

    Only `pending` may move, and only to `approved` or `rejected`. Anything else
    -- re-deciding an approved ticket, moving something back to pending -- is
    refused rather than silently applied. A review queue where a decision can be
    quietly overwritten is not a record of what a human decided.
    """
    target = Status(status)
    if target is Status.PENDING:
        raise InvalidTransition("cannot move a ticket back to pending")

    row = conn.execute(
        f"SELECT status FROM {TABLE} WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    if row is None:
        raise RecordNotFound(ticket_id)

    current = Status(row["status"])
    if current is not Status.PENDING:
        raise InvalidTransition(
            f"{ticket_id} is already {current.value}; only pending tickets can be decided"
        )

    with conn:
        conn.execute(
            f"UPDATE {TABLE} SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (target.value, _now(), ticket_id),
        )
