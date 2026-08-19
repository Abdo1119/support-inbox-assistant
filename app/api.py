"""HTTP API over the storage layer. Four endpoints, and deliberately no fifth.

**There is no send endpoint.** Nothing in this file, or anywhere else in the
project, can email a customer. That safety property is enforced by absence
rather than by a runtime check: a guard can be disabled or mis-configured, a
route that does not exist cannot be. Please do not add one.

This layer is thin on purpose. It makes no decisions -- the review state machine
lives in app/storage.py, and the schemas that validate untrusted model output in
app/schemas.py are the same ones that shape these responses. That reuse is the
main reason this project uses FastAPI at all.

Two implementation notes worth knowing before editing:

- Every path operation is `def`, not `async def`. sqlite3 is blocking, so an
  async handler would block the event loop; declaring them sync makes FastAPI
  run them in its threadpool instead.
- Each request gets its own connection. SQLite connections are not safe to share
  across threads, and the threadpool means concurrent requests land on different
  ones.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.storage import (
    DEFAULT_DB_PATH,
    InvalidTransition,
    RecordNotFound,
    Status,
    StoredRecord,
    connect,
    get_record,
    list_records,
    set_edited_reply,
    set_status,
)

STATIC_DIR = Path("static")

_RUN_TRIAGE = "Run `python scripts/run_triage.py` to populate it."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Refuse to start against a missing or empty database.

    Serving an empty list would render as a review queue with nothing in it --
    a system in perfect health with no work to do. That is indistinguishable
    from the truth, which is that triage was never run. Failing at startup makes
    the difference impossible to miss.
    """
    if not DEFAULT_DB_PATH.exists():
        raise RuntimeError(f"database {DEFAULT_DB_PATH} not found. {_RUN_TRIAGE}")
    conn = connect(DEFAULT_DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) FROM triage").fetchone()[0]
    finally:
        conn.close()
    if count == 0:
        raise RuntimeError(f"database {DEFAULT_DB_PATH} holds no tickets. {_RUN_TRIAGE}")
    yield


app = FastAPI(
    title="Support Inbox Assistant",
    description=(
        "First-pass triage in a human review queue. Nothing is ever sent to a "
        "customer; there is no endpoint that could."
    ),
    lifespan=lifespan,
)


def get_conn() -> Iterator[sqlite3.Connection]:
    """One connection per request, closed afterwards."""
    conn = connect(DEFAULT_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# --- Exception mapping -------------------------------------------------------
#
# Handled centrally so the path operations stay free of error plumbing. FastAPI
# already returns 422 for a request body that fails Pydantic validation, which
# is the third case.


@app.exception_handler(RecordNotFound)
def _not_found(request: Request, exc: RecordNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"no ticket {exc}"})


@app.exception_handler(InvalidTransition)
def _conflict(request: Request, exc: InvalidTransition) -> JSONResponse:
    # 409 rather than 400: the request is well formed and would have been valid
    # against a different state. The conflict is with reality, not the syntax.
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# --- Request bodies ----------------------------------------------------------


class EditReplyRequest(BaseModel):
    """The reviewer's edit of the drafted reply."""

    # min_length=1 because edited_reply is NULL until someone edits it, and an
    # empty string would create a third state -- "edited, to nothing" -- that
    # means nothing to a reader.
    edited_reply: str = Field(min_length=1)


class Decision(str, Enum):
    """The only two states a reviewer may move a ticket to.

    Values match Status, so they pass straight through to the storage layer.
    `pending` is absent on purpose: asking to move a ticket back to pending is a
    malformed request (422), not a state conflict (409).
    """

    APPROVED = "approved"
    REJECTED = "rejected"


class DecisionRequest(BaseModel):
    status: Decision


# --- Endpoints ---------------------------------------------------------------


@app.get("/api/tickets", response_model=list[StoredRecord])
def read_tickets(
    status: Status | None = None,
    escalate: bool | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[StoredRecord]:
    """List tickets, optionally filtered by review status and escalation.

    Ordering comes from the storage layer: escalated first, then lowest
    confidence first -- the order a reviewer working the queue wants.
    """
    return list_records(conn, status=status, escalate=escalate)


@app.get("/api/tickets/{ticket_id}", response_model=StoredRecord)
def read_ticket(
    ticket_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> StoredRecord:
    record = get_record(conn, ticket_id)
    if record is None:
        raise RecordNotFound(ticket_id)
    return record


@app.patch("/api/tickets/{ticket_id}", response_model=StoredRecord)
def edit_reply(
    ticket_id: str,
    body: EditReplyRequest,
    conn: sqlite3.Connection = Depends(get_conn),
) -> StoredRecord:
    """Store the reviewer's edit. The model's draft is never overwritten."""
    set_edited_reply(conn, ticket_id, body.edited_reply)
    record = get_record(conn, ticket_id)
    assert record is not None  # set_edited_reply raises RecordNotFound otherwise
    return record


@app.post("/api/tickets/{ticket_id}/decision", response_model=StoredRecord)
def decide(
    ticket_id: str,
    body: DecisionRequest,
    conn: sqlite3.Connection = Depends(get_conn),
) -> StoredRecord:
    """Approve or reject. Only a pending ticket can be decided, once."""
    set_status(conn, ticket_id, Status(body.status.value))
    record = get_record(conn, ticket_id)
    assert record is not None
    return record


# Mounted last, so it cannot shadow the /api routes registered above.
# html=True serves static/index.html at "/", which is all the next phase needs:
# same origin, so no CORS, and no build step.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
