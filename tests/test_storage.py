"""Storage: the two properties that protect a reviewer's work.

In-memory SQLite, so no database file is touched and nothing needs populating.
"""

import pytest

from app.schemas import Category, Priority, TriageRecord
from app.storage import (
    InvalidTransition,
    RecordNotFound,
    Status,
    connect,
    get_record,
    save_record,
    set_edited_reply,
    set_status,
)


@pytest.fixture
def conn():
    connection = connect(":memory:")
    yield connection
    connection.close()


def record(ticket_id: str, summary: str = "The first prediction.") -> TriageRecord:
    return TriageRecord(
        ticket_id=ticket_id,
        category=Category.BILLING,
        priority=Priority.HIGH,
        summary=summary,
        suggested_reply="Thanks for flagging this - could you confirm the invoice numbers?",
        model_confidence=0.8,
        final_confidence=0.97,
        confidence_signals=[],
        escalate=False,
        escalation_reason=None,
        fallback_used=False,
        retries=0,
    )


def test_human_work_survives_a_rerun(conn):
    """11. Re-running triage must never cost a reviewer their work."""
    save_record(conn, record("T-EDIT"))
    save_record(conn, record("T-DECIDED"))
    save_record(conn, record("T-UNTOUCHED"))

    set_edited_reply(conn, "T-EDIT", "Hi Marta, the duplicate charge is refunded.")
    set_status(conn, "T-DECIDED", Status.APPROVED)

    # A later run produces different predictions for all three.
    assert save_record(conn, record("T-EDIT", "A newer prediction.")) == "preserved"
    assert save_record(conn, record("T-DECIDED", "A newer prediction.")) == "preserved"
    assert save_record(conn, record("T-UNTOUCHED", "A newer prediction.")) == "updated"

    edited = get_record(conn, "T-EDIT")
    assert edited.edited_reply == "Hi Marta, the duplicate charge is refunded."
    assert edited.summary == "The first prediction."  # not overwritten
    # The model's draft is never merged into the human's edit.
    assert edited.suggested_reply.startswith("Thanks for flagging")

    assert get_record(conn, "T-DECIDED").status is Status.APPROVED
    assert get_record(conn, "T-DECIDED").summary == "The first prediction."

    # Only the row nobody had touched picked up the new prediction.
    assert get_record(conn, "T-UNTOUCHED").summary == "A newer prediction."


def test_only_pending_can_be_decided_and_only_once(conn):
    """12. A decision that can be quietly overwritten is not a record."""
    save_record(conn, record("T-1"))
    set_status(conn, "T-1", Status.APPROVED)
    assert get_record(conn, "T-1").status is Status.APPROVED

    # Re-deciding, in any direction, is refused.
    for target in (Status.REJECTED, Status.APPROVED):
        with pytest.raises(InvalidTransition):
            set_status(conn, "T-1", target)

    # So is moving anything back to pending.
    save_record(conn, record("T-2"))
    with pytest.raises(InvalidTransition):
        set_status(conn, "T-2", Status.PENDING)
    assert get_record(conn, "T-2").status is Status.PENDING

    # A value outside the enum never reaches the database.
    with pytest.raises(ValueError):
        set_status(conn, "T-2", "archived")

    # An unknown ticket is a lookup failure, not a silent no-op.
    with pytest.raises(RecordNotFound):
        set_status(conn, "T-404", Status.APPROVED)
    with pytest.raises(RecordNotFound):
        set_edited_reply(conn, "T-404", "x")
