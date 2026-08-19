"""Triage every ticket in data/tickets.json and write the results to SQLite.

Run from the repository root:

    python scripts/run_triage.py

Safe to re-run. A ticket whose row a human has already touched -- decided, or
its reply edited -- is left alone and counted as preserved. Untouched pending
rows are refreshed with the new prediction. See app/storage.save_record.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Python puts this script's own directory on sys.path, not the working
# directory, so `import app` would fail on a plain `python scripts/run_triage.py`.
# Adding the repository root keeps that invocation working without turning the
# project into an installable package for one script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.storage import DEFAULT_DB_PATH, connect, list_records, save_record  # noqa: E402
from app.triage import triage_ticket  # noqa: E402

TICKETS = Path("data/tickets.json")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The SDK logs one INFO line per HTTP request; the triage logs already say
    # everything those would, with the ticket id attached.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not TICKETS.exists():
        print(f"error: {TICKETS} not found. Run from the repository root.", file=sys.stderr)
        return 1

    settings = get_settings()
    tickets = json.loads(TICKETS.read_text(encoding="utf-8"))

    print(f"model     : {settings.llm_model} at {settings.llm_base_url}")
    print(f"threshold : {settings.confidence_threshold}")
    print(f"budget    : {settings.llm_max_retries + 1} calls per ticket")
    print(f"database  : {DEFAULT_DB_PATH}")
    print(f"tickets   : {len(tickets)}")
    print()

    conn = connect()
    outcomes: Counter[str] = Counter()
    started = time.monotonic()

    try:
        for ticket in tickets:
            record = triage_ticket(ticket)
            outcomes[save_record(conn, record)] += 1
        elapsed = time.monotonic() - started

        stored = list_records(conn)
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  runtime          : {elapsed / 60:.1f} min ({elapsed:.0f}s), "
              f"{elapsed / len(tickets):.1f}s per ticket")
        print(f"  written          : {dict(outcomes)}")
        print()
        print(f"  by status        : {dict(Counter(r.status.value for r in stored))}")
        print(f"  by category      : {dict(Counter(r.category.value for r in stored))}")
        print(f"  by priority      : {dict(Counter(r.priority.value for r in stored))}")
        print(f"  escalated        : {sum(r.escalate for r in stored)} / {len(stored)}")
        print(f"  fallbacks        : {sum(r.fallback_used for r in stored)} / {len(stored)}")
        print(f"  needed a repair  : {sum(1 for r in stored if r.retries and not r.fallback_used)}"
              f" / {len(stored)}")

        if outcomes.get("preserved"):
            print()
            print(f"  {outcomes['preserved']} record(s) preserved: a human had already decided or")
            print("  edited them, so the new prediction was not written over their work.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
