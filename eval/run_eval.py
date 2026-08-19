"""Score the stored predictions against the labeled subset.

Run from the repository root, after scripts/run_triage.py:

    python eval/run_eval.py

Writes eval/results.json (the graded artefact, a literal match to the shape the
brief specifies) and eval/diagnostics.json (everything needed to write the error
analysis). Prints a per-ticket report to stdout.

Four rules this file exists to get right:

1. **Metrics are computed over the 16 LABELED tickets only.** Dividing by 30
   would score the unlabeled ones as wrong, when the honest position is that
   there is no ground truth for them.
2. **`predictions` contains all 30.** The brief says so.
3. **labels.json spells it `feature_request`; canonical is `feature request`.**
   Converted once, at load, reusing the same normaliser the model output goes
   through. Getting this wrong scores all three feature tickets as errors for
   no reason at all.
4. **Every metric is Python comparing strings.** The model computes nothing
   here, and predictions are READ from the database rather than regenerated --
   the eval and the UI must report the same numbers, and Phase 0 showed four of
   five identical calls disagreeing at temperature=0.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Python puts this script's directory on sys.path, not the working directory,
# so `import app` needs the repository root added.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.schemas import Category, Priority, normalize_llm_payload  # noqa: E402
from app.storage import DEFAULT_DB_PATH, StoredRecord, connect, list_records  # noqa: E402

LABELS_PATH = Path("data/labels.json")
RESULTS_PATH = Path("eval/results.json")
DIAGNOSTICS_PATH = Path("eval/diagnostics.json")
EXPECTED_TICKETS = 30

# Ordinal ranks taken from the enum's declaration order rather than a
# hand-written dict, so the two cannot drift: low < medium < high < urgent.
PRIORITY_RANK = {p: i for i, p in enumerate(Priority)}


def load_labels(path: Path = LABELS_PATH) -> dict[str, tuple[Category, Priority]]:
    """Load ground truth, canonicalised through the same path as model output.

    `normalize_llm_payload` maps underscores to spaces, which is what turns
    labels.json's `feature_request` into the canonical `feature request`. Doing
    it with the model's own normaliser means there is one canonicalisation rule
    in the project, not two that can drift apart.

    Constructing Category/Priority afterwards makes an unrecognised spelling
    fail loudly here, rather than silently scoring every ticket in that class as
    wrong -- which is the exact failure this conversion exists to prevent.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, tuple[Category, Priority]] = {}
    for entry in raw["labels"]:
        normalized = normalize_llm_payload(entry)
        labels[entry["id"]] = (
            Category(normalized["category"]),
            Priority(normalized["priority"]),
        )
    return labels


def load_predictions(db_path: Path = DEFAULT_DB_PATH) -> list[StoredRecord]:
    """Read every stored record, refusing to proceed on an incomplete database.

    A missing row is a pipeline failure, not a model error. Scoring it as a miss
    would understate accuracy for a reason that has nothing to do with the
    model, and would do it silently.
    """
    if not Path(db_path).exists():
        raise SystemExit(
            f"error: {db_path} not found.\n"
            "Run `python scripts/run_triage.py` first to populate the database."
        )
    conn = connect(db_path)
    try:
        records = list_records(conn)
    finally:
        conn.close()

    if len(records) < EXPECTED_TICKETS:
        raise SystemExit(
            f"error: found {len(records)} rows in {db_path}, expected {EXPECTED_TICKETS}.\n"
            "Run `python scripts/run_triage.py` to triage every ticket."
        )
    return records


def score(
    labels: dict[str, tuple[Category, Priority]],
    predictions: dict[str, StoredRecord],
) -> dict:
    """Compute every metric. Pure -- no I/O, so it is testable without a database.

    Only the labeled ids reach the numerator or the denominator.
    """
    missing = sorted(set(labels) - set(predictions))
    if missing:
        raise SystemExit(
            f"error: no prediction for labeled ticket(s): {', '.join(missing)}.\n"
            "Run `python scripts/run_triage.py` to triage every ticket."
        )

    labeled_ids = sorted(labels)
    n = len(labeled_ids)

    category_hits = 0
    priority_exact = 0
    priority_within_one = 0
    over = 0
    under = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for tid in labeled_ids:
        true_cat, true_pri = labels[tid]
        pred = predictions[tid]

        if pred.category is true_cat:
            category_hits += 1
        # actual -> predicted, so each row reads "of the tickets truly labeled
        # X, this is where they went".
        confusion[true_cat.value][pred.category.value] += 1

        distance = PRIORITY_RANK[pred.priority] - PRIORITY_RANK[true_pri]
        if distance == 0:
            priority_exact += 1
        if abs(distance) <= 1:
            priority_within_one += 1
        if distance > 0:
            over += 1
        elif distance < 0:
            under += 1

    # Baselines are recomputed from the labels rather than hardcoded, so they
    # stay correct if the labeled subset ever changes.
    cat_counts = Counter(c for c, _p in labels.values())
    pri_counts = Counter(p for _c, p in labels.values())
    top_cat, top_cat_n = cat_counts.most_common(1)[0]
    top_pri, top_pri_n = pri_counts.most_common(1)[0]

    all_confidences = sorted(r.final_confidence for r in predictions.values())
    buckets: Counter[str] = Counter()
    for value in all_confidences:
        low = min(int(value * 10) / 10, 0.9)
        buckets[f"{low:.1f}-{low + 0.1:.1f}"] += 1

    return {
        # --- the two the brief requires ---
        "category_accuracy": round(category_hits / n, 4),
        "priority_agreement": round(priority_exact / n, 4),
        # --- scale: without these the numbers above mean nothing ---
        "category_accuracy_baseline": round(top_cat_n / n, 4),
        "priority_agreement_baseline": round(top_pri_n / n, 4),
        "baseline_category": top_cat.value,
        "baseline_priority": top_pri.value,
        # --- the visible denominator ---
        "labeled_count": n,
        "total_predictions": len(predictions),
        # --- priority is ordinal, so exact agreement is not the whole story ---
        "priority_within_one": round(priority_within_one / n, 4),
        "priority_exact_count": priority_exact,
        "priority_over_estimated": over,
        "priority_under_estimated": under,
        # --- where the category errors actually went ---
        "category_confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        # --- pipeline health, over all 30 ---
        "escalated_count": sum(r.escalate for r in predictions.values()),
        "fallback_count": sum(r.fallback_used for r in predictions.values()),
        "repair_retry_count": sum(
            1 for r in predictions.values() if r.retries and not r.fallback_used
        ),
        "confidence_distribution": dict(sorted(buckets.items())),
        "confidence_min": all_confidences[0],
        "confidence_max": all_confidences[-1],
        "confidence_mean": round(sum(all_confidences) / len(all_confidences), 4),
    }


def build_predictions(records: list[StoredRecord]) -> list[dict]:
    """The graded artefact: exactly the seven keys the brief specifies.

    `confidence` is final_confidence, the system's number, because `escalate`
    sits beside it and is derived from it -- pairing escalate with a number it
    does not follow from would make each record incoherent to read.

    `suggested_reply` is always the model's draft, never `edited_reply`. The
    eval measures the model; a human's improvement is not the model's output.
    """
    return [
        {
            "id": r.ticket_id,
            "category": r.category.value,
            "priority": r.priority.value,
            "summary": r.summary,
            "suggested_reply": r.suggested_reply,
            "confidence": r.final_confidence,
            "escalate": r.escalate,
        }
        for r in sorted(records, key=lambda r: r.ticket_id)
    ]


def build_diagnostics(
    records: list[StoredRecord],
    labels: dict[str, tuple[Category, Priority]],
    metrics: dict,
) -> dict:
    """Everything the error analysis needs, kept out of the graded artefact."""
    settings = get_settings()
    tickets = []
    for r in sorted(records, key=lambda r: r.ticket_id):
        label = labels.get(r.ticket_id)
        entry = {
            "id": r.ticket_id,
            "category": r.category.value,
            "priority": r.priority.value,
            "model_confidence": r.model_confidence,
            "final_confidence": r.final_confidence,
            "escalate": r.escalate,
            "escalation_reason": r.escalation_reason,
            "confidence_signals": r.confidence_signals,
            "fallback_used": r.fallback_used,
            "retries": r.retries,
            # null rather than absent, so "unlabeled" is explicit in the data
            # rather than inferred from a missing key.
            "label_category": None,
            "label_priority": None,
            "category_match": None,
            "priority_match": None,
            "priority_distance": None,
        }
        if label is not None:
            true_cat, true_pri = label
            entry.update(
                label_category=true_cat.value,
                label_priority=true_pri.value,
                category_match=r.category is true_cat,
                priority_match=r.priority is true_pri,
                priority_distance=PRIORITY_RANK[r.priority] - PRIORITY_RANK[true_pri],
            )
        tickets.append(entry)

    return {
        # Provenance: an unlabelled number is not a result.
        "run": {
            "model": settings.llm_model,
            "confidence_threshold": settings.confidence_threshold,
            "llm_max_retries": settings.llm_max_retries,
            "llm_max_tokens": settings.llm_max_tokens,
            "source_database": str(DEFAULT_DB_PATH),
            "labeled_count": metrics["labeled_count"],
            "total_predictions": metrics["total_predictions"],
        },
        "metrics": metrics,
        "tickets": tickets,
    }


def print_report(
    metrics: dict,
    labels: dict[str, tuple[Category, Priority]],
    predictions: dict[str, StoredRecord],
) -> None:
    m = metrics
    print("=" * 78)
    print(f"EVALUATION  ({m['labeled_count']} labeled of {m['total_predictions']} tickets)")
    print("=" * 78)

    for name, key, base_key, base_name in [
        ("category_accuracy", "category_accuracy", "category_accuracy_baseline", "baseline_category"),
        ("priority_agreement", "priority_agreement", "priority_agreement_baseline", "baseline_priority"),
    ]:
        got, base = m[key], m[base_key]
        delta = got - base
        hits = round(got * m["labeled_count"])
        print(f"\n  {name:<20} {got:.4f}  ({hits}/{m['labeled_count']})")
        print(f"  {'majority baseline':<20} {base:.4f}  (always \"{m[base_name]}\")")
        print(f"  {'difference':<20} {delta:+.4f}  = {delta * m['labeled_count']:+.1f} tickets")

    print(f"\n  priority_within_one  {m['priority_within_one']:.4f}  "
          f"(exact {m['priority_exact_count']}/{m['labeled_count']})")
    print(f"    over-estimated  {m['priority_over_estimated']}  (noise in the queue)")
    print(f"    under-estimated {m['priority_under_estimated']}  (the costly direction)")

    print("\n" + "=" * 78)
    print("CATEGORY CONFUSION  (rows = actual label, columns = predicted)")
    print("=" * 78)
    cats = [c.value for c in Category]
    # Bound outside the f-string: Python 3.10 forbids a backslash in the
    # expression part of an f-string.
    corner = r"actual \ pred"
    print(f"  {corner:<17}" + "".join(f"{c[:8]:>10}" for c in cats))
    for actual in cats:
        row = m["category_confusion"].get(actual)
        if not row:
            continue
        cells = "".join(f"{row.get(c, 0) or '.':>10}" for c in cats)
        print(f"  {actual:<17}{cells}")

    print("\n" + "=" * 78)
    print("PER-TICKET (labeled only)   * marks a mismatch")
    print("=" * 78)
    print(f"  {'id':<7} {'category':<32} {'priority':<26} conf")
    for tid in sorted(labels):
        true_cat, true_pri = labels[tid]
        p = predictions[tid]
        cat_ok = p.category is true_cat
        pri_ok = p.priority is true_pri
        dist = PRIORITY_RANK[p.priority] - PRIORITY_RANK[true_pri]
        cat_cell = f"{p.category.value} vs {true_cat.value}"
        pri_cell = f"{p.priority.value} vs {true_pri.value}"
        if not pri_ok:
            pri_cell += f" ({dist:+d})"
        print(f"  {tid:<7} {'' if cat_ok else '*'}{cat_cell:<31} "
              f"{'' if pri_ok else '*'}{pri_cell:<25} {p.final_confidence:.3f}")

    print("\n" + "=" * 78)
    print("PIPELINE (all 30)")
    print("=" * 78)
    print(f"  escalated {m['escalated_count']}   fallbacks {m['fallback_count']}   "
          f"repair retries {m['repair_retry_count']}")
    print(f"  final_confidence  min {m['confidence_min']:.3f}  "
          f"mean {m['confidence_mean']:.3f}  max {m['confidence_max']:.3f}")
    for bucket, count in m["confidence_distribution"].items():
        print(f"    {bucket}  {'#' * count} ({count})")


def main() -> int:
    labels = load_labels()
    records = load_predictions()
    predictions = {r.ticket_id: r for r in records}

    metrics = score(labels, predictions)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The graded artefact: metrics and predictions, nothing else.
    RESULTS_PATH.write_text(
        json.dumps({"metrics": metrics, "predictions": build_predictions(records)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    DIAGNOSTICS_PATH.write_text(
        json.dumps(build_diagnostics(records, labels, metrics), indent=2) + "\n",
        encoding="utf-8",
    )

    print_report(metrics, labels, predictions)
    print(f"\nwrote {RESULTS_PATH} and {DIAGNOSTICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
