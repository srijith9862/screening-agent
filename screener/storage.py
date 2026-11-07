"""Where call outputs go, the results sheet, and retention clean-up."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .candidates import Candidate
from .config import Settings

RESULTS_COLUMNS = [
    "called_at",
    "name",
    "phone",
    "outcome",
    "recommendation",
    "recommendation_reason",
    "avg_score",
    "score_B_recruitment",
    "score_C_remote",
    "score_D_employee_relations",
    "score_E_compliance",
    "reason_for_change",
    "current_location",
    "current_ctc",
    "expected_ctc",
    "within_budget",
    "ctc_flexibility",
    "notice_period",
    "notice_flexibility",
    "office_commute_ok",
    "knockouts",
    "strengths",
    "red_flags",
    "questions_for_hr",
    "callback_request",
    "callback_due",
    "summary",
    "duration_min",
    "recording_consent",
    "folder",
]


def new_call_dir(settings: Settings, candidate: Candidate) -> Path:
    now = datetime.now()
    path = settings.output_dir / "calls" / now.strftime("%Y-%m-%d") / f"{candidate.slug}-{now.strftime('%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def results_path(settings: Settings) -> Path:
    return settings.output_dir / "results.csv"


def append_result(settings: Settings, row: dict[str, Any]) -> None:
    path = results_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        if header != RESULTS_COLUMNS:  # columns changed: keep the old file, start a new one
            path.rename(path.with_name(f"results-before-{datetime.now():%Y%m%d-%H%M%S}.csv"))
    new_file = not path.exists()
    # utf-8-sig so Excel shows ₹ and names correctly.
    with path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})


FINISHED_OUTCOMES = ("completed", "completed_unscored", "partial", "declined_recording", "not_interested", "wrong_person")


def _all_rows(settings: Settings) -> list[dict[str, str]]:
    """Every results row, oldest file first, in the order they were written."""
    folder = results_path(settings).parent
    paths = [*sorted(folder.glob("results-before-*.csv")), results_path(settings)]
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def _parse_time(text: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(text or "", fmt)
    except ValueError:
        return None


def pending_callbacks(settings: Settings) -> dict[str, datetime]:
    """Phone -> requested call-back time, for candidates whose latest call asked for one."""
    latest: dict[str, dict[str, str]] = {}
    for row in _all_rows(settings):
        latest[row.get("phone", "")] = row
    due = {}
    for phone, row in latest.items():
        when = _parse_time(row.get("callback_due", ""), "%Y-%m-%d %H:%M")
        if when is not None and row.get("outcome") not in FINISHED_OUTCOMES:
            due[phone] = when
    return due


def already_screened(settings: Settings, retry_after_hours: float = 0) -> set[str]:
    """Phones not to call now: finished screenings; call-backs that aren't due yet;
    and other recent attempts within `retry_after_hours` (no instant redials)."""
    now = datetime.now()
    callbacks = pending_callbacks(settings)
    done = set()
    for row in _all_rows(settings):
        phone = row.get("phone", "")
        if row.get("outcome") in FINISHED_OUTCOMES:
            done.add(phone)
            continue
        if phone in callbacks:
            continue  # decided below by the requested time
        called = _parse_time(row.get("called_at", ""), "%Y-%m-%d %H:%M")
        if called is not None and retry_after_hours and (now - called).total_seconds() < retry_after_hours * 3600:
            done.add(phone)
    for phone, when in callbacks.items():
        if when > now and phone not in done:
            done.add(phone)
    return done


def scorecard_row(card: dict[str, Any]) -> dict[str, Any]:
    scores = {s["section_id"]: s["score"] for s in card.get("sections", [])}
    covered = [v for v in scores.values() if isinstance(v, (int, float)) and v > 0]
    facts = card.get("confirmed_facts", {})
    return {
        "outcome": card.get("call_outcome", ""),
        "recording_consent": card.get("recording_consent", ""),
        "recommendation": card.get("recommendation", ""),
        "recommendation_reason": card.get("recommendation_reason", ""),
        "avg_score": round(sum(covered) / len(covered), 1) if covered else "",
        "score_B_recruitment": scores.get("B", ""),
        "score_C_remote": scores.get("C", ""),
        "score_D_employee_relations": scores.get("D", ""),
        "score_E_compliance": scores.get("E", ""),
        **{key: facts.get(key, "") for key in (
            "reason_for_change", "current_location", "current_ctc", "expected_ctc", "within_budget", "ctc_flexibility",
            "notice_period", "notice_flexibility", "office_commute_ok",
        )},
        "knockouts": "; ".join(card.get("knockouts", [])),
        "strengths": "; ".join(card.get("strengths", [])),
        "red_flags": "; ".join(card.get("red_flags", [])),
        "questions_for_hr": "; ".join(card.get("unanswered_candidate_questions", [])),
        "callback_request": card.get("callback_request", ""),
        "summary": card.get("summary", ""),
    }


def purge(settings: Settings, days: int) -> list[Path]:
    """Deletes call folders (recordings, transcripts, scorecards) older than `days`."""
    root = settings.output_dir / "calls"
    if not root.exists():
        return []
    cutoff = time.time() - days * 86400
    removed = []
    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir():
            continue
        for call_dir in sorted(day_dir.iterdir()):
            if call_dir.is_dir() and call_dir.stat().st_mtime < cutoff:
                shutil.rmtree(call_dir)
                removed.append(call_dir)
        if not any(day_dir.iterdir()):
            day_dir.rmdir()
    return removed
