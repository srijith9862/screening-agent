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
    "recording_consent",
    "recommendation",
    "score_B_recruitment",
    "score_C_remote",
    "score_D_employee_relations",
    "score_E_compliance",
    "knockouts",
    "red_flags",
    "callback_request",
    "summary",
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
    new_file = not path.exists()
    # utf-8-sig so Excel shows ₹ and names correctly.
    with path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})


def already_screened(settings: Settings) -> set[str]:
    """Phones that already have a finished screening (so re-runs skip them)."""
    path = results_path(settings)
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("outcome") in ("completed", "partial", "declined_recording", "not_interested", "wrong_person"):
                done.add(row.get("phone", ""))
    return done


def scorecard_row(card: dict[str, Any]) -> dict[str, Any]:
    scores = {s["section_id"]: s["score"] for s in card.get("sections", [])}
    return {
        "outcome": card.get("call_outcome", ""),
        "recording_consent": card.get("recording_consent", ""),
        "recommendation": card.get("recommendation", ""),
        "score_B_recruitment": scores.get("B", ""),
        "score_C_remote": scores.get("C", ""),
        "score_D_employee_relations": scores.get("D", ""),
        "score_E_compliance": scores.get("E", ""),
        "knockouts": "; ".join(card.get("knockouts", [])),
        "red_flags": "; ".join(card.get("red_flags", [])),
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
