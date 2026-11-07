"""Turns a candidate's call-back request ("in 5 minutes", "after 6 pm") into a time."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "half an": 30,
}


def parse_callback(text: str, now: datetime) -> datetime | None:
    """Best-effort time for a call-back request; None if it can't be worked out."""
    if not text:
        return None
    t = text.lower().strip()

    # "half an hour"
    if re.search(r"\bhalf an hour\b", t):
        return now + timedelta(minutes=30)

    # "in 5 minutes", "after ten mins", "2 hours later"
    m = re.search(r"\b(\d+|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\s*(minutes?|mins?|hours?|hrs?)\b", t)
    if m:
        amount = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMBERS[m.group(1)]
        unit = m.group(2)
        delta = timedelta(hours=amount) if unit.startswith(("hour", "hr")) else timedelta(minutes=amount)
        return now + delta

    # "6 pm", "6:30 pm", "18:00", "at 11"
    day = now
    if "tomorrow" in t:
        day = now + timedelta(days=1)
    m = re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", t)
    if m and (m.group(3) or m.group(2) or re.search(r"\b(at|after|around|by)\s+\d", t)):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").replace(".", "")
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and 1 <= hour <= 7:
            hour += 12  # "at 6" for a call-back means 6 pm
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return None
        when = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if when <= now and "tomorrow" not in t:
            when += timedelta(days=1)
        return when

    if "tomorrow" in t:
        return (now + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
    if re.search(r"\b(evening)\b", t):
        when = now.replace(hour=18, minute=0, second=0, microsecond=0)
        return when if when > now else when + timedelta(days=1)
    if re.search(r"\b(later|some time|sometime)\b", t):
        return now + timedelta(hours=1)
    return None
