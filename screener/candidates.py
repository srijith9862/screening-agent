"""Reads the candidate list (e.g. the Naukri shortlist export) from CSV."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# CSV column -> field name. Column names are matched case-insensitively,
# ignoring spaces and underscores, so "Phone", "phone_number" and
# "Phone Number" all work.
COLUMN_ALIASES: dict[str, list[str]] = {
    "name": ["name", "candidatename"],
    "phone": ["phone", "phonenumber", "mobile", "mobilenumber", "contactnumber"],
    "hr_experience": ["hrexperience"],
    "e2e_recruitment_experience": ["e2erecruitmentexperience", "recruitmentexperience"],
    "total_experience": ["totalexperience"],
    "current_role": ["currentroleandcompany", "currentrole"],
    "location": ["location"],
    "current_ctc": ["currentctc"],
    "expected_ctc": ["expectedctc"],
    "notice_period": ["noticeperiod"],
    "office_ok": ["malakpetoffice", "officeok"],
    "remote_hr_experience": ["remotehrexperience"],
    "english": ["english"],
    "education": ["education"],
    # Internal recruiter notes: used only to pick follow-up questions,
    # never read out to the candidate.
    "internal_notes": ["reasonstrengthsandgaps", "notes", "internalnotes"],
    "preferred_call_time": ["preferredcalltime", "calltime"],
}

# Columns deliberately NOT passed to the AI (ratings would bias the live
# agent and the scorer): Category, Rating_out_of_10.


def _norm(header: str) -> str:
    return re.sub(r"[\s_\-]+", "", header.strip().lower())


@dataclass
class Candidate:
    name: str
    phone: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    row_number: int = 0

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else ""

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or f"row-{self.row_number}"

    def script_values(self) -> dict[str, str]:
        values = {k: v for k, v in self.fields.items() if k != "internal_notes"}
        values["name"] = self.name
        values["first_name"] = self.first_name
        return values

    def application_facts(self) -> dict[str, str]:
        labels = {
            "total_experience": "Total experience",
            "hr_experience": "HR experience",
            "e2e_recruitment_experience": "End-to-end recruitment experience",
            "current_role": "Current role and company",
            "location": "Location",
            "current_ctc": "Current CTC",
            "expected_ctc": "Expected CTC",
            "notice_period": "Notice period",
            "office_ok": "Can work from Malakpet office (Naukri answer)",
            "remote_hr_experience": "Has managed remote employees (Naukri answer)",
            "english": "English (Naukri profile)",
            "education": "Education",
            "preferred_call_time": "Preferred call time",
        }
        return {label: self.fields[key] for key, label in labels.items() if self.fields.get(key)}


def normalize_phone(raw: str, default_country_code: str = "+91") -> str:
    """Returns +<digits>, or "" when the number doesn't look valid."""
    if not raw:
        return ""
    # Take the first phone-like number, e.g. "+91 98765 43210 (WhatsApp: ...)" -> "+91 98765 43210".
    match = re.search(r"\+?\d[\d\s().-]{8,}\d", raw)
    if not match:
        return ""
    raw = match.group(0).strip()
    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)
    if has_plus:
        return "+" + digits if 10 <= len(digits) <= 15 else ""
    if digits.startswith("00"):
        return "+" + digits[2:]
    cc = re.sub(r"\D", "", default_country_code)
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"+{cc}{digits}"
    if len(digits) == 10 + len(cc) and digits.startswith(cc):
        return "+" + digits
    return ""


def load_candidates(path: Path, default_country_code: str = "+91") -> list[Candidate]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header_map: dict[str, str] = {}
        for header in reader.fieldnames or []:
            normalized = _norm(header)
            for key, aliases in COLUMN_ALIASES.items():
                if normalized in aliases and key not in header_map.values():
                    header_map[header] = key
                    break
        candidates: list[Candidate] = []
        for index, row in enumerate(reader, start=2):  # row 1 is the header
            mapped = {header_map[h]: (v or "").strip() for h, v in row.items() if h in header_map}
            name = mapped.pop("name", "").strip()
            if not name:
                continue
            phone = normalize_phone(mapped.pop("phone", ""), default_country_code)
            candidates.append(Candidate(name=name, phone=phone, fields=mapped, row_number=index))
    return candidates
