"""Loads settings.yaml, script.yaml and faq.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class _SafeDict(dict):
    """str.format_map helper that leaves unknown {placeholders} as they are."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def fill(template: str, values: dict[str, Any]) -> str:
    return template.format_map(_SafeDict({k: v for k, v in values.items() if v not in (None, "")}))


@dataclass
class Settings:
    raw: dict[str, Any]
    script: dict[str, Any]
    faq: list[dict[str, str]]
    root: Path = ROOT

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def agent_name(self) -> str:
        return self.raw.get("agent_name", "Maya")

    @property
    def output_dir(self) -> Path:
        out = Path(self.get("storage.output_dir", "output"))
        return out if out.is_absolute() else self.root / out

    def script_values(self) -> dict[str, Any]:
        """Values for {placeholders} that are the same for every candidate."""
        values = {
            "company": self.script.get("company", ""),
            "role": self.script.get("role", ""),
            "location": self.script.get("location", ""),
            "target_minutes": self.script.get("target_minutes", 12),
        }
        values.update(self.raw.get("script_values") or {})
        return values

    def usable_faq(self) -> list[dict[str, str]]:
        """FAQ entries with real answers (TODO entries are left out)."""
        return [
            item
            for item in self.faq
            if item.get("answer") and not str(item["answer"]).strip().upper().startswith("TODO")
        ]

    def todo_warnings(self) -> list[str]:
        warnings = [
            f"faq.yaml: '{item.get('topic')}' still has a TODO answer"
            for item in self.faq
            if str(item.get("answer", "")).strip().upper().startswith("TODO")
        ]
        for key, value in (self.raw.get("script_values") or {}).items():
            if str(value).strip().upper().startswith("TODO"):
                warnings.append(f"settings.yaml: script_values.{key} is still TODO")
        return warnings


def load_settings(config_dir: Path | None = None) -> Settings:
    config_dir = config_dir or CONFIG_DIR
    raw = yaml.safe_load((config_dir / "settings.yaml").read_text(encoding="utf-8")) or {}
    script = yaml.safe_load((config_dir / "script.yaml").read_text(encoding="utf-8")) or {}
    faq_doc = yaml.safe_load((config_dir / "faq.yaml").read_text(encoding="utf-8")) or {}
    return Settings(raw=raw, script=script, faq=faq_doc.get("faq", []), root=config_dir.parent)


def load_env() -> None:
    """Load API keys from .env (never from the YAML files)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is in requirements.txt
        return
    load_dotenv(ROOT / ".env")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set. Add it to the .env file (see .env.example).")
    return value


@dataclass
class TimeWindow:
    start_minutes: int
    end_minutes: int

    @classmethod
    def parse(cls, text: str) -> "TimeWindow":
        start, end = (part.strip() for part in text.split("-"))
        return cls(_to_minutes(start), _to_minutes(end))

    def contains(self, hour: int, minute: int) -> bool:
        now = hour * 60 + minute
        return self.start_minutes <= now < self.end_minutes


def _to_minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


__all__ = ["Settings", "load_settings", "load_env", "require_env", "fill", "TimeWindow"]
