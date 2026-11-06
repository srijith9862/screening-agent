"""Controls the Android phone (Galaxy S24 Ultra) over ADB: dial, hang up, call state."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from dataclasses import dataclass

# TelephonyManager call states
CALL_STATE_IDLE = 0
CALL_STATE_RINGING = 1
CALL_STATE_OFFHOOK = 2


class PhoneError(RuntimeError):
    pass


def parse_call_state(dumpsys_output: str) -> int:
    """Parses `dumpsys telephony.registry`. Returns the highest state across SIMs."""
    states = [int(m) for m in re.findall(r"mCallState=(\d)", dumpsys_output)]
    return max(states) if states else CALL_STATE_IDLE


def count_telecom_active(dumpsys_output: str) -> int:
    """Counts mentions of a call in the ACTIVE state in `dumpsys telecom`.

    The dump also contains event history, so callers compare against a count
    taken before dialing: the call was answered once the count goes up.
    """
    return len(re.findall(r"(?i)\bstate\s*[=:]\s*ACTIVE\b", dumpsys_output))


@dataclass
class Phone:
    adb_path: str = "adb"
    serial: str | None = None

    def _cmd(self, *args: str) -> list[str]:
        base = [self.adb_path]
        if self.serial:
            base += ["-s", self.serial]
        return base + list(args)

    def _run(self, *args: str, timeout: float = 10) -> str:
        try:
            # Android output is UTF-8; Windows would otherwise decode it as cp1252 and crash.
            result = subprocess.run(
                self._cmd(*args),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise PhoneError(
                f"ADB not found at '{self.adb_path}'. Install Android platform-tools and set phone.adb_path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PhoneError(f"ADB timed out: {' '.join(args)}") from exc
        if result.returncode != 0:
            raise PhoneError((result.stderr or "").strip() or (result.stdout or "").strip() or "ADB command failed")
        return result.stdout or ""

    async def _arun(self, *args: str, timeout: float = 10) -> str:
        return await asyncio.to_thread(self._run, *args, timeout=timeout)

    # --- checks -----------------------------------------------------------

    def available(self) -> bool:
        return shutil.which(self.adb_path) is not None or self.adb_path.lower().endswith(".exe")

    def connected_devices(self) -> list[str]:
        out = self._run("devices")
        return [line.split()[0] for line in out.splitlines()[1:] if line.strip().endswith("device")]

    # --- call control -------------------------------------------------------

    async def dial(self, number: str) -> None:
        if not re.fullmatch(r"\+?\d{10,15}", number):
            raise PhoneError(f"Refusing to dial an invalid number: {number!r}")
        await self._arun("shell", "am", "start", "-a", "android.intent.action.CALL", "-d", f"tel:{number}")

    async def hang_up(self) -> None:
        try:
            await self._arun("shell", "input", "keyevent", "KEYCODE_ENDCALL")
        except PhoneError:
            pass

    async def call_state(self) -> int:
        return parse_call_state(await self._arun("shell", "dumpsys", "telephony.registry"))

    async def active_count(self) -> int:
        return count_telecom_active(await self._arun("shell", "dumpsys", "telecom"))

    def debug_state(self) -> str:
        registry = self._run("shell", "dumpsys", "telephony.registry")
        telecom = self._run("shell", "dumpsys", "telecom")
        state_lines = [line.strip() for line in registry.splitlines() if "mCallState" in line]
        telecom_lines = [line.strip() for line in telecom.splitlines() if re.search(r"(?i)state", line)][:40]
        return (
            "telephony.registry:\n  " + "\n  ".join(state_lines or ["(no mCallState lines)"])
            + "\n\ntelecom (lines mentioning state):\n  " + "\n  ".join(telecom_lines or ["(none)"])
        )
