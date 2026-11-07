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


UI_DUMP_PATH = "/sdcard/screener_ui.xml"
BLUETOOTH_PATTERN = re.compile(r"(?i)\bbluetooth\b")


@dataclass
class UiNode:
    label: str
    bounds: tuple[int, int, int, int]
    clickable: bool
    selected: bool

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2


def parse_ui_nodes(xml_text: str) -> list[UiNode]:
    """Parses `uiautomator dump` output into labelled, on-screen nodes."""
    import xml.etree.ElementTree as ET

    start = xml_text.find("<?xml")
    if start == -1:
        start = xml_text.find("<hierarchy")
    if start == -1:
        return []
    try:
        root = ET.fromstring(xml_text[start:].strip())
    except ET.ParseError:
        return []
    nodes = []
    for el in root.iter("node"):
        label = (el.get("text") or "").strip() or (el.get("content-desc") or "").strip()
        match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", el.get("bounds", ""))
        if not label or not match:
            continue
        x1, y1, x2, y2 = (int(v) for v in match.groups())
        if x2 <= x1 or y2 <= y1:
            continue
        nodes.append(
            UiNode(
                label=label,
                bounds=(x1, y1, x2, y2),
                clickable=el.get("clickable") == "true",
                selected=el.get("selected") == "true" or el.get("checked") == "true",
            )
        )
    return nodes


def find_node(nodes: list[UiNode], pattern: re.Pattern[str]) -> UiNode | None:
    """First node whose label matches; clickable ones preferred."""
    matches = [n for n in nodes if pattern.search(n.label)]
    matches.sort(key=lambda n: not n.clickable)
    return matches[0] if matches else None


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

    async def tap_end_call_on_screen(self) -> bool:
        """Taps the red end-call button on the phone's call screen."""
        button = find_node(await self.screen_nodes(), re.compile(r"(?i)^(end call|end|hang up|decline)$"))
        if button is None:
            return False
        await self.tap(*button.center)
        return True

    async def hang_up(self) -> None:
        try:
            await self._arun("shell", "input", "keyevent", "KEYCODE_ENDCALL")
        except PhoneError:
            pass

    async def call_state(self) -> int:
        return parse_call_state(await self._arun("shell", "dumpsys", "telephony.registry"))

    async def active_count(self) -> int:
        return count_telecom_active(await self._arun("shell", "dumpsys", "telecom"))

    # --- on-screen buttons (in-call audio route) ------------------------------

    async def screen_nodes(self) -> list[UiNode]:
        """Reads the phone's current screen (needs the phone unlocked)."""
        await self._arun("shell", "uiautomator", "dump", UI_DUMP_PATH, timeout=15)
        return parse_ui_nodes(await self._arun("shell", "cat", UI_DUMP_PATH))

    async def tap(self, x: int, y: int) -> None:
        await self._arun("shell", "input", "tap", str(x), str(y))

    async def bluetooth_button(self) -> UiNode | None:
        return find_node(await self.screen_nodes(), BLUETOOTH_PATTERN)

    async def tap_bluetooth(self, device_name: str = "") -> str:
        """Taps the in-call Bluetooth button and, if a device list opens, the laptop.

        Returns a short description of what happened.
        """
        button = await self.bluetooth_button()
        if button is None:
            return "Bluetooth button not found on the call screen (is the phone unlocked?)"
        await self.tap(*button.center)
        await asyncio.sleep(1.5)
        # Some phones open a list of audio devices instead of switching directly.
        nodes = await self.screen_nodes()
        if device_name:
            choice = find_node(nodes, re.compile(re.escape(device_name), re.I))
            if choice is not None:
                await self.tap(*choice.center)
                return f"tapped Bluetooth, then '{choice.label}'"
        return "tapped Bluetooth"

    def debug_state(self) -> str:
        registry = self._run("shell", "dumpsys", "telephony.registry")
        telecom = self._run("shell", "dumpsys", "telecom")
        state_lines = [line.strip() for line in registry.splitlines() if "mCallState" in line]
        telecom_lines = [line.strip() for line in telecom.splitlines() if re.search(r"(?i)state", line)][:40]
        return (
            "telephony.registry:\n  " + "\n  ".join(state_lines or ["(no mCallState lines)"])
            + "\n\ntelecom (lines mentioning state):\n  " + "\n  ".join(telecom_lines or ["(none)"])
        )
