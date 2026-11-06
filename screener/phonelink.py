"""Dial through Windows Phone Link, so the call audio is on the laptop from the start.

Opens a tel: link (handled by Phone Link) and presses its Call button using
Windows UI Automation from PowerShell, so no extra Python packages are needed.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys

# Button names Phone Link may use for placing the call.
CALL_BUTTON_NAMES = r"^(Call|Dial|Make a call|Call now|Start call)$"

_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$number = '__NUMBER__'
$pattern = '__PATTERN__'
Start-Process ("tel:" + $number)
$root = [System.Windows.Automation.AutomationElement]::RootElement
$isButton = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Button)
$seen = @{}
$deadline = (Get-Date).AddSeconds(__TIMEOUT__)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 600
    $windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($w in $windows) {
        if ($w.Current.Name -notlike '*Phone Link*') { continue }
        foreach ($b in $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, $isButton)) {
            $name = $b.Current.Name
            if ($name) { $seen[$name] = 1 }
            if ($name -match $pattern -and $b.Current.IsEnabled) {
                $invoke = $b.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
                $invoke.Invoke()
                Write-Output ("CLICKED " + $name)
                exit 0
            }
        }
    }
}
Write-Output ("NOT_FOUND " + (($seen.Keys | Select-Object -First 40) -join ' | '))
exit 1
"""


def build_script(number: str, timeout_s: int = 12) -> str:
    if not re.fullmatch(r"\+?\d{10,15}", number):
        raise ValueError(f"Invalid number: {number!r}")
    return (
        _SCRIPT.replace("__NUMBER__", number)
        .replace("__PATTERN__", CALL_BUTTON_NAMES)
        .replace("__TIMEOUT__", str(int(timeout_s)))
    )


def available() -> bool:
    return sys.platform == "win32"


def _run(number: str, timeout_s: int) -> tuple[bool, str]:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", build_script(number, timeout_s)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s + 20,
        check=False,
    )
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0 and output.startswith("CLICKED"), output


async def dial(number: str, timeout_s: int = 12) -> tuple[bool, str]:
    """Places the call via Phone Link. Returns (clicked, details)."""
    if not available():
        return False, "Phone Link dialing only works on Windows"
    try:
        return await asyncio.to_thread(_run, number, timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
