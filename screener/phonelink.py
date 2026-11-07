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
# e.g. "Call Dialled number + 9 1 9 1 2 1 7 6 3 7 9 1" (not the "Calls" tab).
CALL_BUTTON_NAMES = r"(?i)^(Call|Dial|Make a call|Call now|Start call)( dialled number.*)?$"

_SCRIPT = r"""
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ScreenerDial {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int cmd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    public static void Click(int x, int y) {
        SetCursorPos(x, y);
        mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero);
        mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero);
    }
}
'@
$number = '__NUMBER__'
$pattern = '__PATTERN__'
$mode = '__MODE__'
Start-Process ("tel:" + $number)
$root = [System.Windows.Automation.AutomationElement]::RootElement
$isButton = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Button)
$seen = @{}
$deadline = (Get-Date).AddSeconds(__TIMEOUT__)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 600
    try {
        $windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children,
            [System.Windows.Automation.Condition]::TrueCondition)
    } catch { continue }
    foreach ($w in $windows) {
        try {
            if ($w.Current.Name -notlike '*Phone Link*') { continue }
            foreach ($b in $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, $isButton)) {
                $name = $b.Current.Name
                if ($name) { $seen[$name] = 1 }
                if (-not ($name -match $pattern -and $b.Current.IsEnabled)) { continue }
                if ($mode -eq 'invoke') {
                    $b.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
                    Write-Output ("CLICKED " + $name + " (invoked)")
                    exit 0
                }
                # Bring Phone Link to the front and click the button like a person would.
                $hwnd = [IntPtr]$w.Current.NativeWindowHandle
                [ScreenerDial]::ShowWindow($hwnd, 9) | Out-Null
                [ScreenerDial]::SetForegroundWindow($hwnd) | Out-Null
                Start-Sleep -Milliseconds 400
                $r = $b.Current.BoundingRectangle
                [ScreenerDial]::Click([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
                Write-Output ("CLICKED " + $name + " (mouse click)")
                exit 0
            }
        } catch {}
    }
}
Write-Output ("NOT_FOUND " + (($seen.Keys | Select-Object -First 40) -join ' | '))
exit 1
"""


_TRANSFER_SCRIPT = r"""
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ScreenerMouse {
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    public static void Click(int x, int y) {
        SetCursorPos(x, y);
        mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero);
        mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero);
    }
}
'@
$A = [System.Windows.Automation.AutomationElement]
$Scope = [System.Windows.Automation.TreeScope]
$All = [System.Windows.Automation.Condition]::TrueCondition
$target = '__PATTERN__'
# Never search these (big trees, or our own console whose title mentions "transfer").
$skip = '(?i)chrome|edge|firefox|brave|visual studio|vs code|cmd\.exe|command prompt|powershell|terminal|screener|python|file explorer'

function Click-Element($el) {
    try {
        $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        return 'invoked'
    } catch {}
    try {
        $r = $el.Current.BoundingRectangle
        if (-not $r.IsEmpty -and $r.Width -gt 0) {
            [ScreenerMouse]::Click([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
            return 'mouse click'
        }
    } catch {}
    return $null
}

$windowsSeen = @{}
$namesSeen = @{}
$deadline = (Get-Date).AddSeconds(__TIMEOUT__)
while ((Get-Date) -lt $deadline) {
    try { $windows = $A::RootElement.FindAll($Scope::Children, $All) } catch { $windows = @() }
    foreach ($w in $windows) {
        try {
            $title = $w.Current.Name
            $windowsSeen[("'" + $title + "' (" + $w.Current.ClassName + ")")] = 1
            if ($title -match $skip) { continue }
            $interesting = $title -match '(?i)phone|link|notification|call'
            # Elsewhere, only small popup-sized windows (never e.g. an email mentioning a "transfer").
            $box = $w.Current.BoundingRectangle
            $small = (-not $box.IsEmpty) -and $box.Width -lt 900 -and $box.Height -lt 700
            if (-not ($interesting -or $small)) { continue }
            foreach ($el in $w.FindAll($Scope::Descendants, $All)) {
                try {
                    $name = $el.Current.Name
                    if (-not $name) { continue }
                    if ($interesting -and $namesSeen.Count -lt 60) { $namesSeen[$name] = 1 }
                    if ($name -match $target) {
                        $how = Click-Element $el
                        if ($how) {
                            Write-Output ("CLICKED " + $name + " (" + $how + ")")
                            exit 0
                        }
                    }
                } catch {}
            }
        } catch {}
    }
    Start-Sleep -Milliseconds 400
}
Write-Output ("NOT_FOUND. Windows: " + (($windowsSeen.Keys | Select-Object -First 30) -join ' | '))
Write-Output ("Items in Phone Link/notification windows: " + (($namesSeen.Keys | Select-Object -First 60) -join ' | '))
exit 1
"""

TRANSFER_NAMES = r"(?i)transfer|switch (audio |call )?to (this )?(pc|laptop|computer)|move (the )?call|use (this )?(pc|laptop)"


END_CALL_NAMES = r"(?i)^(end call|hang up|end the call|end)$"


def build_transfer_script(timeout_s: int = 20, pattern: str = TRANSFER_NAMES) -> str:
    return _TRANSFER_SCRIPT.replace("__TIMEOUT__", str(int(timeout_s))).replace("__PATTERN__", pattern)


def _run_transfer(timeout_s: int, pattern: str = TRANSFER_NAMES) -> tuple[bool, str]:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", build_transfer_script(timeout_s, pattern)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s + 20,
        check=False,
    )
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0 and output.startswith("CLICKED"), output


async def click_end_call(timeout_s: int = 4) -> tuple[bool, str]:
    """Clicks Phone Link's End call button (Phone Link calls ignore the phone's end-call key)."""
    if not available():
        return False, "Phone Link only works on Windows"
    try:
        return await asyncio.to_thread(_run_transfer, timeout_s, END_CALL_NAMES)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


async def click_transfer(timeout_s: int = 20) -> tuple[bool, str]:
    """Clicks Phone Link's 'Transfer to laptop' popup when a call is on the phone."""
    if not available():
        return False, "Phone Link only works on Windows"
    try:
        return await asyncio.to_thread(_run_transfer, timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def build_script(number: str, timeout_s: int = 12, mode: str = "click") -> str:
    """mode "click": bring Phone Link forward and mouse-click Call; "invoke": UI Automation press."""
    if not re.fullmatch(r"\+?\d{10,15}", number):
        raise ValueError(f"Invalid number: {number!r}")
    if mode not in ("click", "invoke"):
        raise ValueError(f"Invalid mode: {mode!r}")
    return (
        _SCRIPT.replace("__NUMBER__", number)
        .replace("__PATTERN__", CALL_BUTTON_NAMES)
        .replace("__TIMEOUT__", str(int(timeout_s)))
        .replace("__MODE__", mode)
    )


def available() -> bool:
    return sys.platform == "win32"


def _run(number: str, timeout_s: int, mode: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", build_script(number, timeout_s, mode)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s + 20,
        check=False,
    )
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0 and output.startswith("CLICKED"), output


async def dial(number: str, timeout_s: int = 12, mode: str = "click") -> tuple[bool, str]:
    """Clicks Call in Phone Link. Returns (clicked, details). The caller must check a call started."""
    if not available():
        return False, "Phone Link dialing only works on Windows"
    try:
        return await asyncio.to_thread(_run, number, timeout_s, mode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
