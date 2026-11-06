# One-time setup for the screening agent on Windows.
# Run it by double-clicking setup.bat (or: powershell -ExecutionPolicy Bypass -File setup_windows.ps1).
# Safe to run again: finished steps are skipped.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes downloads much faster in Windows PowerShell
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Work = Join-Path $env:TEMP "screening-agent-setup"
New-Item -ItemType Directory -Force $Work | Out-Null
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$RebootNeeded = $false

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "  OK  $text" -ForegroundColor Green }
function Note($text) { Write-Host "  ..  $text" -ForegroundColor Yellow }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Download($url, $file) {
    $target = Join-Path $Work $file
    Note "Downloading $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $target
    return $target
}
function Has-AudioDevice($pattern) {
    [bool](Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -like $pattern })
}

# ---------------------------------------------------------------------------
Step "1/6  Python"
function Get-PythonExe {
    if (Have "py") {
        $exe = & py -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    $found = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}
$Python = Get-PythonExe
if ($Python) {
    & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { $Python = $null; Note "Python is older than 3.11, installing 3.12" }
}
if (-not $Python) {
    if (-not (Have "winget")) { throw "winget is missing. Install Python 3.12 from https://www.python.org/downloads/ and run this again." }
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    Refresh-Path
    $Python = Get-PythonExe
    if (-not $Python) { throw "Python installed but not found. Close this window and run setup.bat again." }
}
Ok "Python: $Python"

# ---------------------------------------------------------------------------
Step "2/6  ADB (Android platform-tools)"
$AndroidDir = Join-Path $env:LOCALAPPDATA "Android"
$Adb = Join-Path $AndroidDir "platform-tools\adb.exe"
if (-not (Test-Path $Adb)) {
    $zip = Download "https://dl.google.com/android/repository/platform-tools-latest-windows.zip" "platform-tools.zip"
    Expand-Archive $zip -DestinationPath $AndroidDir -Force
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$ptDir = Split-Path $Adb
if ($userPath -notlike "*$ptDir*") { [Environment]::SetEnvironmentVariable("Path", "$userPath;$ptDir", "User") }
Refresh-Path
# Point the agent at this adb.exe.
$settingsPath = Join-Path $Root "config\settings.yaml"
$settings = [IO.File]::ReadAllText($settingsPath)
$settings = [regex]::Replace($settings, "(?m)^(\s*adb_path:)[^#\r\n]*", "`$1 '$Adb'   ")
[IO.File]::WriteAllText($settingsPath, $settings, $Utf8NoBom)
Ok "ADB: $Adb"

# ---------------------------------------------------------------------------
Step "3/6  Virtual audio cables (VB-Audio, free)"
if (Has-AudioDevice "*VB-Audio Virtual Cable*") {
    Ok "VB-CABLE already installed"
} else {
    $zip = Download "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip" "vbcable.zip"
    $dir = Join-Path $Work "vbcable"
    Expand-Archive $zip -DestinationPath $dir -Force
    $setup = if ([Environment]::Is64BitOperatingSystem) { "VBCABLE_Setup_x64.exe" } else { "VBCABLE_Setup.exe" }
    Note "A VB-CABLE window will open: click 'Install Driver', then close it."
    Start-Process (Join-Path $dir $setup) -Verb RunAs -Wait
    $RebootNeeded = $true
}
if (Has-AudioDevice "*Hi-Fi Cable*") {
    Ok "Hi-Fi Cable already installed"
} else {
    $zip = Download "https://download.vb-audio.com/Download_CABLE/HiFiCableAsioBridgeSetup_v1007.zip" "hificable.zip"
    $dir = Join-Path $Work "hificable"
    Expand-Archive $zip -DestinationPath $dir -Force
    Note "A Hi-Fi Cable window will open: click 'Install', then close it."
    Start-Process (Join-Path $dir "HiFiCableAsioBridgeSetup.exe") -Verb RunAs -Wait
    $RebootNeeded = $true
}

# ---------------------------------------------------------------------------
Step "4/6  Python packages"
$Venv = Join-Path $Root ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { & $Python -m venv $Venv }
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed (see above)." }
Ok "Packages installed in .venv"

# ---------------------------------------------------------------------------
Step "5/6  API keys (.env)"
$EnvFile = Join-Path $Root ".env"
$values = [ordered]@{ ANTHROPIC_API_KEY = ""; DEEPGRAM_API_KEY = "" }
if (Test-Path $EnvFile) {
    foreach ($line in [IO.File]::ReadAllLines($EnvFile)) {
        if ($line -match "^\s*([A-Z_]+)\s*=\s*(.*)$" -and $values.Contains($Matches[1])) { $values[$Matches[1]] = $Matches[2].Trim() }
    }
}
$pages = @{
    ANTHROPIC_API_KEY = "https://console.anthropic.com/settings/keys"
    DEEPGRAM_API_KEY  = "https://console.deepgram.com/"
}
foreach ($name in @($values.Keys)) {
    if ($values[$name]) { Ok "$name already set"; continue }
    Note "Opening $($pages[$name]) - create a key there, then paste it here."
    Start-Process $pages[$name]
    $secure = Read-Host -AsSecureString "  Paste $name (input is hidden; press Enter to skip)"
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    $values[$name] = $plain.Trim()
    if ($values[$name]) { Ok "$name saved" } else { Note "$name skipped - add it to .env later" }
}
$content = ($values.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`r`n"
[IO.File]::WriteAllText($EnvFile, $content + "`r`n", $Utf8NoBom)

# ---------------------------------------------------------------------------
Step "6/6  Check"
$candidates = Join-Path $Root "data\candidates.csv"
if (-not (Test-Path $candidates)) {
    Note "Copy your Naukri sheet (with a Phone column) to: $candidates"
}
Push-Location $Root
try { & $VenvPy -m screener check } catch { Note $_ }
Pop-Location

Write-Host ""
Write-Host "Setup done. Remaining steps (see README, section 1):" -ForegroundColor Cyan
if ($RebootNeeded) { Write-Host "  * RESTART the laptop now (audio drivers were installed)." -ForegroundColor Yellow }
Write-Host "  * Phone: enable USB debugging, connect by USB, tap Allow."
Write-Host "  * Phone Link: pair the phone and set up Calls."
Write-Host "  * Route Phone Link audio to the cables, then run: .venv\Scripts\python -m screener audio-test"
Write-Host ""
Read-Host "Press Enter to close"
