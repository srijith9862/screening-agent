@echo off
rem Shortcut: "screener practice", "screener call", ... without activating the venv.
pushd "%~dp0"
".venv\Scripts\python.exe" -m screener %*
popd
