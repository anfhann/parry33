@echo off
REM parry33 launcher. Runs the CLI out of the project venv so you never have to
REM activate it or remember the interpreter path. Works from cmd and PowerShell.
REM
REM   parry33 doctor
REM   parry33 bench capture --delay 20 --seconds 30
setlocal
set "HERE=%~dp0"
if not exist "%HERE%.venv\Scripts\python.exe" (
  echo [parry33] venv not found at %HERE%.venv
  echo [parry33] run:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -e ".[capture,dev]"
  exit /b 1
)
"%HERE%.venv\Scripts\python.exe" -m parry33 %*
