@echo off
SET script=setup.ps1

REM Check if PowerShell is available
where powershell >nul 2>&1
IF ERRORLEVEL 1 (
    echo PowerShell not found. Please make sure PowerShell is installed.
    pause
    exit /b
)

REM Run the PowerShell script with proper execution policy
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0%script%"
