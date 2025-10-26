@echo off
REM EndoAI Start Server - Windows Batch Version
echo 🚀 EndoAI Server Startup
echo =======================

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ PowerShell not found! Please use PowerShell for the full functionality.
    echo    You can run: powershell -ExecutionPolicy Bypass -File start.ps1
    pause
    exit /b 1
)

REM Run the PowerShell start script with bypass execution policy
echo 🔄 Starting server with PowerShell...
REM Forward PythonExe if set in environment
if defined PYTHON_EXE (
    powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1" -PythonExe "$env:PYTHON_EXE"
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
)

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Server failed to start! Check the errors above.
    pause
    exit /b 1
)