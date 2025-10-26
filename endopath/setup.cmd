@echo off
REM EndoAI Quick Start - Windows Batch Version
echo 🚀 EndoAI Quick Start
echo ==================

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ PowerShell not found! Please use PowerShell for the full setup.
    echo    You can run: powershell -ExecutionPolicy Bypass -File setup.ps1
    pause
    exit /b 1
)

REM Run the PowerShell setup script with bypass execution policy
echo 🔄 Running PowerShell setup script...
REM Forward PythonExe if set in environment
if defined PYTHON_EXE (
    powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -PythonExe "$env:PYTHON_EXE"
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Setup failed! Check the errors above.
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed!
echo.
echo 🚀 To start the server, run:
echo    start.cmd
echo.
echo    Or use PowerShell:
echo    powershell -ExecutionPolicy Bypass -File start.ps1
echo.
pause