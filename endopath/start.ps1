# EndoAI Startup Script
# This script starts the backend server with the frontend integrated

param(
    [int]$Port = 8080,
    [string]$ServerHost = "0.0.0.0",
    [switch]$NoReload = $false,
    [switch]$UseVenv = $false,
    [string]$PythonExe = ""
)

Write-Host "🚀 EndoAI Startup Script" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

# Get the script directory (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "📁 Project root: $ProjectRoot" -ForegroundColor Cyan

# Determine Python executable preference
$RequestedPython = $null
if ($PythonExe -and $PythonExe.Trim() -ne "") {
    $RequestedPython = $PythonExe
}
elseif ($env:PYTHON_EXE -and $env:PYTHON_EXE.Trim() -ne "") {
    $RequestedPython = $env:PYTHON_EXE
}

# Activate virtual environment if requested
if ($UseVenv) {
    $VenvPath = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
    if (Test-Path $VenvPath) {
        Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
        & $VenvPath
    }
    else {
        Write-Host "❌ Virtual environment not found. Run setup.ps1 -UseVenv first." -ForegroundColor Red
        exit 1
    }
}

# Check if setup has been run
$StaticDir = Join-Path $ProjectRoot "endoserver\static"
if (-not (Test-Path $StaticDir) -or (Get-ChildItem $StaticDir | Measure-Object).Count -eq 0) {
    Write-Host "⚠️ Static files not found. Running setup first..." -ForegroundColor Yellow
    & ".\setup.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Setup failed. Please check the errors above." -ForegroundColor Red
        exit 1
    }
}

# Check if we're in the correct directory structure
$EndoServerDir = Join-Path $ProjectRoot "endoserver"
if (-not (Test-Path $EndoServerDir)) {
    Write-Host "❌ endoserver directory not found!" -ForegroundColor Red
    Write-Host "   Make sure you're running this script from the project root." -ForegroundColor Red
    exit 1
}

# Choose Python command
$PythonCmd = "python"
if ($RequestedPython) {
    $PythonCmd = $RequestedPython
}

# Check for Python and required modules
Write-Host "🐍 Checking Python and dependencies..." -ForegroundColor Yellow

try {
    $PythonVersion = & $PythonCmd --version 2>$null
    if (-not $PythonVersion) {
        throw "Python not found"
    }
    Write-Host "✅ Python: $PythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python not found or not in PATH!" -ForegroundColor Red
    exit 1
}

# Check if FastAPI is installed
try {
    & $PythonCmd -c "import fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Dependencies not installed"
    }
    Write-Host "✅ FastAPI dependencies found" -ForegroundColor Green
}
catch {
    Write-Host "❌ Required dependencies not installed!" -ForegroundColor Red
    Write-Host "   Run: .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Change to endoserver directory
Set-Location $EndoServerDir

# Check for app module
if (-not (Test-Path "app\main.py")) {
    Write-Host "❌ app\main.py not found!" -ForegroundColor Red
    exit 1
}

# Prepare server arguments
$ServerArgs = @(
    "-m", "uvicorn",
    "app.main:app",
    "--host", $ServerHost,
    "--port", $Port.ToString()
)

if (-not $NoReload) {
    $ServerArgs += "--reload"
}

# Display startup information
Write-Host ""
Write-Host "🌐 Starting EndoAI Server..." -ForegroundColor Green
Write-Host "   URL: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "   Host: $ServerHost" -ForegroundColor Cyan
Write-Host "   Reload: $(-not $NoReload)" -ForegroundColor Cyan
Write-Host "   Static files: $StaticDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔑 Default credentials:" -ForegroundColor Yellow
Write-Host "   Username: admin" -ForegroundColor White
Write-Host "   Password: admin" -ForegroundColor White
Write-Host ""
Write-Host "📖 Available endpoints:" -ForegroundColor Yellow
Write-Host "   • Main app: http://localhost:$Port" -ForegroundColor White
Write-Host "   • Login: http://localhost:$Port/login.html" -ForegroundColor White
Write-Host "   • Health: http://localhost:$Port/health" -ForegroundColor White
Write-Host "   • API docs: http://localhost:$Port/docs" -ForegroundColor White
Write-Host ""
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Start the server
try {
    & $PythonCmd @ServerArgs
}
catch {
    Write-Host ""
    Write-Host "❌ Server failed to start!" -ForegroundColor Red
    Write-Host "   Check the error messages above for details." -ForegroundColor Red
    exit 1
}

# Return to project root
Set-Location $ProjectRoot