# EndoAI Setup Script
# This script sets up the entire development environment

param(
    [switch]$UseVenv = $false,
    [switch]$Force = $false,
    [string]$PythonExe = ""
)

Write-Host "🚀 EndoAI Setup Script" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green

# Get the script directory (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "📁 Project root: $ProjectRoot" -ForegroundColor Cyan

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) { return $true }
    }
    catch {
        return $false
    }
}

# Prefer explicit Python if provided (parameter or env)
$RequestedPython = $null
if ($PythonExe -and $PythonExe.Trim() -ne "") {
    $RequestedPython = $PythonExe
}
elseif ($env:PYTHON_EXE -and $env:PYTHON_EXE.Trim() -ne "") {
    $RequestedPython = $env:PYTHON_EXE
}

# Check Python installation
Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
if ($RequestedPython) {
    if (-not (Test-Path $RequestedPython)) {
        Write-Host "❌ Provided Python executable not found: $RequestedPython" -ForegroundColor Red
        exit 1
    }
    $PythonVersion = & $RequestedPython --version
}
else {
    if (-not (Test-Command "python")) {
        Write-Host "❌ Python not found! Please install Python 3.11+ and add it to PATH." -ForegroundColor Red
        exit 1
    }
    $PythonVersion = python --version
}
Write-Host "✅ Found: $PythonVersion" -ForegroundColor Green

# Setup Python environment
if ($UseVenv) {
    Write-Host "🔧 Setting up virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv" -and -not $Force) {
        Write-Host "📦 Virtual environment already exists. Use -Force to recreate." -ForegroundColor Yellow
        Write-Host "🔄 Activating existing virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
    }
    else {
        if ($Force -and (Test-Path "venv")) {
            Write-Host "🗑️ Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force "venv"
        }
        
        Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
        if ($RequestedPython) {
            & $RequestedPython -m venv venv
        }
        else {
            python -m venv venv
        }
        
        Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
    }
}
else {
    Write-Host "🔄 Using system Python (skipping virtual environment)" -ForegroundColor Yellow
}

# Install backend dependencies
Write-Host "📥 Installing backend dependencies..." -ForegroundColor Yellow
Set-Location "endoserver"

if (Test-Path "requirements.txt") {
    if ($UseVenv) {
        # Use venv's python/pip
        & "$ProjectRoot\venv\Scripts\python.exe" -m pip install --upgrade pip
        & "$ProjectRoot\venv\Scripts\pip.exe" install -r requirements.txt
    }
    else {
        if ($RequestedPython) {
            & $RequestedPython -m pip install --upgrade pip
            & $RequestedPython -m pip install -r requirements.txt
        }
        else {
            python -m pip install --upgrade pip
            pip install -r requirements.txt
        }
    }
    Write-Host "✅ Backend dependencies installed" -ForegroundColor Green
}
else {
    Write-Host "❌ requirements.txt not found in endoserver directory!" -ForegroundColor Red
    exit 1
}

# Copy frontend files to static directory
Write-Host "📄 Setting up frontend files..." -ForegroundColor Yellow
Set-Location $ProjectRoot

$FrontendSource = Join-Path $ProjectRoot "endoui"
$StaticDest = Join-Path $ProjectRoot "endoserver\static"

if (Test-Path $FrontendSource) {
    # Create static directory if it doesn't exist
    if (-not (Test-Path $StaticDest)) {
        New-Item -ItemType Directory -Path $StaticDest -Force | Out-Null
    }
    
    # Copy files using robocopy
    Write-Host "📋 Copying frontend files to static directory..." -ForegroundColor Yellow
    $Result = robocopy "$FrontendSource" "$StaticDest" /E /XF "*.ps1" "*.md" "start_server.py" /NFL /NDL /NJH /NJS
    
    if ($LASTEXITCODE -le 1) {
        # robocopy returns 0 or 1 for success
        Write-Host "✅ Frontend files copied successfully" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ Some files may not have been copied correctly" -ForegroundColor Yellow
    }
}
else {
    Write-Host "❌ Frontend source directory not found: $FrontendSource" -ForegroundColor Red
    exit 1
}

# Create database directory
$DbDir = Join-Path $ProjectRoot "endoserver\app"
if (-not (Test-Path $DbDir)) {
    New-Item -ItemType Directory -Path $DbDir -Force | Out-Null
}

# Check for model files
$ModelDir = Join-Path $ProjectRoot "program1-trainer\models"
if (Test-Path $ModelDir) {
    $ModelFiles = Get-ChildItem -Path $ModelDir -Filter "*.h5" | Measure-Object
    if ($ModelFiles.Count -gt 0) {
        Write-Host "✅ Found $($ModelFiles.Count) model file(s) in $ModelDir" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ No model files (.h5) found in $ModelDir" -ForegroundColor Yellow
        Write-Host "   Train a model first or place model files in this directory" -ForegroundColor Yellow
    }
}
else {
    Write-Host "⚠️ Model directory not found: $ModelDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To start the application:" -ForegroundColor Cyan
Write-Host "   .\start.ps1" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Application will be available at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "🔑 Default credentials:" -ForegroundColor Cyan
Write-Host "   Username: admin" -ForegroundColor White
Write-Host "   Password: admin" -ForegroundColor White
Write-Host ""

Set-Location $ProjectRoot