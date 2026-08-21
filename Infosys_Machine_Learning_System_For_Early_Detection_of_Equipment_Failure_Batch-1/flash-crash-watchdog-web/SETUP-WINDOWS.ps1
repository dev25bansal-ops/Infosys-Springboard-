# First-time setup - run ONCE before the first START-WINDOWS.ps1
# Installs: npm deps, binance-stream deps, Python venv + requirements, SQLite db

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step($m) { Write-Host ""; Write-Host "[step] $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "[ok]   $m" -ForegroundColor Green }
function Write-Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }

$hasBun = $null -ne (Get-Command bun -ErrorAction SilentlyContinue)
$hasPy  = $null -ne (Get-Command python -ErrorAction SilentlyContinue)

Write-Host "================================================" -ForegroundColor White
Write-Host " Flash Crash Watchdog - First-Time Setup"         -ForegroundColor White
Write-Host "================================================" -ForegroundColor White

# 1. Root deps
# Set env vars to make Prisma install more robust against network flakes:
#   - PRISMA_SKIP_POSTINSTALL_GENERATE: don't download 50MB engine during install
#   - PRISMA_ENGINES_MIRROR: use a more reliable CDN
$env:PRISMA_SKIP_POSTINSTALL_GENERATE = "true"
$env:PRISMA_ENGINES_MIRROR = "https://binaries.prisma.sh"

Write-Step "Installing root dependencies..."
$installRetries = 0
$maxRetries = 3
$installOk = $false
while ($installRetries -lt $maxRetries -and -not $installOk) {
    $installRetries++
    if ($hasBun) {
        bun install
    } else {
        npm install --legacy-peer-deps
    }
    if ($LASTEXITCODE -eq 0) {
        $installOk = $true
    } else {
        Write-Warn "Install attempt $installRetries failed. Retrying in 5s..."
        Start-Sleep -Seconds 5
    }
}
if (-not $installOk) {
    throw "Root install failed after $maxRetries attempts. Try running 'bun install' manually, or set `$env:PRISMA_ENGINES_MIRROR = 'https://registry.npmmirror.com/-/binary/prisma' and retry."
}
Write-Ok "Root deps installed"

# 2. Stream service deps
Write-Step "Installing binance-stream dependencies..."
Push-Location "$ProjectRoot\mini-services\binance-stream"
if ($hasBun) {
    bun install
} else {
    npm install --legacy-peer-deps
}
Pop-Location
if ($LASTEXITCODE -ne 0) {
    throw "Stream install failed"
}
Write-Ok "Stream deps installed"

# 3. Python venv + requirements
if ($hasPy) {
    Write-Step "Creating Python venv + installing ML deps..."
    Push-Location "$ProjectRoot\ml-inference"
    python -m venv venv
    & "$ProjectRoot\ml-inference\venv\Scripts\python.exe" -m pip install --upgrade pip

    # Detect NVIDIA GPU and install CUDA PyTorch if available (much faster inference)
    $hasNvidia = $false
    try {
        $gpuInfo = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match "NVIDIA" }
        if ($gpuInfo) {
            $hasNvidia = $true
            Write-Ok "NVIDIA GPU detected: $($gpuInfo.Name)"
            Write-Step "Installing PyTorch with CUDA support (this is a large download ~2.5GB)..."
            & "$ProjectRoot\ml-inference\venv\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cu121
        }
    } catch {
        # GPU detection failed, continue with CPU-only
    }

    if (-not $hasNvidia) {
        Write-Host "[info] No NVIDIA GPU detected - using CPU-only PyTorch (slower but works)"
    }

    # Install the rest of the requirements
    & "$ProjectRoot\ml-inference\venv\Scripts\pip.exe" install -r requirements.txt
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python setup failed - you can still run START-WINDOWS.ps1 with -SkipPython"
    } else {
        if ($hasNvidia) {
            Write-Ok "Python venv + ML deps installed (GPU mode)"
        } else {
            Write-Ok "Python venv + ML deps installed (CPU mode)"
        }
    }
} else {
    Write-Warn "Python not found - skipping ML sidecar."
    Write-Warn "Install Python 3.10+ from https://www.python.org/downloads/ to use the trained TCN."
}

# 4. Database
# Ensure prisma CLI is installed locally (npm v10+ doesn't auto-add .bin to PATH)
# IMPORTANT: pin to prisma@^6 — prisma@7 has breaking schema changes
if (-not (Test-Path "$ProjectRoot\node_modules\.bin\prisma.cmd") -and -not (Test-Path "$ProjectRoot\node_modules\.bin\prisma")) {
    Write-Step "Installing prisma CLI locally (pinned to v6)..."
    if ($hasBun) {
        bun add -d prisma@^6.11.1
    } else {
        npm install --legacy-peer-deps -d prisma@^6.11.1
    }
}

Write-Step "Generating Prisma Client and initializing SQLite database..."
if ($hasBun) {
    bun run db:generate
    bun run db:push
} else {
    npm run db:generate
    npm run db:push
}
if ($LASTEXITCODE -ne 0) {
    throw "Database setup failed"
}
Write-Ok "Prisma Client generated + database initialized"

Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host " Setup complete!"                                  -ForegroundColor Green
Write-Host "================================================" -ForegroundColor White
Write-Host ""
Write-Host "Next step:  .\START-WINDOWS.ps1"
Write-Host ""
