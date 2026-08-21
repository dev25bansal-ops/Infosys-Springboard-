# Flash Crash Watchdog - Windows PowerShell launcher
#
# Usage:
#   1. Right-click this file -> Run with PowerShell
#   2. Or from PowerShell:  .\START-WINDOWS.ps1
#
# What it does:
#   - Opens 3 PowerShell windows:
#       [1] Python TCN inference sidecar (port 8001)
#       [2] Binance WebSocket stream + Socket.io (port 3003)
#       [3] Next.js dashboard (port 3000)
#   - Then opens http://localhost:3000 in your default browser
#
# Prerequisites:
#   - Node.js 18+  (https://nodejs.org/)
#   - Python 3.10+ (https://www.python.org/downloads/)  [optional - skip with -SkipPython]
#   - Optional: Bun (faster) - https://bun.sh/
#
# First-time setup (run once before the first start):
#   .\SETUP-WINDOWS.ps1

param(
    [switch]$SkipBrowser,
    [switch]$SkipPython,
    [string]$PyExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step($msg) { Write-Host ""; Write-Host "[step] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ok]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[err]  $msg" -ForegroundColor Red }

Write-Host "================================================" -ForegroundColor White
Write-Host " Flash Crash Watchdog - Windows Launcher"         -ForegroundColor White
Write-Host "================================================" -ForegroundColor White

# --- Sanity checks ---
Write-Step "Checking prerequisites..."

$hasNode = $null -ne (Get-Command node -ErrorAction SilentlyContinue)
$hasNpm  = $null -ne (Get-Command npm  -ErrorAction SilentlyContinue)
$hasBun  = $null -ne (Get-Command bun  -ErrorAction SilentlyContinue)
$hasPy   = $null -ne (Get-Command $PyExe -ErrorAction SilentlyContinue)

if (-not $hasNode) {
    Write-Err "Node.js not found. Install from https://nodejs.org/"
    exit 1
}
if (-not $hasNpm) {
    Write-Err "npm not found (should come with Node.js)."
    exit 1
}
$nodeVer = & node --version 2>&1
Write-Ok "Node.js found: $nodeVer"

if (-not $hasPy -and -not $SkipPython) {
    Write-Warn "Python ('$PyExe') not found."
    Write-Warn "  -> Install Python 3.10+ from https://www.python.org/downloads/"
    Write-Warn "  -> OR run with -SkipPython to use heuristic-only mode"
    $SkipPython = $true
}
if ($hasPy -and -not $SkipPython) {
    $pyVer = & $PyExe --version 2>&1
    Write-Ok "Python found: $pyVer"
}

if ($hasBun) {
    $bunVer = & bun --version 2>&1
    Write-Ok "Bun found: $bunVer - will use Bun for faster dev"
} else {
    Write-Host "[info] Bun not installed - will use npm (slower but works)"
}

# --- Verify dependencies are installed ---
if (-not (Test-Path "$ProjectRoot\node_modules")) {
    Write-Step "Installing root npm dependencies (first run only)..."
    if ($hasBun) {
        bun install
    } else {
        npm install --legacy-peer-deps
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Root install failed"
        exit 1
    }
    Write-Ok "Root dependencies installed"
}

$streamPath = "$ProjectRoot\mini-services\binance-stream"
if (-not (Test-Path "$streamPath\node_modules")) {
    Write-Step "Installing binance-stream dependencies..."
    Push-Location $streamPath
    if ($hasBun) {
        bun install
    } else {
        npm install --legacy-peer-deps
    }
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Err "binance-stream install failed"
        exit 1
    }
    Write-Ok "Stream service dependencies installed"
}

if (-not $SkipPython) {
    $pyPath = "$ProjectRoot\ml-inference"
    if (-not (Test-Path "$pyPath\venv")) {
        Write-Step "Creating Python venv + installing requirements..."
        Push-Location $pyPath
        & $PyExe -m venv venv
        & "$pyPath\venv\Scripts\python.exe" -m pip install --upgrade pip
        & "$pyPath\venv\Scripts\pip.exe" install -r requirements.txt
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Python install failed - falling back to heuristic mode"
            $SkipPython = $true
        } else {
            Write-Ok "Python venv + dependencies installed"
        }
    }
}

# --- Initialize the SQLite database + generate Prisma Client ---
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

Write-Step "Generating Prisma Client + initializing SQLite database..."
if ($hasBun) {
    bun run db:generate
    bun run db:push
} else {
    npm run db:generate
    npm run db:push
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn "db setup had issues - dashboard may not persist alerts"
} else {
    Write-Ok "Database ready + Prisma Client generated"
}

# --- SEC-1: ensure a strong SESSION_SECRET (signed sessions) ---
# Prefer the environment; else a strong value already persisted in .env; else
# generate + persist one. NEVER falls back to a public constant.
Write-Step "Ensuring a strong SESSION_SECRET..."
$envFile = "$ProjectRoot\.env"
$envSecret = $env:SESSION_SECRET
$envSecretOk = $envSecret -and $envSecret.Length -ge 32 -and `
    $envSecret -notmatch '^(dev-insecure-change-me|change-me-to-a-long-random-string)$'
$dotenvSecret = $null
if (Test-Path $envFile) {
    $dotenvSecret = (Get-Content $envFile | Where-Object { $_ -match '^SESSION_SECRET=' } | Select-Object -First 1)
}
if ($envSecretOk) {
    Write-Ok "Using SESSION_SECRET from environment"
} elseif ($dotenvSecret -and ($dotenvSecret.Trim() -notmatch '^SESSION_SECRET="?(dev-insecure-change-me|change-me-to-a-long-random-string)"?$') -and ($dotenvSecret -match '[A-Za-z0-9_\-]{32,}')) {
    $env:SESSION_SECRET = ($dotenvSecret -replace '^SESSION_SECRET="?([^"]*)"?$', '$1').Trim()
    Write-Ok "Using SESSION_SECRET from .env"
} else {
    $newSecret = -join ((48..57) + (97..122) + (65..90) | Get-Random -Count 48 | ForEach-Object { [char]$_ })
    $env:SESSION_SECRET = $newSecret
    if (Test-Path $envFile) {
        $content = Get-Content $envFile
        $updated = $content | ForEach-Object {
            if ($_ -match '^SESSION_SECRET=') { "SESSION_SECRET=`"$newSecret`"" } else { $_ }
        }
        if ($updated -notmatch '^SESSION_SECRET=') { $updated += "SESSION_SECRET=`"$newSecret`"" }
        Set-Content $envFile $updated
    } else {
        Set-Content $envFile "SESSION_SECRET=`"$newSecret`""
    }
    Write-Ok "SESSION_SECRET generated and persisted to .env"
}

# --- Launch the three services ---

# [1] Python TCN sidecar
if (-not $SkipPython) {
    Write-Step "Starting Python TCN sidecar (port 8001)..."
    $cmd = "Set-Location '$ProjectRoot\ml-inference'; "
    $cmd += "Write-Host '[1] Python TCN sidecar - port 8001' -ForegroundColor Cyan; "
    $cmd += ".\venv\Scripts\python.exe -m uvicorn server:app --port 8001 --reload"
    $pyProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru
    Write-Ok "Python sidecar started (PID $($pyProc.Id))"
    Start-Sleep -Seconds 3
} else {
    Write-Warn "Skipping Python sidecar - dashboard will use heuristic scorer"
}

# [2] Binance stream
Write-Step "Starting Binance stream + Socket.io (port 3003)..."
$cmd2 = "Set-Location '$streamPath'; "
$cmd2 += "Write-Host '[2] Binance stream - port 3003' -ForegroundColor Cyan; "
if ($hasBun) {
    $cmd2 += "bun run dev"
} else {
    $cmd2 += "npm run dev"
}
$streamProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd2 -PassThru
Write-Ok "Stream service started (PID $($streamProc.Id))"

# [2b] Crash replay service (port 3004) — powers the Replay page.
Write-Step "Starting Crash replay service (port 3004)..."
$cmdReplay = "Set-Location '$streamPath'; "
$cmdReplay += "Write-Host '[2b] Crash replay - port 3004' -ForegroundColor Cyan; "
if ($hasBun) {
    $cmdReplay += "bun run dev:replay"
} else {
    $cmdReplay += "npm run dev:replay"
}
$replayProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmdReplay -PassThru
Write-Ok "Replay service started (PID $($replayProc.Id))"

# [3] Next.js dashboard
Write-Step "Starting Next.js dashboard (port 3000)..."
$cmd3 = "Set-Location '$ProjectRoot'; "
$cmd3 += "Write-Host '[3] Next.js dashboard - port 3000' -ForegroundColor Cyan; "
if ($hasBun) {
    $cmd3 += "bun run dev"
} else {
    $cmd3 += "npm run dev"
}
$webProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd3 -PassThru
Write-Ok "Dashboard started (PID $($webProc.Id))"

# --- Wait for Next.js to be ready, then open browser ---
if (-not $SkipBrowser) {
    Write-Step "Waiting for Next.js to be ready..."
    $maxWait = 60
    $waited = 0
    while ($waited -lt $maxWait) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:3000/api/auth/me" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200 -or $r.StatusCode -eq 401) {
                break
            }
        } catch {
            # ignore - keep waiting
        }
        Start-Sleep -Seconds 2
        $waited += 2
        Write-Host "  ...waiting ($waited s)" -ForegroundColor Gray
    }
    if ($waited -lt $maxWait) {
        Write-Ok "Dashboard is up - opening browser"
        Start-Process "http://localhost:3000"
    } else {
        Write-Warn "Dashboard didn't respond in 60s - open http://localhost:3000 manually"
    }
}

# --- Summary ---
Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host " All services launched"                            -ForegroundColor Green
Write-Host "================================================" -ForegroundColor White
Write-Host ""
Write-Host "  Dashboard:        http://localhost:3000"          -ForegroundColor Cyan
Write-Host "  Stream service:   http://localhost:3003"          -ForegroundColor Cyan
if (-not $SkipPython) {
    Write-Host "  TCN health:       http://localhost:8001/health" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "  To stop all services: close the 3 PowerShell windows that opened."
Write-Host "  Or run:  .\STOP-WINDOWS.ps1"
Write-Host ""

if (-not $SkipPython) {
    Write-Host "  Score source badge on the dashboard:" -ForegroundColor Gray
    Write-Host "    TCN (green)  = real PyTorch model is scoring" -ForegroundColor Gray
    Write-Host "    HEURISTIC    = Python unavailable - fallback scorer" -ForegroundColor Gray
    Write-Host "    WARMUP       = Python is buffering ticks (first ~5s)" -ForegroundColor Gray
} else {
    Write-Host "  Running in HEURISTIC-ONLY mode (no Python)" -ForegroundColor Yellow
    Write-Host "  Alerts still fire - but score is from the TS fallback, not the trained TCN" -ForegroundColor Yellow
}
Write-Host ""
