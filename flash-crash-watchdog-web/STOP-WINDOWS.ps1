# Stop all Flash Crash Watchdog services (Python sidecar, Binance stream, Next.js)

Write-Host "Stopping Flash Crash Watchdog services..." -ForegroundColor Cyan

$ports = @(8001, 3003, 3000)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            $proc = Get-Process -Id $c.OwningProcess -ErrorAction Stop
            Write-Host "  Killing $($proc.ProcessName) (PID $($proc.Id)) on port $port"
            Stop-Process -Id $proc.Id -Force
        } catch {
            # ignore
        }
    }
}

# SEC-12: scope the name-based fallback to THIS project's processes only — the
# old code killed every python/node/bun on the machine. Match on the watchdog
# path in the command line so innocent processes are never touched.
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'flash-crash-watchdog'
    } |
    ForEach-Object {
        Write-Host "  Stopping $($_.Name) (PID $($_.ProcessId))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host ""
Write-Host "Done. All services stopped." -ForegroundColor Green
