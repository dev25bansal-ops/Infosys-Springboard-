# Windows service supervision (MLOPS-03 / MLOPS-04)

Always-on deployment for the three services on a Windows host, using
[WinSW](https://github.com/winsw/winsw) so the stack restarts automatically on
crash and survives reboot.

## Install

1. Download `WinSW-x64.exe` from the WinSW releases and place it next to the
   three `.xml` configs here (or reference the full paths).
2. Rename it per-service, or run with an explicit config:
   ```
   WinSW-x64.exe install  D:\flash-crash-watchdog\flash-crash-watchdog-web\deploy\winsw\fcw-sidecar.xml
   WinSW-x64.exe install  D:\flash-crash-watchdog\flash-crash-watchdog-web\deploy\winsw\fcw-stream.xml
   WinSW-x64.exe install  D:\flash-crash-watchdog\flash-crash-watchdog-web\deploy\winsw\fcw-dashboard.xml
   ```
3. Start them (dependency order is already encoded: dashboard → stream → sidecar):
   ```
   WinSW-x64.exe start fcw-sidecar
   WinSW-x64.exe start fcw-stream
   WinSW-x64.exe start fcw-dashboard
   ```

## Health / staleness (MLOPS-04)

- Sidecar: `http://127.0.0.1:8000/health` → `{model_loaded, window_size, ...}`
- Stream:  `http://127.0.0.1:3005/health` → `{connected, lastTickAt, stalenessMs, stale}`
  — `stale:true` means no Binance ticks for 15s; the dashboard shows a "Feed
  stale" alarm instead of silently going dark. (Health is on its own port
  because the stream's socket.io owns :3003.)
- Dashboard: `http://localhost:3000/api/auth/me` (200/401 = alive)

## Update

```
WinSW-x64.exe stop  fcw-dashboard
WinSW-x64.exe stop  fcw-stream
WinSW-x64.exe stop  fcw-sidecar
# deploy the new build (next build, new venv deps, etc.)
WinSW-x64.exe start fcw-sidecar
WinSW-x64.exe start fcw-stream
WinSW-x64.exe start fcw-dashboard
```

## Uninstall

```
WinSW-x64.exe uninstall fcw-dashboard
WinSW-x64.exe uninstall fcw-stream
WinSW-x64.exe uninstall fcw-sidecar
```

> Note: paths in the `.xml` files are hardcoded to `D:\flash-crash-watchdog\...`.
> Adjust them if the repo lives elsewhere. The dev launcher
> (`START-WINDOWS.ps1`) remains the right choice for interactive development;
> these services are for always-on operation.
