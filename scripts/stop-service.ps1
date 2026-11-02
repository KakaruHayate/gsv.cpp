# Stop the gsv service stack (gsv_server via /control, sidecar via process match).
param(
    [int]$Port = 9880
)
$ErrorActionPreference = "SilentlyContinue"
try { Invoke-RestMethod "http://127.0.0.1:$Port/control?command=exit" -TimeoutSec 3 | Out-Null } catch { }
Start-Sleep -Seconds 1
Get-Process gsv_server -ErrorAction SilentlyContinue | Stop-Process -Force
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*text_sidecar.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Host "[service] stopped"
