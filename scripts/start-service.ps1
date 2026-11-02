# Start the gsv service stack: text sidecar (Python) + gsv_server (C++).
#   .\scripts\start-service.ps1 [-Port 9880] [-SidecarPort 9882] [-Device vulkan] [-Token secret]
param(
    [int]$Port = 9880,
    [int]$SidecarPort = 9882,
    [string]$Device = "vulkan",
    [string]$Token = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$sidecarPy = Join-Path $root ".venv-sidecar\Scripts\python.exe"
if (-not (Test-Path $sidecarPy)) {
    Write-Warning ".venv-sidecar not found; falling back to 'python' (run scripts\setup-sidecar.bat to create it)"
    $sidecarPy = "python"
}
$env:NLTK_DATA = Join-Path $root ".nltk_data"

$serverExe = Join-Path $root "tests\relcpu\gsv_server.exe"
if (-not (Test-Path $serverExe)) {
    throw "gsv_server.exe not found at $serverExe (run scripts\build-server.bat)"
}

Write-Host "[service] starting text sidecar on 127.0.0.1:$SidecarPort ..."
$scArgs = @((Join-Path $root "tools\text_sidecar.py"), "--port", "$SidecarPort")
Start-Process -FilePath $sidecarPy -ArgumentList $scArgs -WorkingDirectory $root -WindowStyle Hidden | Out-Null

$srvArgs = @("--device", $Device, "--port", "$Port", "--sidecar", "127.0.0.1:$SidecarPort")
if ($Token -ne "") { $srvArgs += @("--token", $Token) }
Write-Host "[service] starting gsv_server on http://127.0.0.1:$Port ..."
Start-Process -FilePath $serverExe -ArgumentList $srvArgs -WorkingDirectory $root | Out-Null

for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($h.engine -eq "ready") {
            Write-Host "[service] ready: http://127.0.0.1:$Port  (voice=$($h.voice), sidecar=$($h.sidecar))"
            Write-Host "[service] GAG: point the API base to this URL; stop with scripts\stop-service.ps1"
            exit 0
        }
    } catch { }
}
Write-Warning "[service] server did not become ready in 120 s; check its console/logs"
exit 1
