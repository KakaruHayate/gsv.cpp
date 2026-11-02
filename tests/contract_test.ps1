# HTTP contract tests for gsv_server, run against a mock engine (no models/GPU).
#   powershell -ExecutionPolicy Bypass -File tests\contract_test.ps1
# Exits 0 when all checks pass, 1 otherwise.
param(
    [int]$Port = 19880,
    [int]$TokenPort = 19881
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "tests\relcpu\gsv_server.exe"
if (-not (Test-Path $exe)) { throw "missing $exe (run scripts\build-server.bat)" }

$script:fails = 0
$script:checks = 0
function Check($name, $cond, $detail = "") {
    $script:checks++
    if ($cond) { Write-Host ("  ok   " + $name) }
    else { $script:fails++; Write-Host ("  FAIL " + $name + "  " + $detail) }
}
function J($obj) { ($obj | ConvertTo-Json -Depth 6 -Compress) }

function Start-Srv([int]$p, [string[]]$extra) {
    $all = @("--mock-engine", "--mock-sidecar", "--port", "$p") + $extra
    $proc = Start-Process -FilePath $exe -ArgumentList $all -WorkingDirectory $root -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Milliseconds 250
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$p/health" -TimeoutSec 2
            if ($h.engine -eq "ready") { return $proc }
        } catch { }
    }
    throw "server did not start on $p"
}

$srv = $null
$srv2 = $null
try {
    Write-Host "[contract] mock server on 127.0.0.1:$Port"
    $srv = Start-Srv $Port @()
    $base = "http://127.0.0.1:$Port"

    $h = Invoke-RestMethod "$base/health"
    Check "health.engine=ready" ($h.engine -eq "ready")
    Check "health.sidecar=true" ($h.sidecar -eq $true)

    $body = J @{ text = "hello"; text_lang = "zh"; ref_audio_path = "ref.wav";
                 prompt_text = "prompt"; prompt_lang = "zh"; sample_steps = 4 }
    $r = Invoke-WebRequest "$base/tts" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing
    Check "POST /tts 200" ($r.StatusCode -eq 200)
    Check "POST /tts audio/wav" ($r.Headers["Content-Type"] -like "audio/wav*")
    Check "POST /tts wav bytes>44" ($r.RawContentLength -gt 44)
    Check "POST /tts request id" ($r.Headers["X-Request-Id"].Length -gt 0)
    $wav = $r.Content
    Check "wav RIFF" ($wav[0] -eq 0x52 -and $wav[1] -eq 0x49)
    Check "wav 48k" ([BitConverter]::ToInt32($wav, 24) -eq 48000)

    function Expect400($name, $json) {
        try {
            Invoke-WebRequest "$base/tts" -Method Post -ContentType "application/json" -Body (J $json) -UseBasicParsing | Out-Null
            Check $name $false "expected 400"
        } catch {
            $code = $_.Exception.Response.StatusCode.value__
            Check $name ($code -eq 400) "got $code"
        }
    }
    Expect400 "tts missing text -> 400" @{ text_lang = "zh"; ref_audio_path = "r"; prompt_text = "p" }
    Expect400 "tts missing prompt_text -> 400" @{ text = "hi"; text_lang = "zh"; ref_audio_path = "r" }
    Expect400 "tts speed out of range -> 400" @{ text = "hi"; text_lang = "zh"; ref_audio_path = "r"; prompt_text = "p"; speed_factor = 5 }
    Expect400 "tts media_type ogg -> 400" @{ text = "hi"; text_lang = "zh"; ref_audio_path = "r"; prompt_text = "p"; media_type = "ogg" }

    $r = Invoke-WebRequest "$base/tts?text=hello&text_lang=zh&ref_audio_path=r&prompt_text=p" -UseBasicParsing
    Check "GET /tts 200" ($r.StatusCode -eq 200)

    $body = J @{ texts = @("one", "two"); text_lang = "zh"; ref_audio_path = "r";
                 prompt_text = "p"; text_split_method = "cut0"; sample_steps = 4 }
    $r = Invoke-RestMethod "$base/batch" -Method Post -ContentType "application/json" -Body $body
    $job = $r.job_id
    Check "POST /batch job_id" ($null -ne $job -and $job.Length -gt 0)
    $done = $false
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Milliseconds 250
        $j = Invoke-RestMethod "$base/jobs/$job"
        if ($j.state -eq "done") { $done = $true; break }
        if ($j.state -eq "failed") { break }
    }
    Check "job done" $done
    $j = Invoke-RestMethod "$base/jobs/$job"
    Check "job done count 2" ($j.done -eq 2)
    $r0 = Invoke-WebRequest "$base/jobs/$job/result/0" -UseBasicParsing
    Check "job result 200 wav" ($r0.StatusCode -eq 200 -and $r0.RawContentLength -gt 44)
    try { Invoke-WebRequest "$base/jobs/$job/result/9" -UseBasicParsing | Out-Null; Check "job bad index 404" $false }
    catch { Check "job bad index 404" ($_.Exception.Response.StatusCode.value__ -eq 404) }
    try { Invoke-WebRequest "$base/jobs/does-not-exist" -UseBasicParsing | Out-Null; Check "unknown job 404" $false }
    catch { Check "unknown job 404" ($_.Exception.Response.StatusCode.value__ -eq 404) }

    $sse = (& curl.exe -s -N --max-time 10 "$base/jobs/$job/events") -join "`n"
    Check "jobs SSE has data:" ($sse -like "*data: *")

    $m = Invoke-RestMethod "$base/models"
    Check "models has voices" (@($m.voices).Count -ge 1)
    $met = Invoke-RestMethod "$base/metrics"
    Check "metrics sched_batches" ($null -ne $met.sched_batches)
    Check "metrics tts_requests>0" ($met.tts_requests -ge 1)

    try { Invoke-WebRequest "$base/set_gpt_weights?weights_path=nope" -UseBasicParsing | Out-Null; Check "unknown voice 400" $false }
    catch { Check "unknown voice 400" ($_.Exception.Response.StatusCode.value__ -eq 400) }

    try { Invoke-WebRequest "$base/nope" -UseBasicParsing | Out-Null; Check "unknown path 404" $false }
    catch { Check "unknown path 404" ($_.Exception.Response.StatusCode.value__ -eq 404) }

    Write-Host "[contract] token server on 127.0.0.1:$TokenPort"
    $srv2 = Start-Srv $TokenPort @("--token", "t0k")
    $tb = "http://127.0.0.1:$TokenPort"
    $h = Invoke-RestMethod "$tb/health"
    Check "token health open" ($h.engine -eq "ready")
    try { Invoke-WebRequest "$tb/metrics" -UseBasicParsing | Out-Null; Check "no token 401" $false }
    catch { Check "no token 401" ($_.Exception.Response.StatusCode.value__ -eq 401) }
    $r = Invoke-WebRequest "$tb/metrics" -Headers @{ Authorization = "Bearer t0k" } -UseBasicParsing
    Check "bearer token 200" ($r.StatusCode -eq 200)
    try {
        Invoke-WebRequest "$tb/tts" -Method Post -ContentType "application/json" -Body (J @{ text = "hi"; text_lang = "zh"; ref_audio_path = "r"; prompt_text = "p" }) -UseBasicParsing | Out-Null
        Check "tts without token 401" $false
    } catch { Check "tts without token 401" ($_.Exception.Response.StatusCode.value__ -eq 401) }
}
finally {
    if ($srv2 -and -not $srv2.HasExited) { $srv2 | Stop-Process -Force }
    if ($srv -and -not $srv.HasExited) { $srv | Stop-Process -Force }
}

Write-Host ""
if ($script:fails -eq 0) { Write-Host "CONTRACT_OK ($script:checks checks)"; exit 0 }
else { Write-Host "CONTRACT_FAIL ($script:fails/$script:checks failed)"; exit 1 }
