# End-to-end A/B of the DiT algorithm switches through the HTTP frontend.
# Compares WAV outputs (and timing) of the same request with different dit_* params.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads .ps1 as ANSI).
$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:9880"
$out = "tests\relcpu\e2e"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$ref = "models\ref_test.wav"
if (-not (Test-Path $ref)) { $ref = (Get-ChildItem models -Filter *.wav | Select-Object -First 1).FullName }
$text = (Get-Content "tests\relcpu\e2e_text.txt" -Encoding UTF8 -Raw).Trim()
$ptext = (Get-Content "tests\relcpu\e2e_prompt.txt" -Encoding UTF8 -Raw).Trim()
Write-Host "ref = $ref"

function Tts($name, $extra) {
    $body = @{
        text = $text
        text_lang = "zh"
        ref_audio_path = $ref
        prompt_text = $ptext
        prompt_lang = "zh"
        seed = 12345
    }
    foreach ($k in $extra.Keys) { $body[$k] = $extra[$k] }
    $json = $body | ConvertTo-Json -Compress
    # warm-up run (pipeline compile / voice switch), then the measured run
    $null = Invoke-WebRequest "$base/tts" -Method Post -ContentType "application/json" -Body $json -UseBasicParsing -TimeoutSec 900
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-WebRequest "$base/tts" -Method Post -ContentType "application/json" -Body $json -UseBasicParsing -TimeoutSec 900
    $sw.Stop()
    [System.IO.File]::WriteAllBytes("$out\$name.wav", $r.Content)
    # parse WAV chunks properly (there may be a LIST chunk before data)
    $b = $r.Content
    $pos = 12; $n = 0
    while ($pos + 8 -le $b.Length) {
        $id = [System.Text.Encoding]::ASCII.GetString($b, $pos, 4)
        $sz = [BitConverter]::ToInt32($b, $pos + 4)
        if ($id -eq "data") { $n = $sz; break }
        $pos += 8 + $sz + ($sz % 2)
    }
    $samples = [int16[]]::new([int]($n / 2))
    [System.Buffer]::BlockCopy($b, $pos + 8, $samples, 0, $n)
    [pscustomobject]@{ name = $name; ms = $sw.ElapsedMilliseconds; samples = $samples }
}

function Cmp($a, $b) {
    $n = [Math]::Min($a.samples.Length, $b.samples.Length)
    $mx = 0.0; $sa = 0.0; $na = 0.0; $num = 0.0
    for ($i = 0; $i -lt $n; $i++) {
        $x = $a.samples[$i]; $y = $b.samples[$i]
        $d = [Math]::Abs($x - $y); if ($d -gt $mx) { $mx = $d }
        $sa += $d; $na += [Math]::Abs($x); $num += $x * $y
    }
    $ma = 0.0; $mb = 0.0
    for ($i = 0; $i -lt $n; $i++) { $ma += $a.samples[$i]; $mb += $b.samples[$i] }
    $ma /= $n; $mb /= $n
    $va = 0.0; $vb = 0.0; $cov = 0.0
    for ($i = 0; $i -lt $n; $i++) {
        $va += ($a.samples[$i] - $ma) * ($a.samples[$i] - $ma)
        $vb += ($b.samples[$i] - $mb) * ($b.samples[$i] - $mb)
        $cov += ($a.samples[$i] - $ma) * ($b.samples[$i] - $mb)
    }
    $corr = $cov / ([Math]::Sqrt($va * $vb) + 1e-30)
    [pscustomobject]@{ max = $mx; rel = $sa / ($na + 1e-30); corr = $corr }
}

$voices = @("turbo-q8", "v5dev")
foreach ($v in $voices) {
    Write-Host "`n================ voice=$v ================"
    # baseline: 显式全关 (与改动前逐位一致的口径)
    $baseRun = Tts "$v-base" @{ voice = $v; dit_cfg_off_t = 0; dit_uncond_reuse = 0 }
    Write-Host ("  base(all-off) {0,6} ms  ({1} samples)" -f $baseRun.ms, $baseRun.samples.Length)
    # 生产默认 (不传 dit_* -> 服务端默认 cfg_off=0.7)
    $cases = [ordered]@{
        "default(no-fields)" = @{ voice = $v }
        "cfg.7(default)"   = @{ voice = $v; dit_cfg_off_t = 0.7 }
        "cfg.7+u1"         = @{ voice = $v; dit_cfg_off_t = 0.7; dit_uncond_reuse = 1 }
        "cfg.6+u1"         = @{ voice = $v; dit_cfg_off_t = 0.6; dit_uncond_reuse = 1 }
        "cfg.7+u2"         = @{ voice = $v; dit_cfg_off_t = 0.7; dit_uncond_reuse = 2 }
        "timeout_ms=1000"  = @{ voice = $v; timeout_ms = 1000 }
    }
    foreach ($k in $cases.Keys) {
        try {
            $r = Tts "$v-$k" $cases[$k]
        } catch {
            # timeout_ms 用例预期被拒: 服务端返回 504 {"code":"timeout"}
            $code = $null
            if ($_.Exception.Response) { $code = $_.Exception.Response.StatusCode.value__ }
            $body = ""
            try {
                $sr = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $body = $sr.ReadToEnd(); $sr.Close()
            } catch { }
            Write-Host ("  {0,-20} -> HTTP {1} {2}" -f $k, $code, $body)
            continue
        }
        $c = Cmp $baseRun $r
        Write-Host ("  {0,-20} {1,6} ms  ({2,5:0.00}x)  max|d|={3,8:0} relL1={4:0.000000} corr={5:0.00000}" -f `
            $k, $r.ms, ($baseRun.ms / [Math]::Max($r.ms,1)), $c.max, $c.rel, $c.corr)
    }
}
Write-Host "`nDONE -> $out"
