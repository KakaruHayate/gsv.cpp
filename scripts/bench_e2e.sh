#!/usr/bin/env bash
# End-to-end RTF benchmark helper for gsv_server on a CPU box (e.g. the 4 vCPU shanghai box).
#
# Usage:
#   scripts/bench_e2e.sh --binary build/gsv_server --out /tmp/bench [-- extra server args]
#
# It runs the server once per case, synthesises the same sentence N times, and prints
# wall time, audio duration and RTF per run. Audio duration is read from the WAV header,
# so no ffmpeg/sox is required.
#
# Typical cases once the v2 pipeline lands:
#   scripts/bench_e2e.sh --binary build/gsv_server --out /tmp/bench -- \
#       --models models --device cpu --sidecar 127.0.0.1:9882 --version v2 \
#       --ar models/gsv-ar-v2-q8.gguf --sovits models/gsv-v2sovits.gguf
set -euo pipefail

BIN=""
OUT="/tmp/bench_e2e"
RUNS=3
TEXT="人工智能的发展正在深刻改变我们的生活方式，语音合成技术已经可以生成非常自然的人声。"
REF=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BIN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$BIN" ]] || { echo "--binary is required" >&2; exit 2; }
mkdir -p "$OUT"

run_one() {
  local tag="$1"; shift
  local wav="$OUT/$tag.wav"
  local t0 t1
  t0=$(date +%s.%N)
  "$BIN" "$@" --text "$TEXT" ${REF:+--ref "$REF"} --out "$wav" >"$OUT/$tag.log" 2>&1
  t1=$(date +%s.%N)
  python3 - "$wav" "$t0" "$t1" "$tag" <<'PY'
import sys, wave
wav, t0, t1, tag = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
with wave.open(wav) as w:
    dur = w.getnframes() / float(w.getframerate())
    sr = w.getframerate()
wall = t1 - t0
print(f"{tag:<28} wall={wall:7.3f}s audio={dur:7.3f}s sr={sr} RTF={wall/dur:6.3f} x{dur/wall:5.2f}")
PY
}

echo "binary : $BIN"
echo "runs   : $RUNS   text: ${TEXT:0:24}..."
echo "extra  : ${EXTRA[*]:-<none>}"
echo

for i in $(seq 1 "$RUNS"); do
  run_one "run$i" "${EXTRA[@]}"
done
echo
echo "logs and wavs in $OUT"
