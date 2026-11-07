#!/usr/bin/env bash
# Serialized A/B benchmark for gsv.cpp on a shared CPU box.
#
# The 4-vCPU box this project is measured on is often shared (builds, other agents,
# stray servers). Any timing taken while something else is running is inflated by the
# contention, so this script refuses to start unless the box is quiet, and it runs the
# two binaries strictly one after another (never in parallel).
#
# Usage:
#   scripts/bench_serial.sh --a /path/to/baseline/probe_v2sovits \
#                           --b /path/to/candidate/probe_v2sovits \
#                           --args "--models /path/models --threads 4" [--reps 3]
#
# It prints, per binary: median/min/max of the reported RESULT line plus the AR and
# synth sub-timings, and the A/B ratio. Nothing is written outside --out (default .tmp).
set -uo pipefail

A=""; B=""; ARGS=""; REPS=3; OUT=".tmp/bench_serial"; MAXLOAD=0.60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --a) A="$2"; shift 2 ;;
    --b) B="$2"; shift 2 ;;
    --args) ARGS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --max-load) MAXLOAD="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$A" && -n "$B" ]] || { echo "--a and --b are required" >&2; exit 2; }
mkdir -p "$OUT"

quiet_check() {
  local load
  load=$(awk '{print $1}' /proc/loadavg)
  local busy
  busy=$(ps -eo pcpu,comm --no-headers | awk '$1 > 20 {print $2}' | grep -vcE '^(ps|awk|grep)$' || true)
  # loadavg decays slowly (~1 min) after a heavy job is killed, so it lags reality. Trust the
  # per-process check first: if nothing is above 20% CPU the box is idle right now and the load
  # number is just decay. Only refuse on load when there is also something actually running.
  if [[ "${busy:-0}" -gt 0 ]] || awk -v l="$load" -v m="$MAXLOAD" 'BEGIN{exit !(l > m * 6)}'; then
    echo "REFUSING: load=$load (max $MAXLOAD), busy procs: ${busy:-0}" >&2
    ps -eo pcpu,comm --sort=-pcpu --no-headers | head -5 >&2
    return 1
  fi
  return 0
}

run_bin() {
  local bin="$1" tag="$2" i
  for i in $(seq 1 "$REPS"); do
    # /usr/bin/time reports the CPU percentage this job actually got. On a 4-vCPU box a clean
    # 4-thread run is ~380-400%; anything lower means another process stole cycles and the
    # wall-clock number must be thrown away. This catches contention that the loadavg gate
    # misses (short bursts between samples).
    local tf="$OUT/$tag.$i.time"
    # shellcheck disable=SC2086
    /usr/bin/time -f "%P" -o "$tf" "$bin" $ARGS 2>&1 | tee "$OUT/$tag.$i.log" \
      | grep -E '^\[RESULT\]|AR\(|synthesize:' | sed "s/^/[$tag#$i] /"
    local pct
    pct=$(tr -d '%\n' < "$tf" 2>/dev/null)
    # 365-370% is normal steal on a shared cloud vCPU (hypervisor + neighbours), not contention
    if [[ -n "$pct" && "$pct" -lt 360 ]]; then
      echo "[$tag#$i] DISCARD: job only got ${pct}% CPU (4 threads need ~365%+) - box was busy"
      rm -f "$OUT/$tag.$i.log"
    else
      echo "[$tag#$i] cpu=${pct:-?}% (clean)"
    fi
  done
}

parse() {
  local tag="$1"
  python3 - "$OUT" "$tag" <<'PY'
import glob, re, statistics, sys
out, tag = sys.argv[1], sys.argv[2]
res, ar, syn = [], [], []
for p in sorted(glob.glob(f"{out}/{tag}.*.log")):
    t = open(p, errors="replace").read()
    for m in re.finditer(r"\[RESULT\].*?AR=([\d.]+)ms synth=([\d.]+)ms total=([\d.]+)ms RTF\(AR\+synth\)=([\d.]+)", t):
        ar.append(float(m.group(1))); syn.append(float(m.group(2)))
        res.append((float(m.group(3)), float(m.group(4))))
if not res:
    print(f"{tag}: no [RESULT] lines"); sys.exit(0)
tot = [r[0] for r in res]; rtf = [r[1] for r in res]
print(f"{tag:>10}  AR {statistics.median(ar):8.1f} ms   synth {statistics.median(syn):8.1f} ms   "
      f"total {statistics.median(tot):8.1f} ms   RTF {statistics.median(rtf):6.3f}   (n={len(res)})")
PY
}

echo "== quiet check =="
quiet_check || exit 3
echo "== A: $A"; run_bin "$A" a
echo "== quiet check between runs =="
quiet_check || { echo "contention appeared mid-run; discard results" >&2; exit 3; }
echo "== B: $B"; run_bin "$B" b

echo
parse a
parse b
echo
python3 - "$OUT" <<'PY'
import glob, re, statistics
def tot(tag):
    v = []
    for p in glob.glob(f"{__import__('sys').argv[1]}/{tag}.*.log"):
        for m in re.finditer(r"\[RESULT\].*?total=([\d.]+)ms", open(p, errors="replace").read()):
            v.append(float(m.group(1)))
    return statistics.median(v) if v else None
a, b = tot("a"), tot("b")
if a and b:
    print(f"B/A = {b/a:.3f}  ->  {'faster' if b < a else 'slower'} by {abs(1-b/a)*100:.1f}%")
PY
echo "logs: $OUT"
