#!/bin/bash
# Wait for a quiet window on the shared 4-vCPU box, then run the remaining measurements.
# Contention from other agents makes wall-clock AR numbers meaningless (observed 2486 ms/token
# vs 4.4 ms/token clean), so gate on 1-min load average.
GATE=${GATE:-2.5}
MAXWAIT=${MAXWAIT:-1800}
NEW=$HOME/aropt/v2
OLD=$HOME/aropt/v2old
MODELS=$HOME/v2models
FIX=/tmp/v2fix

waited=0
while [ "$waited" -lt "$MAXWAIT" ]; do
  l=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$l" -v g="$GATE" 'BEGIN{exit !(l < g)}'; then break; fi
  sleep 10
  waited=$((waited + 10))
done
echo "QUIET after ${waited}s: load=$(cut -d' ' -f1-3 /proc/loadavg)"

# ---- 1) AR-only, real pipeline shape (S = 148 phones + 214 prompt codes = 362), 179 steps ----
cd "$NEW" || exit 1
for V in q8 q5k q4k q4ka f16; do
  for r in 1 2 3; do
    GSV_AR_PROFILE=1 ./build/bench_ar /tmp/gsv-ar-v2-$V.gguf --threads 4 --steps 179 --phones 148 --prompt 214 2>&1 \
      | grep -E "prof:decode.*(prefill|steps)|ms/token" | tr '\n' '|' | sed "s/^/T $V r$r kv=f32 /"; echo
  done
done
for V in q8 q4k; do
  for r in 1 2 3; do
    GSV_AR_KV_F16=1 GSV_AR_PROFILE=1 ./build/bench_ar /tmp/gsv-ar-v2-$V.gguf --threads 4 --steps 179 --phones 148 --prompt 214 2>&1 \
      | grep -E "prof:decode.*(prefill|steps)|ms/token" | tr '\n' '|' | sed "s/^/T $V r$r kv=f16 /"; echo
  done
done

# ---- 2) F16 KV numerical delta (60 steps, teacher forced, same context) ----
GSV_AR_PROB_DUMP=/tmp/kv32.probs ./build/bench_ar /tmp/gsv-ar-v2-q8.gguf --threads 4 --steps 60 \
    --phones 148 --prompt 214 >/dev/null 2>&1
GSV_AR_KV_F16=1 GSV_AR_PROB_DUMP=/tmp/kv16.probs ./build/bench_ar /tmp/gsv-ar-v2-q8.gguf --threads 4 --steps 60 \
    --phones 148 --prompt 214 >/dev/null 2>&1
python3 $HOME/aropt/ar_prob_compare.py /tmp/kv32.probs /tmp/kv16.probs | sed 's/^/KVQ /'

# ---- 3) full chain, before (v2old) vs after (v2), interleaved, + q4k, + F16 KV ----
run_chain() { # tree variant [kvf16]
  local out env_p=""
  [ -n "$3" ] && env_p="GSV_AR_KV_F16=1"
  out=$(cd "$1" && env $env_p ./build/probe_v2sovits $MODELS/gsv-v2sovits.gguf $FIX --models $MODELS \
        --ref-wav $MODELS/ref8s.wav --threads 4 --ar-file /tmp/gsv-ar-v2-$2.gguf 2>&1)
  echo "$(echo "$out" | grep -o 'AR=[0-9.]*' | head -1 | cut -d= -f2) $(echo "$out" | grep -o 'sem=[0-9]*' | head -1 | cut -d= -f2) $(echo "$out" | grep -o 'synth=[0-9.]*' | head -1 | cut -d= -f2)"
}
for r in 1 2 3; do
  for spec in "OLD q8 -" "NEW q8 -" "NEW q4k -" "NEW q8 f16" "NEW q4k f16" "OLD f16 -" "NEW f16 -"; do
    set -- $spec
    tree=$([ "$1" = OLD ] && echo "$OLD" || echo "$NEW")
    kv=$([ "$3" = f16 ] && echo f16 || echo "")
    read -r ar sem syn <<< "$(run_chain "$tree" "$2" "$kv")"
    echo "AB r$r $1 $2 kv=${3} AR=$ar sem=$sem synth=$syn"
  done
done
echo "DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
