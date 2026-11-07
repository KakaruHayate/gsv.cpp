#!/bin/bash
# A/B interleaved measurement: before (src/gsv_ar.cpp @ HEAD) vs after (KV-concat removal).
# Interleaving matters: the 4-vCPU box is shared with other agents and per-run variance is large.
# usage: ab.sh <rounds>
ROUNDS=${ROUNDS:-3}
NEW=$HOME/aropt/v2
OLD=$HOME/aropt/v2old
FIX=/tmp/v2fix
MODELS=$HOME/v2models

run_chain() { # $1 tree, $2 variant
  local out
  out=$(cd "$1" && ./build/probe_v2sovits $MODELS/gsv-v2sovits.gguf $FIX --models $MODELS \
        --ref-wav $MODELS/ref8s.wav --threads 4 --ar-file /tmp/gsv-ar-v2-$2.gguf 2>&1)
  local ar=$(echo "$out" | grep -o "AR=[0-9.]*" | head -1 | cut -d= -f2)
  local sem=$(echo "$out" | grep -o "sem=[0-9]*" | head -1 | cut -d= -f2)
  local syn=$(echo "$out" | grep -o "synth=[0-9.]*" | head -1 | cut -d= -f2)
  echo "$ar $sem $syn"
}

for r in $(seq "$ROUNDS"); do
  for spec in "OLD q8" "NEW q8" "NEW q4k" "OLD f16" "NEW f16"; do
    set -- $spec
    tree=$([ "$1" = OLD ] && echo "$OLD" || echo "$NEW")
    read -r ar sem syn <<< "$(run_chain "$tree" "$2")"
    echo "AB r$r $1 $2 AR=$ar sem=$sem synth=$syn"
  done
done
