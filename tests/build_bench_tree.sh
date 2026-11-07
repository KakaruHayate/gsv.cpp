#!/bin/bash
# build bench_ar (+ node-prof variant) for a gsv tree.
# usage: build_bench_tree.sh <gsv_tree> <ggml_root>
set -e
T=$1
G=$2
cd "$T"
CFG="-O3 -DNDEBUG -std=c++17 -I src"
g++ $CFG -I $G/ggml/include -I $G/include tests/bench_ar.cpp -o build/bench_ar build/libgsv_engine.a \
    -L $G/build-cpu-rel/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$G/build-cpu-rel/bin -lpthread -ldl
echo "built $T/build/bench_ar"
if [ -d "$HOME/aropt/llama.cpp" ] && [ "$G" != "$HOME/aropt/llama.cpp" ]; then
  G2=$HOME/aropt/llama.cpp
  g++ $CFG -I $G2/ggml/include -I $G2/include tests/bench_ar.cpp -o build/bench_ar_priv build/libgsv_engine.a \
      -L $G2/build-cpu-rel/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$G2/build-cpu-rel/bin -lpthread -ldl
  echo "built $T/build/bench_ar_priv"
fi
