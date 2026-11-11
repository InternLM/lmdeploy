#!/usr/bin/env bash
set -euo pipefail

# Run the original workload, overriding only its destination and result directory.
: "${BENCHMARK_SCRIPT:?Path to the existing evalscope multi-turn script}"
: "${BENCH_URL:?OpenAI base URL, including /v1}"
: "${BENCH_OUTPUT:?Absolute directory for benchmark results}"
export BENCH_URL BENCH_OUTPUT
evalscope() {
    command evalscope "$@" --url "$BENCH_URL" --outputs-dir "$BENCH_OUTPUT" --no-timestamp
}
export -f evalscope
cd "$(dirname "$BENCHMARK_SCRIPT")"
exec bash "$BENCHMARK_SCRIPT"
