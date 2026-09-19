#!/usr/bin/env bash
# Run LMDeploy P/D disaggregation router accuracy tests against externally
# managed LMDeploy Prefill and Decode API servers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH=${MODEL_PATH:-"${LMDEPLOY_E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"}
MODEL_NAME=${MODEL_NAME:-LMDEPLOY_MODEL_NAME:-${MODEL_PATH##*/}}
NUM_REQUESTS=${NUM_REQUESTS:-20}
NUM_STREAMING_REQUESTS=${NUM_STREAMING_REQUESTS:-5}
MAX_TOKENS=${MAX_TOKENS:-30}
ROUTER_PORT=${ROUTER_PORT:-8300}
PREFILL_URLS=${PREFILL_URLS:?set PREFILL_URLS to comma-separated LMDeploy Prefill URLs}
DECODE_URLS=${DECODE_URLS:?set DECODE_URLS to comma-separated LMDeploy Decode URLs}
MIGRATION_PROTOCOL=${MIGRATION_PROTOCOL:-rdma}
RDMA_LINK_TYPE=${RDMA_LINK_TYPE:-roce}

ROUTER_BIN=${LMDEPLOY_ROUTER_BIN:-${REPO_ROOT}/target/debug/lmdeploy-router}
if [[ ! -x "${ROUTER_BIN}" ]]; then
  echo "Router binary not found: ${ROUTER_BIN}" >&2
  exit 1
fi

PREFILL_ARGS=()
for url in ${PREFILL_URLS//,/ }; do
  PREFILL_ARGS+=("--prefill" "${url}")
done
DECODE_ARGS=()
for url in ${DECODE_URLS//,/ }; do
  DECODE_ARGS+=("--decode" "${url}")
done

ROUTER_LOG=${ROUTER_LOG:-/tmp/lmdeploy-pd-router.log}
"${ROUTER_BIN}" \
  --host 127.0.0.1 \
  --port "${ROUTER_PORT}" \
  --policy round_robin \
  --lmdeploy-pd-disaggregation \
  --lmdeploy-migration-protocol "${MIGRATION_PROTOCOL}" \
  --lmdeploy-rdma-link-type "${RDMA_LINK_TYPE}" \
  --worker-startup-check-interval 1 \
  "${PREFILL_ARGS[@]}" \
  "${DECODE_ARGS[@]}" \
  >"${ROUTER_LOG}" 2>&1 &
ROUTER_PID=$!

cleanup() {
  kill "${ROUTER_PID}" 2>/dev/null || true
  wait "${ROUTER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if curl --fail --silent "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
    echo "Router exited during startup. Log tail:" >&2
    tail -100 "${ROUTER_LOG}" >&2
    exit 1
  fi
  sleep 1
done

if ! curl --fail --silent "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null; then
  echo "Router failed to become healthy. Log tail:" >&2
  tail -100 "${ROUTER_LOG}" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/test_pd_accuracy.py" \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --model "${MODEL_NAME}" \
  --num-requests "${NUM_REQUESTS}" \
  --num-streaming-requests "${NUM_STREAMING_REQUESTS}" \
  --max-tokens "${MAX_TOKENS}"
