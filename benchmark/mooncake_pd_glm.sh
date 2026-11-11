#!/usr/bin/env bash
set -euo pipefail

# Start one GLM-5.3-Flash PD side. Run the router separately after both sides start.
: "${MODEL_PATH:?GLM-5.3-Flash checkpoint path is required}"
: "${CUDA_VISIBLE_DEVICES:?Assign disjoint GPU sets to P and D}"
role=${1:?Specify Prefill or Decode}
case "$role" in
    Prefill) kv_role=kv_both; port=${SERVER_PORT:-19500} ;;
    Decode) kv_role=kv_producer; port=${SERVER_PORT:-19700} ;;
    *) printf 'Expected Prefill or Decode\n' >&2; exit 2 ;;
esac
args=(--backend pytorch --server-name 0.0.0.0 --server-port "$port"
      --model-name "${MODEL_NAME:-glm-pd-validation}" --role "$role" --migration-backend Mooncake
      --dp "${DP:-4}" --ep "${EP:-4}" --tp "${TP:-1}" --distributed-executor-backend ray
      --cache-max-entry-count "${CACHE_FRACTION:-0.65}" --max-batch-size "${MAX_BATCH_SIZE:-16}"
      --session-len "${SESSION_LEN:-40000}" --max-prefill-token-num "${PREFILL_TOKENS:-8192}"
      --reasoning-parser default --prefill-save-alignment 64 --prefix-cache-state-budget 16
      --prefill-save-last-only --log-level INFO)
if [[ ${PREFIX_CACHE:-1} == 1 ]]; then args+=(--enable-prefix-caching); fi
if [[ ${MTP_TOKENS:-5} != 0 ]]; then
    args+=(--speculative-algorithm deepseek_mtp --speculative-num-draft-tokens "${MTP_TOKENS:-5}")
fi
if [[ ${STORE:-1} == 1 ]]; then
    : "${MOONCAKE_CONFIG_PATH:?Store configuration path is required}"
    args+=(--kv-transfer-config "{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"$kv_role\"}")
fi
exec python -m lmdeploy serve api_server "$MODEL_PATH" "${args[@]}"
