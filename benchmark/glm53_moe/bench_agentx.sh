#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
set -euo pipefail

ARTIFACT_DIR="${1:?Usage: AIPERF_URL=URL TOKENIZER=PATH bench_agentx.sh ARTIFACT_DIR}"
: "${AIPERF_URL:?Set the router URL including /v1}"
: "${TOKENIZER:?Set the local GLM-5.3-Flash tokenizer directory}"
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID="${AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID:-1}"

exec "${AIPERF_BIN:-aiperf}" profile \
  --scenario inferencex-agentx-mvp \
  --url "$AIPERF_URL" \
  --model "${MODEL_NAME:-glm53-flash}" \
  --tokenizer "$TOKENIZER" \
  --endpoint-type chat \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --max-context-length 256000 \
  --concurrency 80 \
  --benchmark-duration 1800 \
  --streaming \
  --goodput 'time_to_first_token:3000 output_token_throughput_per_user:40' \
  --use-server-token-count \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --random-seed 20260707 \
  --ui simple \
  --warmup-request-count 1 \
  --output-artifact-dir "$ARTIFACT_DIR"
