#!/usr/bin/env bash
set -euo pipefail

# Register every actual DP HTTP endpoint. URLs containing @rank are unsupported.
: "${P_URLS:?Space-separated Prefill endpoint URLs are required}"
: "${D_URLS:?Space-separated Decode endpoint URLs are required}"
read -r -a prefill_urls <<< "$P_URLS"
read -r -a decode_urls <<< "$D_URLS"
args=(--host "${ROUTER_HOST:-0.0.0.0}" --port "${ROUTER_PORT:-8000}"
      --lmdeploy-pd-disaggregation --lmdeploy-migration-protocol rdma
      --lmdeploy-rdma-link-type "${RDMA_LINK_TYPE:-ib}" --policy "${ROUTER_POLICY:-cache_aware}")
for url in "${prefill_urls[@]}"; do args+=(--prefill "$url"); done
for url in "${decode_urls[@]}"; do args+=(--decode "$url"); done
exec "${ROUTER_BIN:-lmdeploy-pd-router}" "${args[@]}" "$@"
