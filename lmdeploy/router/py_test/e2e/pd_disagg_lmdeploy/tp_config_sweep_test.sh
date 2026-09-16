#!/usr/bin/env bash
# Sweep LMDeploy P/D tensor-parallel combinations. The caller must provision
# one prefill and one decode server for each requested TP size.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/run_accuracy_test.sh"

for tp_size in ${TP_SIZES:-1 2}; do
  prefill_urls_var="PREFILL_TP${tp_size}_URLS"
  decode_urls_var="DECODE_TP${tp_size}_URLS"
  prefill_urls=${!prefill_urls_var:-}
  decode_urls=${!decode_urls_var:-}
  if [[ -z "${prefill_urls}" || -z "${decode_urls}" ]]; then
    echo "Skipping TP=${tp_size}: set ${prefill_urls_var} and ${decode_urls_var}." >&2
    continue
  fi

  echo "Running LMDeploy P/D accuracy test with TP=${tp_size}"
  PREFILL_URLS="${prefill_urls}" \
    DECODE_URLS="${decode_urls}" \
    bash "${TEST_SCRIPT}"
done
