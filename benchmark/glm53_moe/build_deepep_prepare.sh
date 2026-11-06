#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SOURCE="${1:?Usage: build_deepep_prepare.sh DEEPEP_SOURCE_DIR}"
PYTHON="${PYTHON:-python3}"
SOURCE="$("$PYTHON" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$SOURCE")"
COMMIT=c997a4ece7831f617ae70d10696f419e9bd33f38
PATCH="$REPO_ROOT/cmake/deepep_prepared_send.patch"

if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --no-checkout --filter=blob:none https://github.com/deepseek-ai/DeepEP.git "$SOURCE"
  git -C "$SOURCE" fetch origin "$COMMIT"
  git -C "$SOURCE" checkout --detach "$COMMIT"
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$COMMIT" ]]; then
  printf 'Expected DeepEP source commit %s\n' "$COMMIT" >&2
  exit 1
fi
if git -C "$SOURCE" apply --check "$PATCH"; then
  git -C "$SOURCE" apply "$PATCH"
else
  git -C "$SOURCE" apply --reverse --check "$PATCH"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-4}"
cd "$SOURCE"
exec "$PYTHON" setup.py build_ext --inplace --build-temp "${DEEPEP_BUILD_DIR:-$SOURCE/build/prepared-send}"
