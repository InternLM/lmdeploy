# Rust API Server

The Rust API server is a TurboMind-only, in-process serving frontend. Python is
used to resolve and load the model, then the HTTP server, request scheduling,
tokenization, chat templating, response parsing, streaming, and metrics run in
Rust.

This guide describes how to prepare the Rust environment, build LMDeploy with
the Rust server, launch a model, and verify the service.

## Scope and requirements

The initial implementation supports these endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Liveness check |
| `/v1/models` | `GET` | List the served model |
| `/metrics` | `GET` | Prometheus metrics |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat, including SSE streaming |
| `/v1/completions` | `POST` | OpenAI-compatible text completion, including SSE streaming |
| `/get_ppl` | `POST` | Perplexity for text or token IDs |

The following constraints apply:

- Linux, an NVIDIA GPU, the CUDA toolkit, and the TurboMind backend are
  required.
- The model must contain a Hugging Face `tokenizer.json`, `config.json`, and a
  `tokenizer_config.json` with `chat_template`.
- Only the language model is loaded. Multimodal input is not supported.
- GPT-OSS and its Harmony parser are not supported.
- Anthropic-compatible endpoints and `/v1/responses` are planned but are not
  part of the initial endpoint set.

For a multi-GPU build, NCCL headers and libraries must also be available. A
system NCCL installation or the `nvidia-nccl-cu12` Python package can provide
them for CUDA 12.

## Prepare the build environment

Use Python 3.10 through 3.13, CMake 3.25 or newer, Ninja, a C++ compiler, and a
CUDA toolkit compatible with the target GPU.

Install the Python build dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/build.txt ninja
```

Install Rust with `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustc --version
cargo --version
```

The repository's `rust-toolchain.toml` selects the stable toolchain and adds
`rustfmt` and `clippy`. Cargo downloads the locked Rust dependencies during the
first build, so the first build needs access to crates.io.

Check CUDA and NCCL before compiling:

```bash
nvcc --version
python -c "import importlib.util; print(importlib.util.find_spec('nvidia.nccl'))"
```

The Python check may print `None` when NCCL is installed system-wide. In that
case, verify that `nccl.h` and `libnccl.so` are visible to CMake instead.

## Build

### Editable installation

For normal development, build and install the current checkout with:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 \
python -m pip install -v -e . --no-build-isolation
```

On Linux, `setup.py` enables `BUILD_MULTI_GPU`, while the root CMake project
enables `BUILD_RUST_API_SERVER`. This produces an `_turbomind` extension that
contains the Rust server entry point and NCCL support.

Verify the result:

```bash
python -c "from lmdeploy.turbomind.turbomind import _tm; print(hasattr(_tm, 'rust_api_server'))"
lmdeploy serve rust_api_server --help
```

The first command must print `True`.

### Direct CMake build

A direct CMake build is useful when iterating on C++, CUDA, or Rust code without
installing the package. The example below targets Hopper GPUs such as H100 and
H200:

```bash
BUILD_DIR=build/rust-api-server
PYTHON_ROOT=$(python -c 'import sys; print(sys.prefix)')

cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90a-real \
  -DPython3_ROOT_DIR="$PYTHON_ROOT" \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DBUILD_PY_FFI=ON \
  -DBUILD_RUST_API_SERVER=ON \
  -DBUILD_MULTI_GPU=ON

cmake --build "$BUILD_DIR" --target _turbomind -j 32
```

Use the architecture of the target GPU instead of `90a-real` when building for
another GPU, for example `80-real` for A100 or `89-real` for Ada. Hopper uses
`90a` rather than plain `90` because some TurboMind kernels use architecture-
accelerated SM90 instructions. The `real` suffix emits a native cubin only and
reduces local build time.

The extension is written to `$BUILD_DIR/lib`. Run the source checkout against
it with:

```bash
PYTHONPATH="$BUILD_DIR/lib:$PWD" \
python -c "from lmdeploy.turbomind.turbomind import _tm; print(hasattr(_tm, 'rust_api_server'))"
```

If NCCL is available only in a Python package and the loader cannot find it,
add its library directory for the current shell:

```bash
NCCL_ROOT=$(python -c \
  "import importlib.util; print(list(importlib.util.find_spec('nvidia.nccl').submodule_search_locations)[0])")
export LD_LIBRARY_PATH="$NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Launch the server

### Installed or editable build

Launch a single-GPU model with:

```bash
MODEL_PATH=/path/to/huggingface/model

CUDA_VISIBLE_DEVICES=0 \
lmdeploy serve rust_api_server "$MODEL_PATH" \
  --server-name 0.0.0.0 \
  --server-port 23333 \
  --model-name my-model \
  --max-batch-size 4 \
  --cache-max-entry-count 0.2 \
  --log-level INFO
```

For a direct CMake build, select the extension with `PYTHONPATH` and invoke the
checkout's CLI:

```bash
MODEL_PATH=/path/to/huggingface/model
BUILD_DIR=build/rust-api-server

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$BUILD_DIR/lib:$PWD" \
python -c 'from lmdeploy.cli.entrypoint import run; run()' \
serve rust_api_server "$MODEL_PATH" \
  --server-name 0.0.0.0 \
  --server-port 23333 \
  --model-name my-model \
  --max-batch-size 4 \
  --cache-max-entry-count 0.2 \
  --log-level INFO
```

For tensor parallelism across two GPUs, expose two devices and set `--tp 2`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
lmdeploy serve rust_api_server "$MODEL_PATH" \
  --server-port 23333 \
  --model-name my-model \
  --tp 2 \
  --max-batch-size 4 \
  --cache-max-entry-count 0.2
```

The number of visible GPUs must be at least the requested tensor parallel
size. The Rust CLI automatically enables language-model-only loading.

Use `--api-keys KEY1 KEY2` to require bearer authentication. Reasoning and tool
call parsing can be selected with `--reasoning-parser` and
`--tool-call-parser`; run `lmdeploy serve rust_api_server --help` for the
available CLI options. `--log-level` controls Python, TurboMind C++, and Rust
server logs; its default is `WARNING`. An explicitly set `TM_LOG_LEVEL` or
`RUST_LOG` environment variable overrides the corresponding native logger.

## Verify the service

Check liveness, the model list, and metrics:

```bash
curl -fsS http://127.0.0.1:23333/health
curl -fsS http://127.0.0.1:23333/v1/models
curl -fsS http://127.0.0.1:23333/metrics
```

Send a chat request. This example intentionally does not set `max_tokens` or
`stop`; generation ends when the model emits EOS:

```bash
curl http://127.0.0.1:23333/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "my-model",
    "messages": [
      {"role": "user", "content": "Introduce yourself in one sentence."}
    ]
  }'
```

To test SSE streaming, add `"stream": true` to the JSON object and use
`curl -N`.

Compute perplexity from text:

```bash
curl http://127.0.0.1:23333/get_ppl \
  -H 'Content-Type: application/json' \
  --data-binary '{"input":"The quick brown fox jumps over the lazy dog."}'
```

## Rust development checks

Run the Rust checks independently of the CUDA build:

```bash
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo test --manifest-path rust/Cargo.toml --workspace
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
cargo check --manifest-path rust/Cargo.toml -p rust-api-server --features ffi
```

After changing the FFI or server implementation, rebuild `_turbomind` before
running an end-to-end test.

## Troubleshooting

### `Unknown communication backend: nccl`

The loaded `_turbomind` extension was built with `BUILD_MULTI_GPU=OFF`. This is
a build-time setting; changing `CUDA_VISIBLE_DEVICES` or `--tp` cannot add NCCL
to an existing binary. Reconfigure and rebuild the same build directory:

```bash
cmake -S . -B "$BUILD_DIR" -DBUILD_MULTI_GPU=ON
cmake --build "$BUILD_DIR" --target _turbomind -j 32
```

During configuration, confirm that CMake reports `Found NCCL`.

### `available_devices=1` with `tp=2`

Only one GPU is visible to the process. Use, for example,
`CUDA_VISIBLE_DEVICES=0,1 --tp 2`, or reduce the tensor parallel size to one.

### The Rust entry point is missing

If `hasattr(_tm, 'rust_api_server')` is false, the wrong extension is being
loaded or it was built with `BUILD_RUST_API_SERVER=OFF`. Check the imported
path and rebuild:

```bash
python -c "from lmdeploy.turbomind.turbomind import _tm; print(_tm.__file__)"
```

### Model validation fails before loading weights

Confirm that the model directory has `tokenizer.json`, `tokenizer_config.json`,
and `config.json`, and that `tokenizer_config.json` defines `chat_template`.
Slow Hugging Face tokenizers that only provide tokenizer Python code are not
supported by this initial implementation.
