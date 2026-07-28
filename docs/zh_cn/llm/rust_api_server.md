# Rust API Server 用户指南

Rust API Server 是一个仅支持 TurboMind、与推理引擎运行在同一进程中的
serving 前端。Python 负责定位和加载模型；HTTP 服务、请求调度、tokenization、
chat template、输出解析、流式传输和指标收集均在 Rust 中执行。

本文介绍如何准备 Rust 环境、构建带 Rust Server 的 LMDeploy、启动模型并验证
服务。

## 功能范围和要求

首期实现支持以下 endpoint：

| Endpoint | 方法 | 用途 |
| --- | --- | --- |
| `/health` | `GET` | 存活检查 |
| `/v1/models` | `GET` | 列出当前模型 |
| `/metrics` | `GET` | Prometheus 指标 |
| `/v1/chat/completions` | `POST` | OpenAI 兼容对话，支持 SSE 流式返回 |
| `/v1/completions` | `POST` | OpenAI 兼容文本补全，支持 SSE 流式返回 |
| `/get_ppl` | `POST` | 计算文本或 token IDs 的困惑度 |

当前有以下限制：

- 需要 Linux、NVIDIA GPU、CUDA Toolkit 和 TurboMind backend。
- 模型必须提供 Hugging Face `tokenizer.json`、`config.json`，以及包含
  `chat_template` 的 `tokenizer_config.json`。
- 只加载语言模型，不支持多模态输入。
- 暂不支持 GPT-OSS 及其 Harmony parser。
- Anthropic 兼容接口和 `/v1/responses` 计划在后续加入，不属于首期 endpoint。

多卡构建还要求系统能够找到 NCCL 的头文件和动态库。CUDA 12 环境可以使用
系统 NCCL 或 Python 包 `nvidia-nccl-cu12`。

## 准备构建环境

需要 Python 3.10 至 3.13、CMake 3.25 或更高版本、Ninja、C++ 编译器，以及与
目标 GPU 匹配的 CUDA Toolkit。

在仓库根目录安装 Python 构建依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/build.txt ninja
```

通过 `rustup` 安装 Rust：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustc --version
cargo --version
```

仓库中的 `rust-toolchain.toml` 会选择 stable toolchain，并安装 `rustfmt` 和
`clippy`。Cargo 首次构建时会下载锁定版本的 Rust 依赖，因此首次构建需要能够
访问 crates.io。

编译前检查 CUDA 和 NCCL：

```bash
nvcc --version
python -c "import importlib.util; print(importlib.util.find_spec('nvidia.nccl'))"
```

如果 NCCL 安装在系统目录中，第二条命令可能输出 `None`；此时只需确保 CMake
能够找到 `nccl.h` 和 `libnccl.so`。

## 构建

### Editable 安装

日常开发推荐直接构建并安装当前 checkout：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 \
python -m pip install -v -e . --no-build-isolation
```

Linux 下 `setup.py` 会启用 `BUILD_MULTI_GPU`，根 CMake 工程会启用
`BUILD_RUST_API_SERVER`。最终生成的 `_turbomind` 扩展同时包含 Rust server
入口和 NCCL 多卡支持。

验证构建结果：

```bash
python -c "from lmdeploy.turbomind.turbomind import _tm; print(hasattr(_tm, 'rust_api_server'))"
lmdeploy serve rust_api_server --help
```

第一条命令必须输出 `True`。

### 直接使用 CMake 构建

如果需要频繁修改 C++、CUDA 或 Rust 代码，可以直接使用 CMake 构建而不安装
package。下面的例子面向 H100、H200 等 Hopper GPU：

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

其他 GPU 应替换 CUDA architecture，例如 A100 使用 `80-real`，Ada 使用
`89-real`。Hopper 使用 `90a` 而不是普通 `90`，是因为部分 TurboMind kernel
使用了 SM90 architecture-accelerated 指令。`real` 后缀只生成目标 GPU 的原生
cubin，可以缩短本机构建时间。

扩展会生成在 `$BUILD_DIR/lib` 中。可以用下面的方式让源码 checkout 加载它：

```bash
PYTHONPATH="$BUILD_DIR/lib:$PWD" \
python -c "from lmdeploy.turbomind.turbomind import _tm; print(hasattr(_tm, 'rust_api_server'))"
```

如果 NCCL 仅安装在 Python package 中，且运行时 loader 找不到它，可以在当前
shell 中加入 NCCL 的动态库目录：

```bash
NCCL_ROOT=$(python -c \
  "import importlib.util; print(list(importlib.util.find_spec('nvidia.nccl').submodule_search_locations)[0])")
export LD_LIBRARY_PATH="$NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 启动服务

### 已安装或 editable build

单卡启动命令如下：

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

直接使用 CMake build 时，通过 `PYTHONPATH` 选择刚构建的扩展，并调用当前
checkout 中的 CLI：

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

两卡 tensor parallel 启动时，需要暴露两张 GPU 并设置 `--tp 2`：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
lmdeploy serve rust_api_server "$MODEL_PATH" \
  --server-port 23333 \
  --model-name my-model \
  --tp 2 \
  --max-batch-size 4 \
  --cache-max-entry-count 0.2
```

可见 GPU 数量不能少于 tensor parallel size。Rust CLI 会自动启用
language-model-only 模式。

使用 `--api-keys KEY1 KEY2` 可以要求 Bearer token 鉴权。Reasoning 和 tool
call parser 分别通过 `--reasoning-parser` 和 `--tool-call-parser` 选择；完整参数
请运行 `lmdeploy serve rust_api_server --help` 查看。`--log-level` 会同时控制
Python、TurboMind C++ 和 Rust server 日志，默认值为 `WARNING`。如果显式设置了
`TM_LOG_LEVEL` 或 `RUST_LOG` 环境变量，则对应的 native logger 以环境变量为准。

## 验证服务

检查服务状态、模型列表和 metrics：

```bash
curl -fsS http://127.0.0.1:23333/health
curl -fsS http://127.0.0.1:23333/v1/models
curl -fsS http://127.0.0.1:23333/metrics
```

发送对话请求。下面的请求特意不设置 `max_tokens` 或 `stop`，由模型输出 EOS
后自然停止：

```bash
curl http://127.0.0.1:23333/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "my-model",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ]
  }'
```

测试 SSE streaming 时，在 JSON 对象中加入 `"stream": true`，并使用
`curl -N`。

通过文本计算困惑度：

```bash
curl http://127.0.0.1:23333/get_ppl \
  -H 'Content-Type: application/json' \
  --data-binary '{"input":"The quick brown fox jumps over the lazy dog."}'
```

## Rust 开发检查

以下检查不需要构建 CUDA 代码：

```bash
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo test --manifest-path rust/Cargo.toml --workspace
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
cargo check --manifest-path rust/Cargo.toml -p rust-api-server --features ffi
```

修改 FFI 或 server 实现后，需要重新构建 `_turbomind` 再进行端到端测试。

## 常见问题

### `Unknown communication backend: nccl`

当前加载的 `_turbomind` 是在 `BUILD_MULTI_GPU=OFF` 下构建的。这是编译期设置；
修改 `CUDA_VISIBLE_DEVICES` 或 `--tp` 无法为已有二进制补上 NCCL。需要重新配置
并构建同一个 build directory：

```bash
cmake -S . -B "$BUILD_DIR" -DBUILD_MULTI_GPU=ON
cmake --build "$BUILD_DIR" --target _turbomind -j 32
```

配置日志中应当出现 `Found NCCL`。

### `available_devices=1`，但设置了 `tp=2`

当前进程只看得到一张 GPU。可以使用 `CUDA_VISIBLE_DEVICES=0,1 --tp 2`，或者
把 tensor parallel size 降为一。

### 找不到 Rust server 入口

如果 `hasattr(_tm, 'rust_api_server')` 为 false，说明加载了错误的扩展，或者构建
时设置了 `BUILD_RUST_API_SERVER=OFF`。先检查实际加载路径：

```bash
python -c "from lmdeploy.turbomind.turbomind import _tm; print(_tm.__file__)"
```

然后用 `BUILD_RUST_API_SERVER=ON` 重新构建。

### 加载权重前模型校验失败

检查模型目录中是否存在 `tokenizer.json`、`tokenizer_config.json` 和
`config.json`，并确认 `tokenizer_config.json` 定义了 `chat_template`。首期实现
不支持只提供 Python tokenizer 代码的 slow Hugging Face tokenizer。
