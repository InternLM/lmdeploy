# Router 支持 LMDeploy 推理与 PD 分离 — 编译与测试指南

## 目录

1. [概述](#1-概述)
2. [代码改动清单](#2-代码改动清单)
3. [环境准备](#3-环境准备)
4. [编译与单元测试](#4-编译与单元测试)
5. [集成测试环境](#5-集成测试环境)
6. [负载均衡策略测试](#6-负载均衡策略测试)
7. [适配 lmdeploy 测试套件](#7-适配-lmdeploy-测试套件)
8. [测试结果汇总](#8-测试结果汇总)
9. [完整复现步骤](#9-完整复现步骤)
10. [Python wheel 打包与安装](#10-python-wheel-打包与安装)

---

## 1. 概述

本指南描述如何为 lmdeploy-router 添加 LMDeploy 推理引擎支持，包括：

- **Phase 1（普通模式）**：Router 天然支持 LMDeploy 的 OpenAI 兼容端点（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/generate`），无需代码改动。
- **Phase 2（PD 分离）**：新增 `LMDeployPDRouter`，支持 LMDeploy 的 DistServe PD 分离协议（`migration_request` + `/distserve/p2p_initialize` + `/distserve/p2p_connect` 两阶段流程）。
- **动态注册**：Router 可以不指定任何后端 URL 先启动；LMDeploy API Server 再通过原生 `--proxy-url` 参数向 Router 的 `/nodes/add` 注册。普通 Hybrid 和 PD Prefill/Decode 两种部署均支持。
- **Token-in-Token-out**：支持 LMDeploy 的 `input_ids`/`output_ids` 透传，并为所有负载均衡策略提供基于 token-id 的路由键。

### PD 协议差异

| 维度 | 传统 PD 语义 | LMDeploy |
|------|--------------|----------|
| PD 协调字段 | `kv_transfer_params`（请求体内嵌） | `migration_request`（decode 请求体） |
| KV 传输 | nixl/mooncake/moriio engine-to-engine | RDMA (DLSlime/Mooncake) out-of-band |
| P2P 建立 | bootstrap 信息交换（无显式端点） | `/distserve/p2p_initialize` + `/distserve/p2p_connect`（显式 HTTP） |
| Prefill 特殊参数 | `max_tokens=1, stream=false` | `max_tokens=1, with_cache=true, preserve_cache=true` |
| 独立 token API | `/inference/v1/generate` + `GenerateRequest{token_ids}` | `/generate` + `GenerateReqInput{input_ids}`；PD 模式的 token-in/out 使用 Chat Completions |

---

## 2. 代码改动清单

核心实现涉及以下文件（包含初始 LMDeploy 适配和动态注册功能）：

| 文件 | 类型 | 改动说明 |
|------|--------------|----------|
| `src/config/types.rs` | 修改 | 新增 `LMDeployMigrationProtocol` 枚举、`LMDeployRdmaConfig` 结构体、`RoutingMode::LMDeployPrefillDecode` 变体；更新 `is_pd_mode`/`is_lmdeploy_pd_mode`/`worker_count`/`get_prefill_policy`/`get_decode_policy`/`mode_type`；新增序列化测试 |
| `src/config/validation.rs` | 修改 | 新增 `LMDeployPrefillDecode` 的 `validate_mode`/`validate_discovery`/`validate_compatibility` 三个分支 |
| `src/routers/http/mod.rs` | 修改 | 注册 `pub mod lmdeploy_pd_router;` |
| `src/routers/http/lmdeploy_pd_router.rs` | **新建** | `LMDeployPDRouter` 实现 |
| `src/routers/factory.rs` | 修改 | 新增 `LMDeployPrefillDecode` match 分支 + `create_lmdeploy_pd_router()` |
| `src/main.rs` | 修改 | LMDeploy PD CLI flags + mode 选择分支 + `pd_mode` 联动；允许普通/LMDeploy PD 模式零 URL 启动 |
| `src/lib.rs` | 修改 | PyO3 Router 配置字段 + 默认值 + `new()` 入参 + `to_router_config` mode 分支 |
| `src/protocols/spec.rs` | 修改 | 为 Chat 和 Generate 请求显式建模 `input_ids`，直接生成 cache-aware/consistent-hash 路由键；为 Chat 普通/流式响应和 Generate 响应建模 `output_ids`；保留 `GenerateRequest` 的 `#[serde(flatten)] other` 以透明转发 LMDeploy 顶层扩展参数 |
| `src/server.rs` | 修改 | 新增与 LMDeploy 原生 proxy 兼容的 `/nodes/add`、`/nodes/remove`、`/nodes/status` 接口，并按 Hybrid/Prefill/Decode role 分发注册 |
| `src/routers/http/router.rs` | 修改 | 普通 Router 支持 LMDeploy startup callback 阶段的无探活注册 |
| `src/routers/http/pd_router.rs` | 修改 | PD Router 支持 Prefill/Decode startup callback 阶段的无探活注册 |
| `tests/lmdeploy_registration_test.rs` | 新建 | 覆盖普通和 PD 动态注册、重复注册、状态查询、注销及非法 role |

### 关键改动详解

#### 2.1 `RoutingMode::LMDeployPrefillDecode` (types.rs)

```rust
LMDeployPrefillDecode {
    prefill_urls: Vec<String>,
    decode_urls: Vec<String>,
    prefill_policy: Option<PolicyConfig>,
    decode_policy: Option<PolicyConfig>,
    migration_protocol: LMDeployMigrationProtocol,
    rdma_config: Option<LMDeployRdmaConfig>,
    dummy_prefill: bool,
}
```

serde tag: `"lmdeploy_prefill_decode"`

#### 2.2 `GenerateRequest` flatten 修复 (spec.rs)

该字段用于普通 Router 的透明转发。LMDeploy 当前的 `/generate` 和 `/v1/responses`
端点不暴露 PD 所需的 cache migration 元数据，因此 LMDeploy PD Router 会对这两个
端点返回 `501 Not Implemented`；PD 模式只支持 `/v1/chat/completions` 和
`/v1/completions`。

**Fix**: 在 `GenerateRequest` 结构体末尾添加：
```rust
#[serde(flatten)]
pub other: serde_json::Map<String, serde_json::Value>,
```

#### 2.3 Token-in 路由键 (spec.rs)

`ChatCompletionRequest.input_ids` 使用显式的 `Option<Vec<i32>>`，与 LMDeploy 的
`list[int] | None` 一致；`GenerateRequest.input_ids` 使用 `Option<InputIds>`，同时兼容
LMDeploy 单请求和 Router 的 batch 格式。`extract_text_for_routing` 直接读取 typed
字段并拼接为稳定路由键，优先级为：`session_id` -> `input_ids` -> 空字符串，不再从
`#[serde(flatten)] other` 中二次解析 JSON。

#### 2.4 CLI flags (main.rs)

| Flag | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lmdeploy-pd-disaggregation` | bool | false | 启用 LMDeploy PD 分离模式 |
| `--prefill` | Vec | - | prefill 节点 URL |
| `--decode` | Vec | - | decode 节点 URL |
| `--lmdeploy-migration-protocol` | enum | rdma | 迁移协议：rdma/nvlink |
| `--lmdeploy-rdma-link-type` | enum | roce | RDMA 链路类型：roce/ib |
| `--lmdeploy-disable-gdr` | bool | false | 禁用 GPU Direct RDMA |
| `--lmdeploy-dummy-prefill` | bool | false | 是否使用 dummy prefill |

RDMA 设备选择不通过 Router 请求传递。DLSlime 请使用
`SLIME_VISIBLE_DEVICES`，Mooncake 使用 LMDeploy 后端发现的 RDMA 网卡。

---

## 3. 环境准备

### 3.1 安装 Rust 工具链

```bash
# 安装 rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 验证
rustc --version  # 1.97.1
cargo --version  # 1.97.1
```

### 3.2 配置 Cargo（磁盘空间 + 代理）

以下命令均从 Router 仓库根目录执行。默认使用仓库内的相对构建和测试产物目录；如磁盘
空间不足，可在执行命令前自行将 `CARGO_TARGET_DIR` 指向其他磁盘：

```bash
export ROUTER_ROOT="$(pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
export CARGO_HTTP_MULTIPLEXING=false  # 代理环境下 HTTP/2 多路复用会挂起
export LMDEPLOY_ROUTER_BIN="${CARGO_TARGET_DIR}/release/lmdeploy-router"
export LMDEPLOY_TEST_ARTIFACT_DIR="${LMDEPLOY_TEST_ARTIFACT_DIR:-.test-artifacts/lmdeploy}"
mkdir -p "${LMDEPLOY_TEST_ARTIFACT_DIR}"
```

### 3.3 配置 crates.io 镜像（中国大陆）

```bash
mkdir -p "${CARGO_HOME:-$HOME/.cargo}"
cat > "${CARGO_HOME:-$HOME/.cargo}/config.toml" << 'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
git-fetch-with-cli = true
EOF
```

### 3.4 安装 Python 测试依赖

```bash
python -m pip install -e '.[dev]'
```

测试使用仓库的 `pytest.ini` 和 `py_test/integration/conftest.py`，无需从 `/tmp` 复制或
单独维护脚本。

---

## 4. 编译与单元测试

### 4.1 编译检查

```bash
cargo check --lib
```

预期输出：
```
    Finished dev profile [unoptimized + debuginfo] target(s) in 11.71s
```

命令应无编译错误；warning 应逐项确认，文档不将 warning 视为默认可忽略项。

### 4.2 运行单元测试

```bash
cargo test --lib
```

本次验证结果：**524 passed, 0 failed**。

关键测试：
- `config::types::test_lmdeploy_prefill_decode_serialization`
- `config::types::test_lmdeploy_migration_protocol_serde_variants`
- `protocols::spec::test_chat_completion_extract_routing_with_input_ids`
- `protocols::spec::test_chat_completion_preserves_typed_input_ids_on_serialize`
- `protocols::spec::test_lmdeploy_chat_response_preserves_output_ids`
- `protocols::spec::test_lmdeploy_generate_response_preserves_output_ids`
- `routers::http::lmdeploy_pd_router::test_build_migration_request_extracts_fields`
- `routers::http::lmdeploy_pd_router::test_prepare_prefill_request_sets_with_cache_and_preserve_cache`

### 4.3 编译二进制

```bash
cargo build --release --bin lmdeploy-router
```

预期输出：
```
    Finished release profile [optimized] target(s) in ...
```

二进制位置：`${CARGO_TARGET_DIR}/release/lmdeploy-router`。后续命令统一使用
`${LMDEPLOY_ROUTER_BIN}`，不依赖某台机器的仓库绝对路径。

---

## 5. 集成测试环境

### 5.1 后端部署

两个 LMDeploy api_server 实例（非 PD 分离，普通模式）：

```bash
export MODEL_PATH="${MODEL_PATH:-qwen3-06b}"

# Backend 1 (port 9000)
lmdeploy serve api_server "${MODEL_PATH}" \
    --server-name 0.0.0.0 --server-port 9000 \
    --logprobs-mode raw_logprobs

# Backend 2 (port 9003；与 PD Prefill 的 9001 错开)
lmdeploy serve api_server "${MODEL_PATH}" \
    --server-name 0.0.0.0 --server-port 9003 \
    --logprobs-mode raw_logprobs
```

验证后端：
```bash
curl http://10.102.98.154:9000/v1/models
# {"id":"qwen3-06b","owned_by":"lmdeploy"}

curl http://10.102.98.154:9003/v1/models
# {"id":"qwen3-06b","owned_by":"lmdeploy"}
```

### 5.2 启动 Router

```bash
# 普通模式（非 PD）
setsid "${LMDEPLOY_ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port 30000 \
    --worker-urls http://10.102.98.154:9000 http://10.102.98.154:9003 \
    --policy random \
    --prometheus-port 19001 \
    > "${LMDEPLOY_TEST_ARTIFACT_DIR}/router.log" 2>&1 < /dev/null &
disown
```

验证 Router：
```bash
curl http://127.0.0.1:30000/health
# All servers healthy

curl http://127.0.0.1:30000/v1/models
# {"data":[{"id":"qwen3-06b",...}]}

curl http://127.0.0.1:30000/list_workers
# {"urls":["http://10.102.98.154:9000","http://10.102.98.154:9003"]}
```

### 5.3 PD 分离模式静态启动（可选）

```bash
setsid "${LMDEPLOY_ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port 30000 \
    --lmdeploy-pd-disaggregation \
    --prefill http://prefill-host:23333 \
    --decode http://decode-host:23333 \
    --lmdeploy-migration-protocol rdma \
    --lmdeploy-rdma-link-type roce \
    --prometheus-port 19001 \
    > "${LMDEPLOY_TEST_ARTIFACT_DIR}/router_pd.log" 2>&1 < /dev/null &
disown
```

### 5.4 LMDeploy 原生动态注册概览

除静态传入 `--worker-urls`、`--prefill` 和 `--decode` 外，Router 还兼容
LMDeploy 原生 proxy 的启动方式：

1. 先启动一个没有后端 URL 的 Router。
2. 启动 LMDeploy API Server，并传入 `--proxy-url http://<router>:<port>`。
3. LMDeploy 在 FastAPI startup callback 中向 Router 发送 `POST /nodes/add`。
4. Router 根据注册请求中的 role，将节点加入普通、Prefill 或 Decode worker pool。
5. API Server 开始监听后，Router 的健康检查器负责维护节点可用状态。

LMDeploy 启动时发送的请求体如下：

```json
{
  "url": "http://10.102.98.154:9001",
  "status": {
    "models": ["qwen3_06B"],
    "role": 2
  }
}
```

role 与 Router 模式的对应关系：

| role 数值 | LMDeploy role | Router 启动模式 | 注册目标 |
|-----------|---------------|-----------------|----------|
| `1` | `Hybrid` | 普通模式 | Regular worker pool |
| `2` | `Prefill` | `--lmdeploy-pd-disaggregation` | Prefill worker pool |
| `3` | `Decode` | `--lmdeploy-pd-disaggregation` | Decode worker pool |

普通模式只接受 Hybrid，LMDeploy PD 模式只接受 Prefill/Decode。role 与 Router
模式不匹配时，`/nodes/add` 返回 `400 Bad Request`。相同 URL 和 role 的重复注册是
幂等操作，不会产生重复 worker。

> **为什么动态注册不在 `/nodes/add` 中同步探活？**
>
> LMDeploy 从 FastAPI startup callback 发起注册，此时 Uvicorn 尚未开始接受
> `/health` 请求。如果 Router 在处理 `/nodes/add` 时等待该节点健康，两端会互相
> 等待直到超时。动态注册路径因此先写入 worker registry，随后由 Router 的后台
> 健康检查器接管；原有 `/add_worker` 和静态 URL 启动流程仍保留探活行为。

### 5.5 普通模式动态注册（Hybrid）

先启动不带 `--worker-urls` 的 Router：

```bash
setsid "${LMDEPLOY_ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port 30000 \
    --policy round_robin \
    --prometheus-port 19001 \
    > "${LMDEPLOY_TEST_ARTIFACT_DIR}/router_dynamic.log" 2>&1 < /dev/null &
disown
```

然后启动 LMDeploy Hybrid 服务。`--role Hybrid` 是默认值，建议显式写出以便排查：

```bash
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server "${MODEL_PATH}" \
    --tp 1 \
    --backend pytorch \
    --server-name 10.102.98.154 \
    --server-port 9000 \
    --model-name qwen3_06B \
    --role Hybrid \
    --proxy-url http://ROUTER_IP:30000
```

验证注册状态和数据面：

```bash
curl http://ROUTER_IP:30000/nodes/status
# {
#   "http://10.102.98.154:9000": {
#     "role": 1,
#     "models": ["qwen3_06B"],
#     "unfinished": 0,
#     "latency": [],
#     "speed": null
#   }
# }

curl http://ROUTER_IP:30000/list_workers
# {"urls":["http://10.102.98.154:9000"]}

curl -X POST http://ROUTER_IP:30000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen3_06B",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 32,
      "stream": false
    }'
```

可以启动多个 Hybrid API Server，并将它们的 `--proxy-url` 指向同一个 Router；每个
实例都会独立注册，Router 按 `--policy` 选择节点。

### 5.6 PD 分离模式动态注册（Prefill/Decode）

先启动不带 `--prefill`、`--decode` 的 LMDeploy PD Router：

```bash
setsid "${LMDEPLOY_ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port 30000 \
    --lmdeploy-pd-disaggregation \
    --lmdeploy-migration-protocol rdma \
    --lmdeploy-rdma-link-type roce \
    --policy round_robin \
    --prefill-policy round_robin \
    --decode-policy round_robin \
    --prometheus-port 19001 \
    > "${LMDEPLOY_TEST_ARTIFACT_DIR}/router_pd_dynamic.log" 2>&1 < /dev/null &
disown
```

再分别启动 Prefill 和 Decode。两端的 `--migration-backend` 必须一致，并与实际环境
配置匹配；以下示例使用 DLSlime：

```bash
export SLIME_VISIBLE_DEVICES="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export SLIME_GID_INDEX=3

# Prefill: GPU 1 / port 9001
CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server "${MODEL_PATH}" \
    --tp 1 \
    --backend pytorch \
    --server-name 10.102.98.154 \
    --server-port 9001 \
    --model-name qwen3_06B \
    --role Prefill \
    --migration-backend DLSlime \
    --proxy-url http://ROUTER_IP:30000

# Decode: GPU 2 / port 9002
CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server "${MODEL_PATH}" \
    --tp 1 \
    --backend pytorch \
    --server-name 10.102.98.154 \
    --server-port 9002 \
    --model-name qwen3_06B \
    --role Decode \
    --migration-backend DLSlime \
    --proxy-url http://ROUTER_IP:30000
```

Mooncake 部署只需将两端改为 `--migration-backend Mooncake`，并完成对应的 Mooncake
网卡和存储配置。Router 的 `--lmdeploy-migration-protocol` 与
`--lmdeploy-rdma-link-type` 描述 Router 发给 LMDeploy 的 migration request；它们不
替代 DLSlime/Mooncake 自身的网卡环境变量。

验证两个 role 均已注册：

```bash
curl http://ROUTER_IP:30000/nodes/status
# {
#   "http://10.102.98.154:9001": {
#     "role": 2,
#     "models": ["qwen3_06B"],
#     "unfinished": 0,
#     "latency": [],
#     "speed": null
#   },
#   "http://10.102.98.154:9002": {
#     "role": 3,
#     "models": ["qwen3_06B"],
#     "unfinished": 0,
#     "latency": [],
#     "speed": null
#   }
# }
```

此后向 Router 的 `/v1/chat/completions` 或 `/v1/completions` 发请求即可触发
Prefill、KV cache migration、Decode 两阶段流程。LMDeploy PD 模式下 `/generate`
和 `/v1/responses` 不具备 migration 元数据，Router 会返回 `501 Not Implemented`。

### 5.7 注册管理接口

#### 查询节点

```bash
curl http://ROUTER_IP:30000/nodes/status
```

响应是以节点 URL 为 key、LMDeploy `Status` 为 value 的 JSON 对象。字段与
`lmdeploy.serve.proxy.proxy.Status` 保持一致：`role`、`models`、`unfinished`、
`latency` 和 `speed`。

#### 手动注册

正常情况下无需手动调用；以下命令可用于调试或接入自定义启动器：

```bash
curl -X POST http://ROUTER_IP:30000/nodes/add \
    -H 'Content-Type: application/json' \
    -d '{
      "url": "http://10.102.98.154:9000",
      "status": {"models": ["qwen3_06B"], "role": 1}
    }'
# "Added successfully"
```

#### 注销节点

```bash
curl -X POST http://ROUTER_IP:30000/nodes/remove \
    -H 'Content-Type: application/json' \
    -d '{"url": "http://10.102.98.154:9000"}'
# "Deleted successfully"
```

注销是幂等操作。当前 LMDeploy API Server 的 shutdown callback 不会主动请求
`/nodes/remove`；进程退出后，Router 的健康检查会将 worker 标记为不可用。如需立即
从 `/nodes/status` 和 worker registry 删除节点，应由进程管理器调用
`/nodes/remove`。同一 URL 的服务重启后可以直接重新注册。

### 5.8 网络、安全与排障

动态注册要求网络双向可达：

- LMDeploy API Server 必须能访问 `--proxy-url` 指向的 Router。
- Router 必须能访问注册请求体 `url` 指向的 LMDeploy API Server。
- 跨机器部署时，`--server-name` 必须使用 Router 可路由的节点 IP 或主机名，不能使用
  `0.0.0.0`、`127.0.0.1` 或仅在容器内部可解析的地址。
- `--server-name` 还决定 Uvicorn 的监听地址；该 IP 必须存在于 LMDeploy 所在主机或
  容器的网络命名空间中。

LMDeploy 原生 `--proxy-url` 注册请求不携带 Authorization header，因此
`/nodes/add`、`/nodes/remove` 和 `/nodes/status` 是未鉴权的控制面接口。生产环境应通过
安全组、防火墙或反向代理，只允许可信 LMDeploy 节点访问这些路径。

常见问题：

| 现象 | 检查项 |
|------|--------|
| API Server 已启动，但 `/nodes/status` 为空 | 检查 LMDeploy 日志是否出现 `Service registration failed`；从 API Server 所在机器 `curl <proxy-url>/nodes/status` |
| 注册成功，但请求报连接失败 | 检查 `--server-name` 是否为 Router 可回连地址；确认注册 JSON 中没有 `0.0.0.0` 或 `127.0.0.1` |
| `/nodes/add` 长时间卡住 | 确认使用的是包含动态注册修复的 Router；注册接口不应同步等待 API Server `/health` |
| PD 注册返回 `400` | Router 必须带 `--lmdeploy-pd-disaggregation`，Prefill/Decode 的 role 分别为 `2`/`3` |
| 普通注册返回 `400` | 普通 Router 只接受 Hybrid role `1` |
| PD 已注册但迁移失败 | 检查两端 migration backend、RDMA 网卡/GID、Router migration protocol/link type 是否一致 |

静态 URL 与动态注册可以共存。例如可用 `--prefill` 配置固定 Prefill，同时让新增的
Decode 节点通过 `--proxy-url` 注册；不过生产环境建议采用一致的节点生命周期管理方式，
以便排障和容量统计。

---

## 6. 负载均衡策略测试

### 6.1 pytest 测试位置

测试位于 `py_test/integration/lmdeploy/test_load_balancing.py`，使用 Router 现有的 pytest
规范和 fixture：

- 每个策略自动分配 Router 与 Prometheus 端口，并在用例结束后回收进程。
- 每个策略发送 10 个 messages 请求和 10 个 `input_ids` 请求。
- 通过 LMDeploy 每个 worker 独立递增的数值响应 `id` 判断实际 worker。
- `temperature=0`，严格比较 Router 响应与对应直连后端的文本；token-in 请求同时比较
  `output_ids`。
- 日志默认写入相对目录 `.test-artifacts/lmdeploy/`。

### 6.2 配置真实后端

```bash
export LMDEPLOY_BACKEND_URLS="http://10.102.98.154:9000,http://10.102.98.154:9003"
export LMDEPLOY_ROUTER_BIN="${CARGO_TARGET_DIR}/release/lmdeploy-router"
```

`LMDEPLOY_MODEL` 默认从所有后端的 `/v1/models` 自动发现；只有后端暴露多个模型且需要
指定其中一个时才设置，例如 `export LMDEPLOY_MODEL="qwen3-06b"`。

如模型 tokenizer 不同，可用 JSON 数组覆盖 token-in 测试数据：

```bash
export LMDEPLOY_TOKEN_IDS='[151644,8948,198,2610]'
```

### 6.3 运行测试

```bash
pytest -v py_test/integration/lmdeploy/test_load_balancing.py
```

未设置 `LMDEPLOY_BACKEND_URLS` 时测试会 skip，不会把开发机地址硬编码进默认 CI。

### 6.4 策略断言

| 策略 | 验证逻辑 |
|------|----------|
| random | 两个 worker 均处理请求 |
| round_robin | messages 请求严格交替 |
| power_of_two | 两个 worker 均处理请求 |
| consistent_hash | 相同 messages/input_ids 路由键分别固定到同一 worker |
| cache_aware | 相同 input_ids 和递增 token 前缀固定到同一 worker |

---

## 7. 适配 lmdeploy 测试套件

### 7.1 测试来源

适配自 LMDeploy 的 autotest 测试套件：
- `<lmdeploy-repo>/autotest/interface/restful/test_restful_chat_completions_v1.py`
- `<lmdeploy-repo>/autotest/interface/restful/test_restful_completions_v1.py`
- `<lmdeploy-repo>/autotest/interface/restful/test_restful_generate.py`
- `<lmdeploy-repo>/autotest/utils/restful_return_check.py`（断言辅助函数）

### 7.2 Router 测试

适配测试位于 `py_test/integration/lmdeploy/test_rest_api.py`，不再使用 `/tmp` 中的独立
main 脚本。每一个验证点都是独立 pytest case，失败时可直接使用 `-k` 选择并由 pytest
输出 traceback。

主要改动：
- Router 地址和端口由 fixture 动态分配。
- 后端、模型、超时和 token IDs 通过环境变量配置。
- 跳过 router 不代理的端点测试（`/encode`、tokenizer 相关）
- 新增 strict content match 测试（temp=0 下比较 router vs 直连后端）
- 新增 token-in-token-out 测试（`input_ids` + `return_token_ids` -> `output_ids`）
- 流式响应按 SSE event 独立校验。

### 7.3 运行测试

```bash
pytest -v py_test/integration/lmdeploy/test_rest_api.py
```

### 7.4 测试矩阵（28 项）

| 类别 | 测试项 | 数量 | 结果 |
|------|--------|------|------|
| 基础端点 | /v1/models, /health, /list_workers | 3 | 3/3 PASS |
| Chat Completions | basic, streaming, single_stopword, array_stopwords, max_tokens, max_completion_tokens, input_ids(token-in), input_ids_vs_direct, temp=0_determinism, logprobs, invalid_model, strict_content_match | 12 | 12/12 PASS |
| Completions | basic, streaming, stopword, max_tokens, strict_content_match | 5 | 5/5 PASS |
| Generate | basic, input_ids(token-in), streaming, stop_token_ids, session_id, conflict_prompt+input_ids(400), empty_prompt(400), strict_content_match | 8 | 8/8 PASS |

### 7.5 注册与 PD 测试

`py_test/integration/lmdeploy/test_registration.py` 覆盖：

- Hybrid role=1 动态注册、重复注册幂等、完整状态字段、真实推理和幂等注销。
- 普通 Router 拒绝 Prefill role，PD Router 拒绝 Hybrid role。
- Prefill role=2 与 Decode role=3 的 PD 动态注册、重复注册和真实推理。
- 通过 `--prefill`/`--decode` 静态配置的 PD 真实推理。
- 静态和动态 PD 模式下 `/generate`、`/v1/responses` 均返回 `501`，避免绕过 KV
  migration 流程。

PD 真实测试需要额外配置：

```bash
export LMDEPLOY_PD_PREFILL_URL="http://10.102.98.154:9001"
export LMDEPLOY_PD_DECODE_URL="http://10.102.98.154:9002"
pytest -v py_test/integration/lmdeploy/test_registration.py
```

只配置 Hybrid 后端而未配置两个 PD 变量时，两个 PD 真实用例 skip。完整套件共 37 个
pytest case：28 个 REST、5 个负载均衡策略和 4 个注册/PD 场景。

---

## 8. 测试结果汇总

### 8.1 单元测试

| 项目 | 结果 |
|------|------|
| `cargo check --lib` | PASS |
| `cargo test --lib` | 524 passed, 0 failed |
| `cargo build --release --bin lmdeploy-router` | PASS |

### 8.2 集成测试

2026-08-04 在 H200 节点上使用两个 Hybrid 服务（9000/9003）和一组 DLSlime PD 服务
（Prefill 9001、Decode 9002）执行完整命令，结果为 **37 passed in 14.61s**。PD 使用
RDMA/RoCE、`SLIME_VISIBLE_DEVICES=mlx5_0,...,mlx5_7` 和 `SLIME_GID_INDEX=3`。四个
LMDeploy 日志均未出现 `ERROR`、`Traceback` 或 `Invalid Free session`。
容器仍会提示当前 Transformers 5.12.1 超出 LMDeploy 声明的 `[4.33.0, 5.3.0)` 范围；
本轮功能测试未受影响，但正式环境应将依赖版本调整到 LMDeploy 支持范围。

| 项目 | 结果 |
|------|------|
| 5 LB 策略测试 | 5/5 PASS |
| 28 适配测试 | 28/28 PASS |
| Hybrid 动态注册 | PASS（零 worker 启动、role=1 注册、真实推理） |
| PD 动态注册 | PASS（role=2/3 注册、Prefill → RDMA/RoCE → Decode 真实推理） |
| 注册与 PD 场景 | 4/4 PASS（静态/动态、幂等、查询、注销、非法 role、PD 不支持端点） |
| Token-in-token-out | PASS（input_ids 透传 + output_ids 返回） |
| 严格内容匹配 | PASS（temp=0 下 router == 直连后端） |
| logprobs | PASS（`--logprobs-mode raw_logprobs`） |

### 8.3 发现并修复的 Bug

| Bug | 文件 | 修复 |
|-----|------|------|
| `GenerateRequest` 缺少 `#[serde(flatten)] other` 字段，导致 lmdeploy /generate 参数被静默丢弃 | `src/protocols/spec.rs:1993` | 添加 `#[serde(flatten)] pub other: serde_json::Map<String, serde_json::Value>` |

---

## 9. 完整复现步骤

```bash
# ========== 1. 安装 Rust ==========
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# ========== 2. 进入仓库并配置相对目录 ==========
cd router
export ROUTER_ROOT="$(pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
export CARGO_HTTP_MULTIPLEXING=false
export LMDEPLOY_ROUTER_BIN="${CARGO_TARGET_DIR}/release/lmdeploy-router"
export LMDEPLOY_TEST_ARTIFACT_DIR="${LMDEPLOY_TEST_ARTIFACT_DIR:-.test-artifacts/lmdeploy}"
mkdir -p "${LMDEPLOY_TEST_ARTIFACT_DIR}"

# ========== 3. 编译 ==========
python -m pip install -e '.[dev]'
cargo check --lib
cargo test --lib
cargo build --release --bin lmdeploy-router

# ========== 4. 启动后端（两个 lmdeploy 实例） ==========
# Backend 1: 10.102.98.154:9000
# Backend 2: 10.102.98.154:9003
# 模型: qwen3-06b
# 需开启 --logprobs-mode raw_logprobs

# ========== 5. 配置测试后端 ==========
export LMDEPLOY_BACKEND_URLS="http://10.102.98.154:9000,http://10.102.98.154:9003"
# 可选：后端暴露多个模型时指定；否则从 /v1/models 自动发现
# export LMDEPLOY_MODEL="qwen3-06b"

# 可选：启用真实 PD 动态注册与推理测试
export LMDEPLOY_PD_PREFILL_URL="http://10.102.98.154:9001"
export LMDEPLOY_PD_DECODE_URL="http://10.102.98.154:9002"

# ========== 6. 运行全部 LMDeploy pytest ==========
# fixture 自动启动/停止 Router，并为每个实例分配空闲端口
pytest -v py_test/integration/lmdeploy
```

---

## 10. Python wheel 打包与安装

### 10.1 打包方式

`lmdeploy-router` 由仓库根目录的 `pyproject.toml` 与 `setup.py` 通过 setuptools-rust 打包
成一个 wheel（与上游 vllm-router 的打包方式一致）。该 wheel 同时包含三部分：

- `lmdeploy_router_rs`：PyO3 扩展模块（`src/lib.rs`），提供 `PolicyType` 与 `Router`。
- `lmdeploy_router`：纯 Python 包（`py_src/lmdeploy_router/`），提供 `RouterArgs`、
  `launch_router` 和 MiniLB。
- `lmdeploy-router`：入口命令，等价于 `python -m lmdeploy_router.launch_router`。

因此安装同一个 wheel 之后，既可以 `import lmdeploy_router_rs`，也可以直接运行
`lmdeploy-router` 命令；二者参数完全一致。PyO3 扩展按 abi3 构建
（`py_limited_api = "cp38"`），同一个 `cp38-abi3` wheel 可用于 CPython 3.8 及以上版本。
不同操作系统或 CPU 架构仍需分别构建。

### 10.2 构建 wheel

```bash
python -m pip install build 'setuptools-rust>=1.5.2'
python -m build --wheel --outdir dist .
```

产物统一放在仓库根目录的 `dist/` 中：

```text
dist/lmdeploy_router-0.0.3-cp38-abi3-linux_x86_64.whl
```

Rust 依赖较多，首次构建需要数分钟；通过 `CARGO_HOME` 复用缓存可以显著加速。`dist/` 已加入
`.gitignore`，适合作为本地或 CI 构建产物目录。正式发布时应将 wheel 上传到内部 PyPI、
制品库或 Release，而不是提交进 Git。wheel 文件名遵循 Python Wheel 规范，不要将其重命名为
`lmdeploy-router.whl`；pip 会校验文件名中的包名、版本和兼容性标签。

`pyproject.toml`、`Cargo.toml` 与 `py_src/lmdeploy_router/version.py` 的版本号必须保持一致。

### 10.3 安装与使用

在目标 Python 环境中安装构建出的 wheel：

```bash
python -m pip install ./dist/lmdeploy_router-0.0.3-cp38-abi3-linux_x86_64.whl
lmdeploy-router --help
python -c "from lmdeploy_router_rs import PolicyType; print(PolicyType)"
```

入口命令走的是 Python `RouterArgs`，它与 Rust CLI 接受同一套 PD 参数：
`--lmdeploy-pd-disaggregation`、`--lmdeploy-migration-protocol`、
`--lmdeploy-rdma-link-type`、`--lmdeploy-disable-gdr` 和 `--lmdeploy-dummy-prefill`。
启动参数与直接运行 Cargo release 二进制完全相同，例如先启动 Router、等待 LMDeploy 服务
通过 `--proxy-url` 动态注册：

```bash
lmdeploy-router \
  --host 0.0.0.0 \
  --port 30000 \
  --lmdeploy-pd-disaggregation \
  --lmdeploy-migration-protocol rdma \
  --lmdeploy-rdma-link-type roce
```

### 10.4 验证多 Python 版本

扩展为 abi3，同一个 wheel 可直接安装在 CPython 3.8 及以上的环境中：

```bash
WHEEL=dist/lmdeploy_router-0.0.3-cp38-abi3-linux_x86_64.whl

for PYTHON in python3.10 python3.11 python3.12; do
  ENV_DIR=".test-artifacts/wheel/${PYTHON}"
  "${PYTHON}" -m venv "${ENV_DIR}"
  "${ENV_DIR}/bin/python" -m pip install --no-deps "${WHEEL}"
  "${ENV_DIR}/bin/lmdeploy-router" --help
done
```

除安装检查外，发布流水线还应在目标 Linux 发行版上运行一次真实的 Router 启动和请求
转发测试。wheel 的平台标签保证的是 Linux 基础系统兼容范围，不代替运行时功能验证。

### 10.5 在 caikun-lmdeploy 容器中验收

目标容器当前为 Linux x86_64、Python 3.12，安装及启动验证可执行：

```bash
WHEEL=dist/lmdeploy_router-0.0.1-py3-none-manylinux_2_17_x86_64.whl
WHEEL_NAME="$(basename "${WHEEL}")"
TARGET=gpu-l-lg-cmc-h-h200-0382.host.h.pjlab.org.cn

scp "${WHEEL}" "ailab@${TARGET}:/tmp/"
ssh "ailab@${TARGET}" \
  "docker cp /tmp/${WHEEL_NAME} caikun-lmdeploy:/tmp/${WHEEL_NAME} && \
   docker exec caikun-lmdeploy python3 -m pip install --no-deps --force-reinstall /tmp/${WHEEL_NAME} && \
   docker exec caikun-lmdeploy lmdeploy-router --version"
```

验收时还应使用待部署的 LMDeploy 参数短暂启动 Router，并请求 `/list_workers` 或 `/health`
确认服务监听成功。不要只检查 pip 的安装返回码。

---

## 附录：文件索引

| 文件 | 说明 |
|------|------|
| `src/config/types.rs` | LMDeployMigrationProtocol, LMDeployRdmaConfig, RoutingMode::LMDeployPrefillDecode |
| `src/config/validation.rs` | LMDeployPrefillDecode 验证分支 |
| `src/routers/http/lmdeploy_pd_router.rs` | LMDeployPDRouter 实现 |
| `src/routers/http/router.rs` | 普通模式动态 worker 注册 |
| `src/routers/http/pd_router.rs` | Prefill/Decode 动态 worker 注册 |
| `src/routers/factory.rs` | create_lmdeploy_pd_router 工厂方法 |
| `src/routers/http/mod.rs` | 模块注册 |
| `src/main.rs` | CLI flags + mode 分支 |
| `src/server.rs` | `/nodes/add`、`/nodes/remove`、`/nodes/status` 注册接口 |
| `src/lib.rs` | PyO3 Router 结构 |
| `src/protocols/spec.rs` | typed input_ids/output_ids + GenerateRequest flatten |
| `tests/lmdeploy_registration_test.rs` | 普通/PD 动态注册 Rust 回归测试 |
| `py_test/integration/lmdeploy/conftest.py` | 真实后端配置、Router 生命周期和测试产物 fixture |
| `py_test/integration/lmdeploy/test_load_balancing.py` | 五种 LB 策略测试 |
| `py_test/integration/lmdeploy/test_rest_api.py` | 28 项 LMDeploy REST 适配测试 |
| `py_test/integration/lmdeploy/test_registration.py` | Hybrid/PD 动态注册测试 |
| `pyproject.toml` + `setup.py` | Python wheel 打包配置（setuptools-rust） |
| `py_src/lmdeploy_router/` | Python 包：`RouterArgs`、`launch_router`、MiniLB |
| `${CARGO_TARGET_DIR}/release/lmdeploy-router` | 编译产物（release 二进制） |
