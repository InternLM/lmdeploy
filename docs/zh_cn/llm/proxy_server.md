# 请求路由服务器

LMDeploy 使用基于 Rust 的 `lmdeploy-router` 为多个 `api_server` 服务分发请求。用户只需访问 router URL，router 会按负载均衡策略将请求转发到可用的 `api_server` 实例。

## 启动

安装 router 包后，通过兼容入口 `lmdeploy serve proxy` 启动：

```shell
pip install lmdeploy-router
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "cache_aware" --serving-strategy Hybrid
```

旧的 `min_expected_latency` 和 `min_observed_latency` 策略仍可传入，并会映射为 `cache_aware`。完整 router 配置请使用 [`lmdeploy/router/README.md`](../../../lmdeploy/router/README.md) 中的独立 `lmdeploy-router` CLI。

启动后，可在启动 `api_server` 时通过 `--proxy-url` 注册 worker，也可以使用下文的节点管理接口：

```shell
lmdeploy serve api_server InternLM/internlm2-chat-1_8b \
    --server-name 127.0.0.1 \
    --server-port 23333 \
    --proxy-url http://127.0.0.1:8000
```

客户端直接访问 router 的 OpenAI 兼容接口，而不是某个 `api_server`：

- `/health`
- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`

## 节点管理

通过 Swagger UI，我们可以看到多个 API。其中，和 api_server 节点管理相关的有：

- /nodes/status
- /nodes/add
- /nodes/remove

他们分别表示，查看所有的 api_server 服务节点，增加某个节点，删除某个节点。他们的使用方式，最直接的可以在浏览器里面直接操作。也可以通过命令行或者 python 操作。

### 通过 command 增删查

```shell
curl -X 'GET' \
  'http://localhost:8000/nodes/status' \
  -H 'accept: application/json'
```

```shell
curl -X 'POST' \
  'http://localhost:8000/nodes/add' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "http://0.0.0.0:23333"
}'
```

```shell
curl -X 'POST' \
  'http://localhost:8000/nodes/remove' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"url": "http://127.0.0.1:23333"}'
```

### 通过 python 脚本增删查

```python
# 查询所有节点
import requests
url = 'http://localhost:8000/nodes/status'
headers = {'accept': 'application/json'}
response = requests.get(url, headers=headers)
print(response.text)
```

```python
# 添加新节点
import requests
url = 'http://localhost:8000/nodes/add'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {"url": "http://0.0.0.0:23333"}
response = requests.post(url, headers=headers, json=data)
print(response.text)
```

```python
# 删除某个节点
import requests
url = 'http://localhost:8000/nodes/remove'
headers = {'accept': 'application/json'}
data = {'url': 'http://127.0.0.1:23333'}
response = requests.post(url, headers=headers, json=data)
print(response.text)
```

## 服务策略

LMDeploy 当前支持混合部署服务（Hybrid），以及 PD 分离部署服务（DistServe）

- Hybrid: 不区分 Prefill 和 Decoding 实例，即传统的推理部署模式。
- DistServe: 将 Prefill 和 Decoding 实例分离，部署在不同的服务节点上以实现更灵活高效的资源调度和扩展。

## 分发策略

兼容入口支持以下 router 策略：

- `random`：随机选择健康 worker。
- `round_robin`：按轮询顺序选择健康 worker。
- `cache_aware`：优先将共享前缀的请求路由到缓存命中最多的 worker。
- `power_of_two`：采样两个健康 worker，并选择负载较低者。
- `consistent_hash`：将相同路由键的请求发送到同一 worker。
- `rendezvous_hash`：用同一路由键为每个健康 worker 计算分数并选择最高分；保持会话粘性的同时分布更均匀。

## PD 分离部署

使用 DistServe 时，先启动 router，再注册 Prefill 与 Decode 服务。router 会把缓存 prefill 阶段发给
Prefill worker，把 OpenAI 兼容响应返回给客户端，并自动为每组 Prefill/Decode worker 建立 P2P 连接。

```shell
lmdeploy-router \
    --host 127.0.0.1 \
    --port 8000 \
    --policy power_of_two \
    --lmdeploy-pd-disaggregation
```

```shell
lmdeploy serve api_server InternLM/internlm2-chat-1_8b \
    --server-name 127.0.0.1 --server-port 23333 \
    --role Prefill --proxy-url http://127.0.0.1:8000

lmdeploy serve api_server InternLM/internlm2-chat-1_8b \
    --server-name 127.0.0.1 --server-port 23334 \
    --role Decode --proxy-url http://127.0.0.1:8000
```

RDMA、DLSlime 与 Mooncake 等基础设施仍需在 router 外部准备。

## 从旧 Proxy 迁移

`lmdeploy serve proxy` 是兼容入口，内部启动 `lmdeploy-router`。

| 旧命令                                                       | Router 命令                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| `lmdeploy serve proxy --server-name HOST --server-port PORT` | `lmdeploy-router --host HOST --port PORT` |
| `--routing-strategy POLICY`                                  | `--policy POLICY`                         |
| `--serving-strategy Hybrid`                                  | 省略 `--lmdeploy-pd-disaggregation`       |
| `--serving-strategy DistServe`                               | 增加 `--lmdeploy-pd-disaggregation`       |
| `--migration-protocol RDMA`                                  | `--lmdeploy-migration-protocol rdma`      |
| `--migration-protocol NVLINK`                                | `--lmdeploy-migration-protocol nvlink`    |
| `--link-type RoCE`                                           | `--lmdeploy-rdma-link-type roce`          |
| `--link-type IB`                                             | `--lmdeploy-rdma-link-type ib`            |

旧值 `min_expected_latency` 和 `min_observed_latency` 只在兼容入口中接受，两者都会映射为 `cache_aware`。

旧 Python 实现 `lmdeploy.serve.proxy` 已移除。不再支持导入其内部模块，包括
`lmdeploy.serve.proxy.proxy`、`lmdeploy.serve.proxy.utils` 和
`lmdeploy.serve.proxy.streaming_response`。请改用 `lmdeploy-router`，或使用公开的
`lmdeploy_router` Python 包。
