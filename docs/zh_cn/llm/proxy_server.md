# 请求路由服务器

LMDeploy 使用基于 Rust 的 `lmdeploy-router` 为多个 `api_server` 服务分发请求。用户只需访问 router URL，router 会按负载均衡策略将请求转发到可用的 `api_server` 实例。

## 启动

安装 router 包后，通过兼容入口 `lmdeploy serve proxy` 启动：

```shell
pip install lmdeploy-router
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "cache_aware" --serving-strategy Hybrid
```

旧的 `min_expected_latency` 和 `min_observed_latency` 策略仍可传入，并会映射为 `cache_aware`。完整 router 配置请使用 [`lmdeploy/router/README.md`](../../../lmdeploy/router/README.md) 中的独立 `lmdeploy-router` CLI。

启动后，可在启动 `api_server` 时通过 `--proxy-url` 注册 worker，也可以使用下文的节点管理接口。

- /v1/models
- /v1/chat/completions
- /v1/completions

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
  'http://localhost:8000/nodes/remove?node_url=http://0.0.0.0:23333' \
  -H 'accept: application/json' \
  -d ''
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
headers = {'accept': 'application/json',}
params = {'node_url': 'http://0.0.0.0:23333',}
response = requests.post(url, headers=headers, data='', params=params)
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
