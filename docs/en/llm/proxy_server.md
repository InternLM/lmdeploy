# Request Router Server

LMDeploy uses the Rust-based `lmdeploy-router` to distribute requests across multiple `api_server` services. Users access the router URL, and the router forwards requests to available `api_server` instances with load balancing.

## Startup

Install the router package, then start the router with the compatibility `lmdeploy serve proxy` entry point:

```shell
pip install lmdeploy-router
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "cache_aware" --serving-strategy Hybrid
```

The legacy `min_expected_latency` and `min_observed_latency` strategies remain accepted and are mapped to `cache_aware`. For the full router configuration, use the standalone `lmdeploy-router` CLI described in [`lmdeploy/router/README.md`](../../../lmdeploy/router/README.md).

After startup, register workers by starting `api_server` with `--proxy-url`, or use the node-management APIs described below:

```shell
lmdeploy serve api_server InternLM/internlm2-chat-1_8b \
    --server-name 127.0.0.1 \
    --server-port 23333 \
    --proxy-url http://127.0.0.1:8000
```

Clients send OpenAI-compatible requests to the router instead of an `api_server`:

- `/health`
- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`

## Node Management

Through Swagger UI, we can see multiple APIs. Those related to api_server node management include:

- /nodes/status
- /nodes/add
- /nodes/remove

They respectively represent viewing all api_server service nodes, adding a certain node, and deleting a certain node.

### Node Management through curl

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

### Node Management through python

```python
# query all nodes
import requests
url = 'http://localhost:8000/nodes/status'
headers = {'accept': 'application/json'}
response = requests.get(url, headers=headers)
print(response.text)
```

```python
# add a new node
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
# delete a node
import requests
url = 'http://localhost:8000/nodes/remove'
headers = {'accept': 'application/json'}
data = {'url': 'http://127.0.0.1:23333'}
response = requests.post(url, headers=headers, json=data)
print(response.text)
```

## Serving Strategy

LMDeploy currently supports two serving strategies:

- Hybrid: Does not distinguish between Prefill and Decoding instances, following the traditional inference deployment mode.
- DistServe: Separates Prefill and Decoding instances, deploying them on different service nodes to achieve more flexible and efficient resource scheduling and scalability.

## Dispatch Strategy

The compatibility entry point supports the following router strategies:

- `random`: selects a healthy worker at random.
- `round_robin`: selects healthy workers in round-robin order.
- `cache_aware`: routes requests with shared prefixes to the worker with the most useful cache state.
- `power_of_two`: samples two healthy workers and selects the less-loaded one.
- `consistent_hash`: routes requests with the same routing key to the same worker.
- `rendezvous_hash`: scores every healthy worker with the same routing key and selects the highest score; it preserves session affinity with near-even distribution.

## PD Disaggregation

For DistServe, start the router first, then register Prefill and Decode services. The router expects
Prefill workers to handle the cache-prefill phase and Decode workers to return the OpenAI-compatible
response. It establishes the required P2P connection between each pair automatically.

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

RDMA, DLSlime, and Mooncake infrastructure must be provisioned outside the router.

## Migration From the Legacy Proxy

`lmdeploy serve proxy` remains a compatibility entry point that starts `lmdeploy-router`.

| Legacy command                                               | Router command                            |
| ------------------------------------------------------------ | ----------------------------------------- |
| `lmdeploy serve proxy --server-name HOST --server-port PORT` | `lmdeploy-router --host HOST --port PORT` |
| `--routing-strategy POLICY`                                  | `--policy POLICY`                         |
| `--serving-strategy Hybrid`                                  | Omit `--lmdeploy-pd-disaggregation`       |
| `--serving-strategy DistServe`                               | Add `--lmdeploy-pd-disaggregation`        |
| `--migration-protocol RDMA`                                  | `--lmdeploy-migration-protocol rdma`      |
| `--migration-protocol NVLINK`                                | `--lmdeploy-migration-protocol nvlink`    |
| `--link-type RoCE`                                           | `--lmdeploy-rdma-link-type roce`          |
| `--link-type IB`                                             | `--lmdeploy-rdma-link-type ib`            |

The legacy `min_expected_latency` and `min_observed_latency` values are accepted only by the compatibility
entry point; both map to `cache_aware`.

The old Python implementation in `lmdeploy.serve.proxy` was removed. Importing its internal modules,
including `lmdeploy.serve.proxy.proxy`, `lmdeploy.serve.proxy.utils`, and
`lmdeploy.serve.proxy.streaming_response`, is no longer supported. Use `lmdeploy-router` or the public
`lmdeploy_router` Python package instead.
