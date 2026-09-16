# Request Router Server

LMDeploy uses the Rust-based `lmdeploy-router` to distribute requests across multiple `api_server` services. Users access the router URL, and the router forwards requests to available `api_server` instances with load balancing.

## Startup

Install the router package, then start the router with the compatibility `lmdeploy serve proxy` entry point:

```shell
pip install lmdeploy-router
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "cache_aware" --serving-strategy Hybrid
```

The legacy `min_expected_latency` and `min_observed_latency` strategies remain accepted and are mapped to `cache_aware`. For the full router configuration, use the standalone `lmdeploy-router` CLI described in [`lmdeploy/router/README.md`](../../../lmdeploy/router/README.md).

After startup, register workers by starting `api_server` with `--proxy-url`, or use the node-management APIs described below.
Subsequently, users can add it directly to the proxy service when starting the `api_server` service by using the `--proxy-url` command. For example:
`lmdeploy serve api_server InternLM/internlm2-chat-1_8b --proxy-url http://0.0.0.0:8000`。
In this way, users can access the services of the `api_server` through the proxy node, and the usage of the proxy node is exactly the same as that of the `api_server`, both of which are compatible with the OpenAI format.

- /v1/models
- /v1/chat/completions
- /v1/completions

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
  'http://localhost:8000/nodes/remove?node_url=http://0.0.0.0:23333' \
  -H 'accept: application/json' \
  -d ''
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
headers = {'accept': 'application/json',}
params = {'node_url': 'http://0.0.0.0:23333',}
response = requests.post(url, headers=headers, data='', params=params)
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
