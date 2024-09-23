# 请求分发服务器

请求分发服务可以将多个 api_server 服务，进行并联。用户可以只需要访问代理 URL，就可以间接访问不同的 api_server 服务。代理服务内部会自动分发请求，做到负载均衡。

## 启动

启动代理服务：

```shell
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --strategy "min_expected_latency"
```

启动成功后，代理服务的 URL 也会被脚本打印。浏览器访问这个 URL，可以打开 Swagger UI。
随后，用户可以在启动 api_server 服务的时候，通过 `--proxy-url` 命令将其直接添加到代理服务中。例如：`lmdeploy serve api_server InternLM/internlm2-chat-1_8b --proxy-url http://0.0.0.0:8000`。
这样，用户可以通过代理节点访问 api_server 的服务，代理节点的使用方式和 api_server 一模一样，都是兼容 OpenAI 的形式。

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

## 分发策略

代理服务目前的分发策略如下：

- random： 根据用户提供的各个 api_server 节点的处理请求的能力，进行有权重的随机。处理请求的吞吐量越大，就越有可能被分配。部分节点没有提供吞吐量，将按照其他节点的平均吞吐量对待。
- min_expected_latency： 根据每个节点现有的待处理完的请求，和各个节点吞吐能力，计算预期完成响应所需时间，时间最短的将被分配。未提供吞吐量的节点，同上。
- min_observed_latency： 根据每个节点过去一定数量的请求，处理完成所需的平均用时，用时最短的将被分配。
