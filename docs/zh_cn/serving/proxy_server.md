# 请求分发服务器

请求分发服务可以将多个 api_server 服务，进行并联。用户可以只需要访问代理 URL，就可以间接访问不同的 api_server 服务。代理服务内部会自动分发请求，做到负载均衡。

## 启动

启动代理服务：

```shell
python3 -m lmdeploy.serve.proxy --server_name {server_name} --server_port {server_port} --strategy "min_expected_latency"
```

启动成功后，代理服务的 URL 也会被脚本打印。浏览器访问这个 URL，可以打开 Swagger UI。

## API

通过 Swagger UI，我们可以看到多个 API。其中，和 api_server 节点管理相关的有：

- /nodes/status
- /nodes/add
- /nodes/remove

他们分别表示，查看所有的 api_server 服务节点，增加某个节点，删除某个节点。

和使用相关的 api 有：

- /v1/models
- /v1/chat/completions
- /v1/completions

这些 API 的使用方式和 api_server 一样。

## 分发策略

代理服务目前的分发策略如下：

- random： 根据用户提供的各个 api_server 节点的处理请求的能力，进行有权重的随机。处理请求的吞吐量越大，就越有可能被分配。部分节点没有提供吞吐量，将按照其他节点的平均吞吐量对待。
- min_expected_latency： 根据每个节点现有的待处理完的请求，和各个节点吞吐能力，计算预期完成响应所需时间，时间最短的将被分配。未提供吞吐量的节点，同上。
- min_observed_latency： 根据每个节点过去一定数量的请求，处理完成所需的平均用时，用时最短的将被分配。
