# api_server 性能测试

api_server 的测试方式与[求吞吐量测试方法](./profile_throughput.md)类似。不同的是，在测试前，需要先启动 api_server，然后再通过测试脚本发送请求进行测试。

测试脚本是 `profile_restful_api.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测量指标

LMDeploy 统计首token延时（first_token_latency）、token吞吐量（tokens/s）和请求吞吐量（RPM）。

`first_token_latency` 只有在流式推理的情况下才会输出。

token吞吐量的计算公式为：

$$
吞吐量 = 生成的token数量 / 总时间
$$

请求吞吐量的计算公式为：

$$
吞吐量 = 请求数量 / 总时间
$$

总时间包括 prefill 时间

## 测量方法

我们以 [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) 为例，展示 api_server 的性能测试流程

### 启动服务

```shell
lmdeploy serve api_server internlm/internlm-7b
```

如果你想改变 server 的端口，或者诸如推理引擎、最大批处理值等参数，请运行 `lmdeploy serve api_server -h` 或者阅读[这篇文档](../serving/restful_api.md)，查看详细的参数说明。

### 测速

```shell
python3 profile_restful_api.py http://0.0.0.0:23333 internlm/internlm-7b ./ShareGPT_V3_unfiltered_cleaned_split.json
```

关于 `profile_restful_api.py` 脚本中的参数，比如请求并发数、采样参数等等，可以通过运行命令 `python3 profile_restful_api.py -h` 查阅。
