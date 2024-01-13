# Triton Inference Server 性能测试方法

Triton Inference Server(TIS) 是 LMDeploy 支持的除了 api_server 之外的另一种 serving 方式。它的性能测试方式和测试指标和 [api_server](./profile_api_server.md) 的测试方式类似。

```{note}
LMDeploy 尚未实现 Triton Inference Server 的 ensemble 推理模式，所以推理性能要比 api_server 弱。对于追求性能的用户，我们推荐使用 api_server 部署服务。
```

TIS 性能测试脚本是 `profile_serving.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

```shell
pip install 'lmdeploy[serve]>=0.1.0a1'
git clone --depth=1 https://github.com/InternLM/lmdeploy
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

测速时，需输入具体的模型。我们推荐把模型下载到本地，并通过 `lmdeploy convert` 把模型转换为 turbomind 格式，然后再进行测试。
这么做的原因是，方便调节推理引擎参数，以达到比较好的推理性能，比如批处理大小（max_batch_size），K/V cache缓存大小（max_cache_entry_count）等等。有关这些参数的详细说明，请参考[这里](../inference/turbomind_config.md).

以下章节中，我们默认模型是 turbomind 格式的。

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

## 测试案例

我们用 `internlm-7b` 为例，api_server的速度测试全流程如下：

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 从huggingface下载internlm-7b，并转为turbomind模型格式
lmdeploy convert internlm internlm/internlm-7b --dst-path ./internlm-7b

# 启动server
bash ./internlm-7b/service_docker_up.sh

# 另起终端，在`lmdeploy/benchmark`目录下，执行测速脚本
python3 ./profile_serving.py 0.0.0.0:33337 ./internlm-7b/triton_models/tokenizer ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测试方法

启动服务

```shell
python3 profile_restful_api.py <server_addr> <tokenizer_path> <dataset> <optional arguments>
```

其中，必填参数是：

- `server_addr`

  api_server 的地址，格式是 `{server_ip}:{server_port}`

- `tokenizer_path`

  tokenizer model 的路径。作用是对测试数据集预先 encode，获取对话数据的 token 长度

- `dataset`

  下载的测试数据集的路径

可选测试参数如下：

- `--concurrency`

  客户端请求线程的数量，并发请求会被推理引擎拼成 batch，默认为 32。并发请求会被推理引擎拼成 batch。建议 concurrency 的值不要超过推理引擎的 `max_batch_size`，也不要超过 triton_models 中的推理实例的数量。
  推理实例数量的配置项是 `instance_group`，在文件 `{model_path}/triton_models/interactive/config.pbtxt` 里，默认是 48。

- `--num-prompts`

  从数据集中采样的prompt数量，默认是 1000

- `--top_k`、`--top_p` 和 `--temperature`

  这三个参数用来采样生成的 token_id

- `--stream_output`

  流式推理的开关。默认值为 `False`

- `--csv`

  一个 csv 文件路径，用来存放测试结果。默认是 `./profile_tis.csv`

- `--seed`

  从测试数据集中随机采样prompt时的种子。默认为0
