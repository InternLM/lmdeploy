# api_server 性能测试

api_server 的测试方式与[求吞吐量测试方法](./profile_throughput.md)类似。不同的是，在测试前，需要先启动 api_server，然后再通过测试脚本发送请求进行测试。

测试脚本是 `profile_restful_api.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

```shell
pip install 'lmdeploy[serve]>=0.1.0a0'
git clone --depth=1 https://github.com/InternLM/lmdeploy
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

测速时，需输入具体的模型。我们推荐把模型下载到本地，并通过 `lmdeploy convert` 把模型转换为 turbomind 格式，然后再进行测试。
这么做的原因是，方便调节推理引擎参数，以达到比较好的推理性能，比如批处理大小（max_batch_size），K/V cache缓存大小（max_cache_entry_count）等等。有关这些参数的详细说明，请参考[这里](../turbomind_config.md).

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

## 测试方法

请参考[这里](../restful_api.md) 启动推理服务。启动时的参数 `--instance-num` 表示推理服务中的推理实例数量。当同一时刻到达 api_server 的请求数超过它时，请求会在推理队列中等待。

```shell
python3 profile_restful_api.py <server_addr> <tokenizer_path> <dataset> <optional arguments>
```

其中，必填参数是：

- `server_addr`

  api_server 的地址，格式是 `http://{server_ip}:{server_port}`

- `tokenizer_path`

  tokenizer model 的路径。作用是对测试数据集预先 encode，获取对话数据的 token 长度

- `dataset`

  下载的测试数据集的路径

可选测试参数如下：

- `--concurrency`

  客户端请求线程的数量，并发请求会被推理引擎拼成 batch，默认为 64。并发请求会被推理引擎拼成 batch。并发数不能超过api_server的`--instance-num`。否则，超出部分的请求会在推理队列中等待。

- `--num-prompts`

  从数据集中采样的prompt数量，默认是 2000

- `--top_p` 和 `--temperature`

  这三个参数用来采样生成的 token_id

- `--stream_output`

  流式推理的开关。默认值为 `False`

- `--csv`

  一个 csv 文件路径，用来存放测试结果。默认是 `./profile_api_server.csv`

- `--seed`

  从测试数据集中随机采样prompt时的种子。默认为0
