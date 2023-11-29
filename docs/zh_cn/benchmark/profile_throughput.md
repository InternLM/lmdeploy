# 请求吞吐量测试方法

在真实应用中，用户输入的 prompt 长度以及模型回复的 token 数量是动态变化的。而静态推理能力不足以反映推理引擎对动态输入输出的处理能力。

所以需要使用真实对话数据，评测推理引擎的动态推理能力。本文将介绍如何在 localhost 上测试 LMDeploy 的动态推理性能。

测试脚本是 `profile_restful_api.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

```shell
pip install 'lmdeploy>=0.1.0a0'
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
token吞吐量 = 生成的token数量 / 总时间
$$

请求吞吐量的计算公式为：

$$
吞吐量 = 请求数量 / 总时间
$$

总时间包括 prefill 时间

## 测试案例

我们用 `internlm-7b` 为例，api_server的速度测试全流程如下：

```shell
pip install 'lmdeploy[serve]>=0.1.0a0'
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 从huggingface下载internlm-7b，并转为turbomind模型格式
lmdeploy convert internlm-7b internlm/internlm-7b --dst-path ./internlm-7b

# 执行测速脚本
python3 profile_throughput.py ./internlm-7b ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测试方法

```shell
python3 profile_throughput.py <dataset> <model_path> <optional arguments>
```

其中，必填参数是：

- `dataset`

  测试数据集的路径

- `model_path`

  turbomind格式的模型在 localhost 上的路径

可选测试参数如下：

- `--concurrency`

  代表请求线程的数量，并发请求会被推理引擎拼成 batch，默认为 64。并发请求会被推理引擎拼成 batch。并发数不能超过`config.ini`中的`max_batch_size`。否则，超出部分的请求会在推理队列中等待。

- `--num-prompts`

  从数据集中采样的prompt数量。默认是 2000

- `--tp`

  模型在张量并行时，使用的显卡数量。必须是2的整数次幂。默认为 1

- `--top_k`、`--top_p` 和 `--temperature`

  这三个参数用来采样生成的 token_id

- `--stream_output`

  流式推理的开关。默认值为 `True`

- `--csv`

  一个 csv 文件路径，用来存放测试结果。默认是 `./profile_throughput.csv`

- `--log-level`

  日志级别。默认是 `ERROR`

- `--seed`

  从测试数据集中随机采样prompt时的种子。默认为0
