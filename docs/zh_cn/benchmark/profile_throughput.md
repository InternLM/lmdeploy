# 测量请求吞吐量

在真实应用中，用户输入的 prompt 长度以及模型回复的 token 数量是动态变化的。而静态推理能力不足以反映推理引擎对动态输入输出的处理能力。

所以需要使用真实对话数据，评测推理引擎的动态推理能力。本文将介绍如何在 localhost 上测试 LMDeploy 的动态推理性能。

测试脚本是 `profile_throughput.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

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
token吞吐量 = 生成的token数量 / 总时间
$$

请求吞吐量的计算公式为：

$$
吞吐量 = 请求数量 / 总时间
$$

总时间包括 prefill 时间

## 测量方法

我们以 [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) 为例，分别介绍测试 LMDeploy 两个推理引擎 turbomind 和 pytorch 的离线请求处理速度

### Turbomind 引擎

```shell
python3 profile_throughput.py ./ShareGPT_V3_unfiltered_cleaned_split.json internlm/internlm-7b
```

### PyTorch 引擎

```shell
python3 profile_throughput.py ./ShareGPT_V3_unfiltered_cleaned_split.json internlm/internlm-7b  --backend pytorch
```

有关 profile_throughput.py 的详细参数，比如并发数、采样参数、k/v内存分配比例等等，请执行 `python3 profile_throughput.py -h` 查阅
