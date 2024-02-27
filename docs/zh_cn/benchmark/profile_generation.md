# 静态推理性能测试

我们把推理引擎在固定 batch、固定输入输出 token 数量的前提下的推理，称之为静态推理。

评测脚本是 `profile_generation.py`，在运行此脚本前，请安装 lmdeploy 预编译包，并下载评测脚本

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

## 测量指标

LMDeploy 统计首token延时（first_token_latency）、token 吞吐量（tokens/s），每个token延时的百分位数据（P50，P75，P95，P99）、GPU mem 等测试结果。

`first_token_latency` 只有在流式推理的情况下才会输出。

吞吐量的计算公式为：

$$
token吞吐量 = 生成的token数量 / 总时间
$$

总时间包括 prefill 时间。

测试过程中，节点上所有的显卡不要运行其他任何程序，否则 GPU mem 的统计会不准确。

## 测量方法

我们以 [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) 为例，分别介绍测试 LMDeploy 两个推理引擎 turbomind 和 pytorch 的静态推理性能测试方法

### Turbomind 引擎

```shell
cd lmdeploy/benchmark
python3 profile_generation.py internlm/internlm-7b
```

### PyTorch 引擎

```shell
cd lmdeploy/benchmark
python3 profile_generation.py internlm/internlm-7b --backend pytorch
```

关于 `profile_generation` 脚本的参数，比如批处理大小，输入输出token的数量等等，可以通过运行命令 `python3 profile_generation.py -h` 查阅。
