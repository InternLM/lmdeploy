# KV Cache 量化

LMDeploy 最新 main 分支支持在线 kv cache 4bit/8bit 量化，量化方式为 per-head per-token 的非对称量化。原来的 kv 离线量化方式移除。

直观上看，量化 kv 利于降低内存占用量。和 fp16 相比，4bit/8bit kv 的内存可以分别减到 1/4 和 1/2。这意味着，在相同的内存条件下，kv 量化后，系统能支撑的并发数可以大幅提升，从而最终提高吞吐量。

但是，通常，量化会伴随一定的模型精度损失。我们使用了 opencompass 评测了若干个模型在应用了 kv 8/4bit 量化后的精度，结果放在了[精度评测](#精度评测)章节中。大家可以参考，根据实际需求酌情选择。

LMDeploy kv 4/8 bit 量化和推理支持如下 NVIDIA 显卡型号：

- volta 架构（sm70）： V100
- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm89）：40 系列

接下来，我们以 internlm2-chat-7b 模型为例，介绍 kv 量化和推理的若干应用。而在此之前，请首先参考[文档](https://lmdeploy.readthedocs.io/en/latest/build.html)，源码安装 lmdeploy，因为 kv cache 4bit/8bit 在线量化尚未发版。

## 应用示例

通过 LMDeploy 应用 kv 量化非常简单，只需要设定 `quant_policy` 参数。

**LMDeploy 规定 `qant_policy=4` 表示 kv 4bit 量化，`quant_policy=8` 表示 kv 8bit 量化。**

### 离线推理 pipeline

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(quant_policy=8)
pipe = pipeline("internlm/internlm2-chat-7b", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### 在线推理服务

```shell
lmdeploy serve api_server internlm/internlm2-chat-7b --quant-policy 8
```

## 精度评测

我们使用 opencompass 评测 lmdeploy kv 量化应用在若干模型上的推理精度，结果如下表所示：

| -           | -       | -             | llama2-7b-chat |         |         | internlm2-chat-7b |         |         | qwen-chat-7b |         |         |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------ | ------- | ------- |
| dataset     | version | metric        | fp16           | kv int8 | kv int4 | fp16              | kv int8 | kv int4 | bf16         | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 28.07   | 28.18   | 60.45             | 60.48   | 58.91   | 59.32        | 59.59   | 59.42   |
| mmlu        | -       | naive_average | 35.61          | 35.63   | 35.11   | 63.92             | 63.78   | 63.29   | 57.27        | 57.39   | 56.07   |
| triviaqa    | 2121ce  | score         | 56.12          | 56.04   | 54.09   | 58.76             | 58.67   | 58.32   | 54.42        | 54.27   | 54.46   |
| gsm8k       | 1d7fe4  | accuracy      | 28.35          | 28.05   | 25.17   | 70.58             | 70.36   | 66.34   | 53.53        | 52.69   | 53.07   |
| race-middle | 9a54b6  | accuracy      | 41.64          | 42.13   | 45.33   | 88.93             | 88.79   | 88.86   | 83.7         | 83.57   | 82.94   |

具体的评测方式可以参考[这份指南](https://lmdeploy.readthedocs.io/en/latest/benchmark/evaluate_with_opencompass.html)。评测时，请在config文件中，为推理引擎添加 `quant_policy` 参数。

## 推理速度

TODO
