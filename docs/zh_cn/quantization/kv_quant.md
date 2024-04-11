# Key-Value(KV) Cache 量化

LMDeploy 最新 main 分支支持在线 kv cache 4bit/8bit 量化，量化方式为 per-head per-token 的非对称量化。原来的 kv 离线量化方式移除。

直观上看，量化 kv 利于降低内存占用量。和 fp16 相比，4bit/8bit kv 的内存可以分别减到 1/4 和 1/2。这意味着，在相同的内存条件下，kv 量化后，系统能支撑的并发数可以大幅提升，从而最终提高吞吐量。

但是，通常，量化会伴随一定的模型精度损失。我们使用了 opencompass 评测了若干个模型在应用了 kv 8/4bit 量化后的精度，结果放在了[精度评测](#精度评测)章节中。大家可以参考，根据实际需求酌情选择。

LMDeploy kv 4/8 bit 量化和推理支持如下 NVIDIA 显卡型号：

- volta 架构（sm70）： V100
- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm89）：40 系列
- Hopper 架构（sm90）: H100, H200

接下来，我们以 internlm2-chat-7b 模型为例，介绍 kv 量化和推理的若干应用。而在此之前，请首先参考[文档](https://lmdeploy.readthedocs.io/en/latest/build.html)，源码安装 lmdeploy，因为 kv cache 4bit/8bit 在线量化尚未发版。

## 应用示例

通过 LMDeploy 应用 kv 量化非常简单，只需要设定 `quant_policy` 参数。

**LMDeploy 规定 `qant_policy=4` 表示 kv 4bit 量化，`quant_policy=8` 表示 kv 8bit 量化。**

### 离线推理

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(quant_policy=8)
pipe = pipeline("internlm/internlm2-chat-7b", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### 推理服务

```shell
lmdeploy serve api_server internlm/internlm2-chat-7b --quant-policy 8
```

## 精度评测

我们把 lmdeploy 的 kv 量化应用在若干 LLM 模型上，并使用 opencompass 评测推理精度，结果如下表所示：

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | qwen-chat-7b | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------ | ------- | ------- |
| dataset     | version | metric        | fp16           | kv int8 | kv int4 | fp16              | kv int8 | kv int4 | fp16         | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 28.38   | 27.18   | 60.45             | 60.71   | 59.8    | 59.34        | 60.05   | 60.77   |
| mmlu        | -       | naive_average | 35.58          | 35.58   | 34.94   | 63.92             | 64      | 62.63   | 57.45        | 57.41   | 56.39   |
| triviaqa    | 2121ce  | score         | 56.13          | 56.08   | 53.79   | 58.74             | 58.69   | 57.87   | 54.07        | 54.05   | 53.64   |
| gsm8k       | 1d7fe4  | accuracy      | 28.28          | 28.43   | 26.54   | 70.58             | 69.75   | 68.08   | 53.53        | 53.22   | 52.69   |
| race-middle | 9a54b6  | accuracy      | 41.64          | 41.78   | 42.41   | 88.93             | 88.86   | 89.28   | 83.15        | 83.08   | 83.29   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.51   | 40.65   | 85.28             | 85.31   | 84.05   | 76.67        | 76.76   | 77.36   |

具体的评测方式可以参考[这份指南](../benchmark/evaluate_with_opencompass.md)。评测时，请在config文件中，为推理引擎添加 `quant_policy` 参数。

## 推理速度

| model             | kv type | test settings                         | RPS   | v.s. kv fp16 |
| ----------------- | ------- | ------------------------------------- | ----- | ------------ |
| llama2-chat-7b    | fp16    | tp1/ratio 0.8 / bs 256/ prompts 10000 | 14.98 | 1.0          |
| -                 | kv8     | tp1/ratio 0.8 / bs 256/ prompts 10000 | 19.01 | 1.27         |
| -                 | kv4     | tp1/ratio 0.8 / bs 256/ prompts 10000 | 20.81 | 1.39         |
| llama2-chat-13b   | fp16    | tp1/ratio 0.9 / bs 128/ prompts 10000 | 8.55  | 1.0          |
| -                 | kv8     | tp1/ratio 0.9 / bs 128/ prompts 10000 | 9.87  | 1.15         |
| -                 | kv4     | tp1/ratio 0.9 / bs 128/ prompts 10000 | 10.65 | 1.25         |
| internlm2-chat-7b | fp16    | tp1/ratio 0.8 / bs 256/ prompts 10000 | 24.13 | 1.0          |
| -                 | kv8     | tp1/ratio 0.8 / bs 256/ prompts 10000 | 25.28 | 1.05         |
| -                 | kv4     | tp1/ratio 0.8 / bs 256/ prompts 10000 | 25.80 | 1.07         |

上述结果使用的测试脚本是 `benchmark/profile_throughput.py`
