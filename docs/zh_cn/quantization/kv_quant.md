# Key-Value(KV) Cache 量化

LMDeploy 最新 main 分支支持**在线** kv cache int4/int8 量化，量化方式为 per-head per-token 的非对称量化。原来的 kv 离线量化方式移除。

直观上看，量化 kv 利于降低内存占用量。和 fp16 相比，int4/int8 kv 的内存可以分别减到 1/4 和 1/2。这意味着，在相同的内存条件下，kv 量化后，系统能支撑的并发数可以大幅提升，从而最终提高吞吐量。

但是，通常，量化会伴随一定的模型精度损失。我们使用了 opencompass 评测了若干个模型在应用了 int4/int8 量化后的精度，int8 kv 精度几乎无损，int4 kv 略有损失。详细结果放在了[精度评测](#精度评测)章节中。大家可以参考，根据实际需求酌情选择。

LMDeploy kv 4/8 bit 量化和推理支持如下 NVIDIA 显卡型号：

- volta 架构（sm70）： V100
- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm89）：40 系列
- Hopper 架构（sm90）: H100, H200

总结来说，LMDeploy kv 量化具备以下优势：

1. 量化不需要校准数据集
2. 支持 volta 架构（sm70）及以上的所有显卡型号
3. kv int8 量化精度几乎无损，kv int4 量化精度在可接受范围之内
4. 推理高效，在 llama2-7b 上加入 int8/int4 kv 量化，RPS 相较于 fp16 分别提升近 30% 和 40%

接下来，我们以 internlm2-chat-7b 模型为例，介绍 kv 量化和推理的若干应用。而在此之前，请首先参考[文档](https://lmdeploy.readthedocs.io/en/latest/build.html)，源码安装 lmdeploy，因为 kv cache 4bit/8bit 在线量化尚未发版。

## 应用示例

通过 LMDeploy 应用 kv 量化非常简单，只需要设定 `quant_policy` 参数。

**LMDeploy 规定 `qant_policy=4` 表示 kv int4 量化，`quant_policy=8` 表示 kv int8 量化。**

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

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | qwen1.5-7b-chat | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | --------------- | ------- | ------- |
| dataset     | version | metric        | kv fp16        | kv int8 | kv int4 | kv fp16           | kv int8 | kv int4 | fp16            | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 27.96   | 27.58   | 60.45             | 60.88   | 60.28   | 70.56           | 70.49   | 68.62   |
| mmlu        | -       | naive_average | 35.64          | 35.58   | 34.79   | 63.91             | 64      | 62.36   | 61.48           | 61.56   | 60.65   |
| triviaqa    | 2121ce  | score         | 56.09          | 56.13   | 53.71   | 58.73             | 58.7    | 58.18   | 44.62           | 44.77   | 44.04   |
| gsm8k       | 1d7fe4  | accuracy      | 28.2           | 28.05   | 27.37   | 70.13             | 69.75   | 66.87   | 54.97           | 56.41   | 54.74   |
| race-middle | 9a54b6  | accuracy      | 41.57          | 41.78   | 41.23   | 88.93             | 88.93   | 88.93   | 87.33           | 87.26   | 86.28   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.77   | 40.77   | 85.33             | 85.31   | 84.62   | 82.53           | 82.59   | 82.02   |

具体的评测方式可以参考[这份指南](../benchmark/evaluate_with_opencompass.md)。评测时，请在config文件中，为推理引擎添加 `quant_policy` 参数。

## 推理效率

| model             | kv type | test settings                            | RPS   | v.s. kv fp16 |
| ----------------- | ------- | ---------------------------------------- | ----- | ------------ |
| llama2-chat-7b    | fp16    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 14.98 | 1.0          |
| -                 | int8    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 19.01 | 1.27         |
| -                 | int4    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 20.81 | 1.39         |
| llama2-chat-13b   | fp16    | tp1 / ratio 0.9 / bs 128 / prompts 10000 | 8.55  | 1.0          |
| -                 | int8    | tp1 / ratio 0.9 / bs 256 / prompts 10000 | 10.96 | 1.28         |
| -                 | int4    | tp1 / ratio 0.9 / bs 256 / prompts 10000 | 11.91 | 1.39         |
| internlm2-chat-7b | fp16    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 24.13 | 1.0          |
| -                 | int8    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 25.28 | 1.05         |
| -                 | int4    | tp1 / ratio 0.8 / bs 256 / prompts 10000 | 25.80 | 1.07         |

上述结果使用的测试脚本是 `benchmark/profile_throughput.py`
