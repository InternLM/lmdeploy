# Key-Value(KV) Cache 量化

自 v0.4.0 起，LMDeploy 支持**在线** kv cache int4/int8 量化，量化方式为 per-head per-token 的非对称量化。原来的 kv 离线量化方式移除。

从直观上看，量化 kv 有利于增加 kv block 的数量。与 fp16 相比，int4/int8 kv 的 kv block 分别可以增加到 4 倍和 2 倍。这意味着，在相同的内存条件下，kv 量化后，系统能支撑的并发数可以大幅提升，从而最终提高吞吐量。

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

## TurboQuant 量化

LMDeploy 支持基于 [Google Research 的 TurboQuant 技术](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)（将在 ICLR 2026 发表）实现的 KV 量化方案，通过 K=4bit QJL4 + V=2bit MSE 的组合，实现更高的压缩率和几乎无损的精度。

### 原理

TurboQuant 通过两个关键步骤实现高效压缩：

1. **高质量压缩（PolarQuant 方法）**：首先对数据向量进行随机旋转（使用 Hadamard 变换等正交变换）。这个巧妙的步骤简化了数据的几何结构，使得可以对向量的每个部分单独应用标准的高质量量化器。这一阶段使用大部分压缩能力（大部分比特）来捕捉原始向量的主要概念和强度。

2. **消除隐藏误差（QJL 方法）**：使用少量剩余的压缩能力（仅 1 bit）将 QJL（Quantized Johnson-Lindenstrauss）算法应用于第一阶段剩余的微小误差。QJL 阶段充当数学误差检查器，消除偏差，从而获得更准确的注意力分数。

### K/V 量化方案

- **K 路径 - QJL4 量化**：

  - 使用 3bit Lloyd-Max 码本进行 MSE 量化（捕捉主要信息）
  - 使用 1bit QJL 存储残差符号（消除误差偏差）
  - 每个 token 的 K 压缩为 4bit

- **V 路径 - MSE int2 量化**：

  - 使用 2bit Lloyd-Max 码本进行 MSE 量化
  - 每个 token 的 V 压缩为 2bit
  - 存储归一化系数用于反量化

### 优势

- **零精度损失**：通过 PolarQuant + QJL 的组合，实现高压缩率的同时保持模型精度
- **更高的压缩率**：K 4bit + V 2bit = 平均 3bit，相比 int4 的 4bit 进一步压缩
- **消除量化偏差**：QJL 算法作为误差检查器，有效消除量化引入的偏差

### 性能测试

在 H200 上使用 Qwen3-30B-A3B-Base 模型、ShareGPT 数据集进行测试：

| 指标           | Baseline (quant_policy=0) | TurboQuant (quant_policy=42) | 变化  |
| -------------- | ------------------------- | ---------------------------- | ----- |
| 输入吞吐       | 2368.8 tok/s              | 2195.8 tok/s                 | -7.3% |
| 输出吞吐       | 2186.7 tok/s              | 2027.0 tok/s                 | -7.3% |
| 请求吞吐       | 10.74 req/s               | 9.96 req/s                   | -7.3% |
| 平均端到端延迟 | 5.888s                    | 6.348s                       | +7.8% |
| 平均 TTFT      | 1.139s                    | 1.235s                       | +8.4% |
| 平均 TPOT      | 0.024s                    | 0.026s                       | +8.3% |
| 平均 ITL       | 0.059s                    | 0.059s                       | 持平  |

**测试配置**：GPU: H200, 模型: Qwen3-30B-A3B-Base, 数据集: ShareGPT, 并发: 64, 请求数: 5000

**结论**：TurboQuant K4V2 实现约 5 倍的 KV cache 内存压缩，端到端性能开销约 7%-8%，在内存受限的 serving 场景中是一个合理的权衡。

### 限制

- **仅支持 PytorchEngine**：TurboQuant 目前仅支持 PyTorch 引擎，不支持 Turbomind 引擎
- **不支持 MLA**：不支持 Multi-head Latent Attention 结构
- **不支持推测解码**：不支持 speculative decoding
- 需要 head_dim 为 2 的幂次方（power of 2）
- 需要安装 `fast_hadamard_transform` 包以获得最佳性能（可选）

### 可选依赖

TurboQuant 使用 Hadamard 变换加速量化过程。安装 `fast_hadamard_transform` 可获得更好的性能：

```shell
pip install fast_hadamard_transform
```

不安装此依赖时，TurboQuant 仍可正常工作，但性能可能略有下降。

接下来，我们以 internlm2-chat-7b 模型为例，介绍 kv 量化和推理的若干应用。而在此之前，请安装 lmdeploy

```shell
pip install lmdeploy
```

## 应用示例

通过 LMDeploy 应用 kv 量化非常简单，只需要设定 `quant_policy` 参数。

**LMDeploy 规定 `quant_policy=4` 表示 kv int4 量化，`quant_policy=8` 表示 kv int8 量化，`quant_policy=42` 表示 TurboQuant 量化。**

### 离线推理

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(quant_policy=8)
pipe = pipeline("internlm/internlm2_5-7b-chat", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### 推理服务

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --quant-policy 8
```

### TurboQuant 量化

TurboQuant 量化使用 `quant_policy=42`，**仅支持 PytorchEngine**：

```python
from lmdeploy import pipeline, PytorchEngineConfig
engine_config = PytorchEngineConfig(
    tp=1,
    cache_max_entry_count=0.8,
    quant_policy=42  # TurboQuant: K=4bit QJL4 + V=2bit MSE
)
pipe = pipeline("Qwen/Qwen3-8B", backend_config=engine_config)
response = pipe.infer("Hello, how are you?", max_new_tokens=30)
print(response.text)
```

## 精度评测

我们把 lmdeploy 的 kv 量化应用在若干 LLM 模型上，并使用 opencompass 评测推理精度，结果如下表所示：

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | internlm2.5-chat-7b | -       | -       | qwen1.5-7b-chat | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------------- | ------- | ------- | --------------- | ------- | ------- |
| dataset     | version | metric        | kv fp16        | kv int8 | kv int4 | kv fp16           | kv int8 | kv int4 | kv fp16             | kv int8 | kv int4 | fp16            | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 27.96   | 27.58   | 60.45             | 60.88   | 60.28   | 78.06               | 77.87   | 77.05   | 70.56           | 70.49   | 68.62   |
| mmlu        | -       | naive_average | 35.64          | 35.58   | 34.79   | 63.91             | 64      | 62.36   | 72.30               | 72.27   | 71.17   | 61.48           | 61.56   | 60.65   |
| triviaqa    | 2121ce  | score         | 56.09          | 56.13   | 53.71   | 58.73             | 58.7    | 58.18   | 65.09               | 64.87   | 63.28   | 44.62           | 44.77   | 44.04   |
| gsm8k       | 1d7fe4  | accuracy      | 28.2           | 28.05   | 27.37   | 70.13             | 69.75   | 66.87   | 85.67               | 85.44   | 83.78   | 54.97           | 56.41   | 54.74   |
| race-middle | 9a54b6  | accuracy      | 41.57          | 41.78   | 41.23   | 88.93             | 88.93   | 88.93   | 92.76               | 92.83   | 92.55   | 87.33           | 87.26   | 86.28   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.77   | 40.77   | 85.33             | 85.31   | 84.62   | 90.51               | 90.42   | 90.42   | 82.53           | 82.59   | 82.02   |

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
