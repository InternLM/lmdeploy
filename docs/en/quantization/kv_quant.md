# INT4/INT8 KV Cache

Since v0.4.0, LMDeploy has supported **online** key-value (kv) cache quantization with int4 and int8 numerical precision, utilizing an asymmetric quantization method that is applied on a per-head, per-token basis. The original kv offline quantization method has been removed.

Intuitively, quantization is beneficial for increasing the number of kv block. Compared to fp16, the number of kv block for int4/int8 kv can be increased by 4 times and 2 times respectively. This means that under the same memory conditions, the system can support a significantly increased number of concurrent operations after kv quantization, thereby ultimately enhancing throughput.

However, quantization typically brings in some loss of model accuracy. We have used OpenCompass to evaluate the accuracy of several models after applying int4/int8 quantization. int8 kv keeps the accuracy while int4 kv has slight loss. The detailed results are presented in the [Evaluation](#evaluation) section. You can refer to the information and choose wisely based on your requirements.

LMDeploy inference with quantized kv supports the following NVIDIA GPU models:

- Volta architecture (sm70): V100
- Turing architecture (sm75): 20 series, T4
- Ampere architecture (sm80, sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace architecture (sm89): 40 series
- Hopper architecture (sm90): H100, H200

In summary, LMDeploy kv quantization has the following advantages:

1. data-free online quantization
2. Supports all nvidia GPU models with Volta architecture (sm70) and above
3. KV int8 quantization has almost lossless accuracy, and KV int4 quantization accuracy is within an acceptable range
4. Efficient inference, with int8/int4 kv quantization applied to llama2-7b, RPS is improved by round 30% and 40% respectively compared to fp16

## TurboQuant

LMDeploy supports KV quantization based on [Google Research's TurboQuant technology](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (to be presented at ICLR 2026), achieving higher compression ratio with near-zero accuracy loss through K=4bit QJL4 + V=2bit MSE combination.

### Principles

TurboQuant achieves efficient compression through two key steps:

1. **High-quality compression (PolarQuant method)**: First randomly rotates the data vectors (using orthogonal transforms like Hadamard transform). This clever step simplifies the data's geometry, making it easy to apply a standard, high-quality quantizer to each part of the vector individually. This stage uses most of the compression power (the majority of the bits) to capture the main concept and strength of the original vector.

2. **Eliminating hidden errors (QJL method)**: Uses a small, residual amount of compression power (just 1 bit) to apply the QJL (Quantized Johnson-Lindenstrauss) algorithm to the tiny amount of error left over from the first stage. The QJL stage acts as a mathematical error-checker that eliminates bias, leading to more accurate attention scores.

### K/V Quantization Scheme

- **K Path - QJL4 Quantization**:

  - Uses 3-bit Lloyd-Max codebook for MSE quantization (captures main information)
  - Uses 1-bit QJL to store residual sign (eliminates error bias)
  - Each token's K is compressed to 4-bit

- **V Path - MSE int2 Quantization**:

  - Uses 2-bit Lloyd-Max codebook for MSE quantization
  - Each token's V is compressed to 2-bit
  - Stores normalization coefficients for dequantization

### Advantages

- **Zero accuracy loss**: Through PolarQuant + QJL combination, achieves high compression rate while maintaining model accuracy
- **Higher compression ratio**: K 4bit + V 2bit = average 3bit, further compression compared to int4's 4bit
- **Eliminates quantization bias**: QJL algorithm acts as error-checker, effectively eliminating quantization-induced bias

### Performance Benchmark

Tested on H200 with Qwen3-30B-A3B-Base model and ShareGPT dataset:

| Metric             | Baseline (quant_policy=0) | TurboQuant (quant_policy=42) | Change     |
| ------------------ | ------------------------- | ---------------------------- | ---------- |
| Input throughput   | 2368.8 tok/s              | 2195.8 tok/s                 | -7.3%      |
| Output throughput  | 2186.7 tok/s              | 2027.0 tok/s                 | -7.3%      |
| Request throughput | 10.74 req/s               | 9.96 req/s                   | -7.3%      |
| Mean E2E latency   | 5.888s                    | 6.348s                       | +7.8%      |
| Mean TTFT          | 1.139s                    | 1.235s                       | +8.4%      |
| Mean TPOT          | 0.024s                    | 0.026s                       | +8.3%      |
| Mean ITL           | 0.059s                    | 0.059s                       | ~unchanged |

**Test configuration**: GPU: H200, Model: Qwen3-30B-A3B-Base, Dataset: ShareGPT, Concurrency: 64, Requests: 5000

**Takeaway**: TurboQuant K4V2 achieves ~5x KV cache memory reduction with about 7%-8% end-to-end performance overhead, which looks like a reasonable trade-off for memory-bound serving scenarios.

### Limitations

- **PytorchEngine only**: TurboQuant currently only supports PyTorch engine, not Turbomind engine
- **MLA not supported**: Does not support Multi-head Latent Attention architecture
- **Speculative decoding not supported**: Does not support speculative decoding
- Requires head_dim to be a power of 2
- Requires `fast_hadamard_transform` package for best performance (optional)

### Optional Dependency

TurboQuant uses Hadamard transform to accelerate the quantization process. Installing `fast_hadamard_transform` provides better performance:

```shell
pip install fast_hadamard_transform
```

Without this dependency, TurboQuant still works correctly, but performance may be slightly reduced.

In the next section, we will take `internlm2-chat-7b` model as an example, introducing the usage of kv quantization and inference of lmdeploy. But before that, please ensure that lmdeploy is installed.

```shell
pip install lmdeploy
```

## Usage

Applying kv quantization and inference via LMDeploy is quite straightforward. Simply set the `quant_policy` parameter.

**LMDeploy specifies that `quant_policy=4` stands for 4-bit kv, `quant_policy=8` indicates 8-bit kv, and `quant_policy=42` indicates TurboQuant.**

### Offline inference

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(quant_policy=8)
pipe = pipeline("internlm/internlm2_5-7b-chat", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### Serving

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --quant-policy 8
```

### TurboQuant

TurboQuant uses `quant_policy=42`, **PytorchEngine only**:

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

## Evaluation

We apply kv quantization of LMDeploy to several LLM models and utilize OpenCompass to evaluate the inference accuracy. The results are shown in the table below:

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | internlm2.5-chat-7b | -       | -       | qwen1.5-7b-chat | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------------- | ------- | ------- | --------------- | ------- | ------- |
| dataset     | version | metric        | kv fp16        | kv int8 | kv int4 | kv fp16           | kv int8 | kv int4 | kv fp16             | kv int8 | kv int4 | fp16            | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 27.96   | 27.58   | 60.45             | 60.88   | 60.28   | 78.06               | 77.87   | 77.05   | 70.56           | 70.49   | 68.62   |
| mmlu        | -       | naive_average | 35.64          | 35.58   | 34.79   | 63.91             | 64      | 62.36   | 72.30               | 72.27   | 71.17   | 61.48           | 61.56   | 60.65   |
| triviaqa    | 2121ce  | score         | 56.09          | 56.13   | 53.71   | 58.73             | 58.7    | 58.18   | 65.09               | 64.87   | 63.28   | 44.62           | 44.77   | 44.04   |
| gsm8k       | 1d7fe4  | accuracy      | 28.2           | 28.05   | 27.37   | 70.13             | 69.75   | 66.87   | 85.67               | 85.44   | 83.78   | 54.97           | 56.41   | 54.74   |
| race-middle | 9a54b6  | accuracy      | 41.57          | 41.78   | 41.23   | 88.93             | 88.93   | 88.93   | 92.76               | 92.83   | 92.55   | 87.33           | 87.26   | 86.28   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.77   | 40.77   | 85.33             | 85.31   | 84.62   | 90.51               | 90.42   | 90.42   | 82.53           | 82.59   | 82.02   |

For detailed evaluation methods, please refer to [this](../benchmark/evaluate_with_opencompass.md) guide. Remember to pass `quant_policy` to the inference engine in the config file.

## Performance

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

The performance data is obtained by `benchmark/profile_throughput.py`
