# Key-Value(KV) Cache Quantization

The latest main branch of LMDeploy supports online key-value (kv) cache quantization with 4-bit and 8-bit precision, utilizing an asymmetric quantization method that is applied on a per-head, per-token basis. The original kv offline quantization method has been removed.

Intuitively, quantizing the kv cache is beneficial for reducing memory usage. Compared to FP16, the memory for 4-bit/8-bit kv can be reduced to 1/4 and 1/2, respectively. This means that under the same memory conditions, the system can support a significantly increased number of concurrent operations after kv quantization, thereby ultimately enhancing throughput.

However, quantization typically brings in some loss of model accuracy. We have used OpenCompass to evaluate the accuracy of several models after applying 8/4-bit kv quantization, and the results are presented in the [Evaluation](#Evaluation) section. You can refer to the information and choose wisely based on your requirements.

LMDeploy inference with quantized kv supports the following NVIDIA GPU models:

- Volta architecture (sm70): V100
- Turing architecture (sm75): 20 series, T4
- Ampere architecture (sm80, sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace architecture (sm89): 40 series
- Hopper architecture (sm90): H100, H200

In the next section, we will take `internlm2-chat-7b` model as an example, introducing the usage of kv quantization and inference of lmdeploy. But before that, please install lmdeploy from source according to the [build](../build.md) guide, because lmdeploy hasn't released this feature yet.

## Usage

Applying kv quantization and inference via LMDeploy is quite straightforward. Simply set the `quant_policy` parameter.

**LMDeploy specifies that `quant_policy=4` stands for 4-bit kv, whereas `quant_policy=8` indicates 8-bit kv.**

### Offline inference

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(quant_policy=8)
pipe = pipeline("internlm/internlm2-chat-7b", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### Serving

```shell
lmdeploy serve api_server internlm/internlm2-chat-7b --quant-policy 8
```

## Evaluation

We apply kv quantization of LMDeploy to several LLM models and utilize OpenCompass to evaluate the inference accuracy. The results are shown in the table below:

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | qwen1.5-7b-chat | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | --------------- | ------- | ------- |
| dataset     | version | metric        | kv fp16        | kv int8 | kv int4 | kv fp16           | kv int8 | kv int4 | fp16            | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 27.96   | 27.58   | 60.45             | 60.88   | 60.28   | 70.56           | 70.49   | 68.62   |
| mmlu        | -       | naive_average | 35.64          | 35.58   | 34.79   | 63.91             | 64      | 62.36   | 61.48           | 61.56   | 60.65   |
| triviaqa    | 2121ce  | score         | 56.09          | 56.13   | 53.71   | 58.73             | 58.7    | 58.18   | 44.62           | 44.77   | 44.04   |
| gsm8k       | 1d7fe4  | accuracy      | 28.2           | 28.05   | 27.37   | 70.13             | 69.75   | 66.87   | 54.97           | 56.41   | 54.74   |
| race-middle | 9a54b6  | accuracy      | 41.57          | 41.78   | 41.23   | 88.93             | 88.93   | 88.93   | 87.33           | 87.26   | 86.28   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.77   | 40.77   | 85.33             | 85.31   | 84.62   | 82.53           | 82.59   | 82.02   |

For detailed evaluation methods, please refer to [this](../benchmark/evaluate_with_opencompass.md) guide. Remember to pass `quant_policy` to the inference engine in the config file.

## Performance

| model             | kv type | test settings                          | RPS   | v.s. kv fp16 |
| ----------------- | ------- | -------------------------------------- | ----- | ------------ |
| llama2-chat-7b    | fp16    | tp1/ratio 0.8 / bs 256 / prompts 10000 | 14.98 | 1.0          |
| -                 | kv8     | tp1/ratio 0.8 / bs 256 / prompts 10000 | 19.01 | 1.27         |
| -                 | kv4     | tp1/ratio 0.8 / bs 256 / prompts 10000 | 20.81 | 1.39         |
| llama2-chat-13b   | fp16    | tp1/ratio 0.9 / bs 128 / prompts 10000 | 8.55  | 1.0          |
| -                 | kv8     | tp1/ratio 0.9 / bs 256 / prompts 10000 | 10.96 | 1.28         |
| -                 | kv4     | tp1/ratio 0.9 / bs 256 / prompts 10000 | 11.91 | 1.39         |
| internlm2-chat-7b | fp16    | tp1/ratio 0.8 / bs 256 / prompts 10000 | 24.13 | 1.0          |
| -                 | kv8     | tp1/ratio 0.8 / bs 256 / prompts 10000 | 25.28 | 1.05         |
| -                 | kv4     | tp1/ratio 0.8 / bs 256 / prompts 10000 | 25.80 | 1.07         |

The performance data is obtained by `benchmark/profile_throughput.py`
