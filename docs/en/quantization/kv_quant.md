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

| -           | -       | -             | llama2-7b-chat | -       | -       | internlm2-chat-7b | -       | -       | qwen-chat-7b | -       | -       |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------ | ------- | ------- |
| dataset     | version | metric        | fp16           | kv int8 | kv int4 | fp16              | kv int8 | kv int4 | fp16         | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 28.38   | 27.18   | 60.45             | 60.71   | 59.8    | 59.34        | 60.05   | 60.77   |
| mmlu        | -       | naive_average | 35.58          | 35.58   | 34.94   | 63.92             | 64      | 62.63   | 57.45        | 57.41   | 56.39   |
| triviaqa    | 2121ce  | score         | 56.13          | 56.08   | 53.79   | 58.74             | 58.69   | 57.87   | 54.07        | 54.05   | 53.64   |
| gsm8k       | 1d7fe4  | accuracy      | 28.28          | 28.43   | 26.54   | 70.58             | 69.75   | 68.08   | 53.53        | 53.22   | 52.69   |
| race-middle | 9a54b6  | accuracy      | 41.64          | 41.78   | 42.41   | 88.93             | 88.86   | 89.28   | 83.15        | 83.08   | 83.29   |
| race-high   | 9a54b6  | accuracy      | 39.65          | 39.51   | 40.65   | 85.28             | 85.31   | 84.05   | 76.67        | 76.76   | 77.36   |

For detailed evaluation methods, please refer to [this](../benchmark/evaluate_with_opencompass.md) guide. Remember to pass `quant_policy` to the inference engine in the config file.

## Performance

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

The performance data is obtained by `benchmark/profile_throughput.py`
