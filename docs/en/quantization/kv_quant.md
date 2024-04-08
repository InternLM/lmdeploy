# KV Cache Quantization

The latest main branch of LMDeploy supports online key-value (KV) cache quantization with 4-bit and 8-bit precision, utilizing an asymmetric quantization method that is applied on a per-head, per-token basis. The original KV offline quantization method has been removed.

Intuitively, quantizing the KV cache is beneficial for reducing memory usage. Compared to FP16, the memory for 4-bit/8-bit KV can be reduced to 1/4 and 1/2, respectively. This means that under the same memory conditions, the system can support a significantly increased number of concurrent operations after KV quantization, thereby ultimately enhancing throughput.

However, quantization typically brings in some loss of model accuracy. We have used OpenCompass to evaluate the accuracy of several models after applying 8/4-bit KV quantization, and the results are presented in the [Evaluation](#Evaluation) section. You can refer to the information and choose wisely based on your requirements.

LMDeploy inference with quantized KV supports the following NVIDIA GPU models:

- Volta architecture (sm70): V100
- Turing architecture (sm75): 20 series, T4
- Ampere architecture (sm80, sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace architecture (sm89): 40 series

In the next section, we will take `internlm2-chat-7b` model as an example, introducing the usage of kv quantization and inference of lmdeploy. But before that, please install lmdeploy from source according to the [build](../build.md) guide, because lmdeploy hasn't released this feature yet.

## Usage

Applying KV quantization and inference via LMDeploy is quit straightforward; simply set the `quant_policy` parameter.

**LMDeploy specifies that `quant_policy=4` stands for 4-bit KV, whereas `quant_policy=8` indicates 8-bit KV.**

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

We apply KV quantization of LMDeploy to several LLM models and utilize OpenCompass to evaluate the inference accuracy. The results are shown in the table below:

| -           | -       | -             | llama2-7b-chat |         |         | internlm2-chat-7b |         |         | qwen-chat-7b |         |         |
| ----------- | ------- | ------------- | -------------- | ------- | ------- | ----------------- | ------- | ------- | ------------ | ------- | ------- |
| dataset     | version | metric        | fp16           | kv int8 | kv int4 | fp16              | kv int8 | kv int4 | bf16         | kv int8 | kv int4 |
| ceval       | -       | naive_average | 28.42          | 28.07   | 28.18   | 60.45             | 60.48   | 58.91   | 59.32        | 59.59   | 59.42   |
| mmlu        | -       | naive_average | 35.61          | 35.63   | 35.11   | 63.92             | 63.78   | 63.29   | 57.27        | 57.39   | 56.07   |
| triviaqa    | 2121ce  | score         | 56.12          | 56.04   | 54.09   | 58.76             | 58.67   | 58.32   | 54.42        | 54.27   | 54.46   |
| gsm8k       | 1d7fe4  | accuracy      | 28.35          | 28.05   | 25.17   | 70.58             | 70.36   | 66.34   | 53.53        | 52.69   | 53.07   |
| race-middle | 9a54b6  | accuracy      | 41.64          | 42.13   | 45.33   | 88.93             | 88.79   | 88.86   | 83.7         | 83.57   | 82.94   |

For detailed evaluation methods, please refer to [this](../benchmark/evaluate_with_opencompass.md) guide. Remember to pass `quant_policy` to the inference engine in the config file.

## Performance

TODO
