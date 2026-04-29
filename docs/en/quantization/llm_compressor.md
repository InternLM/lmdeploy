# llm-compressor Support

This guide introduces how to use LMDeploy's TurboMind inference engine to run models quantized to FP8 or INT4 by [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).
Currently supported `llm-compressor` quantized models include:

- int4 quantization, such as AWQ and GPTQ
- fp8 quantization

These quantized models can run with the TurboMind engine on the following NVIDIA GPU architectures:

| Compute Capability | Micro-architecture | GPUs                            |
| ------------------ | ------------------ | ------------------------------- |
| 7.0                | Volta              | V100                            |
| 7.2                | Volta              | Jetson Xavier                   |
| 7.5                | Turing             | GeForce RTX 20 series, T4       |
| 8.0                | Ampere             | A100, A800, A30                 |
| 8.6                | Ampere             | GeForce RTX 30 series, A40, A10 |
| 8.7                | Ampere             | Jetson Orin                     |
| 8.9                | Ada Lovelace       | GeForce RTX 40 series, L40, L20 |
| 9.0                | Hopper             | H20, H200, H100, GH200          |
| 12.0               | Blackwell          | GeForce RTX 50 series           |

LMDeploy will continue to follow up and expand support for the `llm-compressor` project.

The remainder of this document consists of the following sections:

<!-- toc -->

- [Model Quantization](#model-quantization)
  - [FP8 Quantization Example](#fp8-quantization-example)
  - [INT4 Quantization Example](#int4-quantization-example)
- [Model Deployment](#model-deployment)
  - [FP8 Deployment](#fp8-deployment)
  - [INT4 Deployment](#int4-deployment)
- [Accuracy Evaluation](#accuracy-evaluation)
  - [FP8 Accuracy Evaluation](#fp8-accuracy-evaluation)
  - [INT4 Accuracy Evaluation](#int4-accuracy-evaluation)

<!-- tocstop -->

## Model Quantization

`llm-compressor` provides a rich set of model quantization [examples](https://github.com/vllm-project/llm-compressor/tree/main/examples). Please refer to its tutorials to select a quantization algorithm supported by LMDeploy and complete model quantization.
LMDeploy also provides built-in scripts for FP8 and INT4 quantization of Qwen3-30B-A3B with `llm-compressor` for reference.

### FP8 Quantization Example

LMDeploy provides a built-in [script](https://github.com/InternLM/lmdeploy/blob/main/examples/lite/fp8/qwen3_30b_a3b_fp8.py) for FP8 quantization of Qwen3-30B-A3B with `llm-compressor-fp8`:

```shell
# Create conda environment
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# Install llm-compressor
pip install llmcompressor

# Clone the lmdeploy source code and run the quantization example
git clone https://github.com/InternLM/lmdeploy
cd lmdeploy
python examples/lite/fp8/qwen3_30b_a3b_fp8.py --work-dir ./qwen3_30b_a3b_fp8
```

In the following sections, we use this quantized model as an example to introduce model deployment and accuracy evaluation.

### INT4 Quantization Example

LMDeploy provides a built-in [script](https://github.com/InternLM/lmdeploy/blob/main/examples/lite/int4/qwen3_30b_a3b_awq.py) for AWQ quantization of Qwen3-30B-A3B with `llm-compressor-int4`:

```shell
# Create conda environment
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# Install llm-compressor
pip install llmcompressor

# Clone the lmdeploy source code and run the quantization example
git clone https://github.com/InternLM/lmdeploy
cd lmdeploy
python examples/lite/int4/qwen3_30b_a3b_awq.py --work-dir ./qwen3_30b_a3b_awq
```

In the following sections, we use this quantized model as an example to introduce model deployment and accuracy evaluation.

## Model Deployment

Quantized models can be used for offline inference through LMDeploy, or wrapped as services with api_server to provide OpenAI-compatible RESTful APIs.

### FP8 Deployment

#### Offline Inference

With the FP8-quantized model, offline batch processing can be implemented with just a few lines of code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_fp8", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

For a detailed introduction to the pipeline, please refer to [here](https://lmdeploy.readthedocs.io/en/latest/llm/pipeline.html).

#### Online Serving

Below is an example of starting a service for an FP8-quantized model:

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_fp8 --backend turbomind
```

The default service port is 23333. After the server starts, you can access the service via the OpenAI SDK. For command arguments and ways to access the service, please read [this](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html) document.

### INT4 Deployment

#### Offline Inference

With the INT4-quantized model, offline batch processing can be implemented with just a few lines of code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_awq", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

For a detailed introduction to the pipeline, please refer to [here](https://lmdeploy.readthedocs.io/en/latest/llm/pipeline.html).

#### Online Serving

Below is an example of starting a service for an INT4-quantized model:

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_awq --backend turbomind
```

The default service port is 23333. After the server starts, you can access the service via the OpenAI SDK. For command arguments and ways to access the service, please read [this](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html) document.

## Accuracy Evaluation

We deployed the FP8- and INT4-quantized models of Qwen3-8B (Dense) and Qwen3-30B-A3B (MoE) as services through LMDeploy, and evaluated them on several academic datasets with [opencompass](https://github.com/open-compass/opencompass).

### FP8 Accuracy Evaluation

The results show that the accuracy gap between the FP8-quantized models and the BF16 models of Qwen3-8B and Qwen3-30B-A3B is not significant, which is in line with expectations.

| dataset           | Qwen3-8B |       | Qwen3-30B-A3B |       |
| ----------------- | -------- | ----- | ------------- | ----- |
|                   | bf16     | fp8   | bf16          | fp8   |
| ifeval            | 85.58    | 87.62 | 86.32         | 86.51 |
| hle               | 5.05     | 5.89  | 7.00          | 7.51  |
| gpqa              | 59.97    | 59.22 | 61.74         | 60.73 |
| aime2025          | 69.48    | 70.00 | 73.44         | 71.15 |
| mmlu_pro          | 73.69    | 73.54 | 77.85         | 77.50 |
| LCBCodeGeneration | 50.86    | 49.81 | 56.67         | 56.86 |

### INT4 Accuracy Evaluation

The results show that asymmetric quantization is generally better than symmetric quantization for Qwen3-8B, while Qwen3-30B-A3B shows no significant difference between the two quantization methods. Under symmetric/asymmetric quantization, Qwen3-8B has a smaller accuracy gap from the BF16 model than Qwen3-30B-A3B. Compared with BF16, the quantized models show a more obvious accuracy drop on long-output datasets, such as aime2025 (average 17,635 tokens) and LCB (average 14,157 tokens). On medium- and short-output datasets, such as ifeval (average 1,885 tokens) and mmlu_pro (average 2,826 tokens), the accuracy is in line with expectations.

| dataset           | Qwen3-8B |         |          | Qwen3-30B-A3B |         |          |
| ----------------- | -------- | ------- | -------- | ------------- | ------- | -------- |
|                   | bf16     | awq sym | awq asym | bf16          | awq sym | awq asym |
| ifeval            | 85.58    | 83.73   | 85.77    | 86.32         | 84.10   | 84.29    |
| hle               | 5.05     | 5.05    | 5.24     | 7.00          | 5.47    | 5.65     |
| gpqa              | 59.97    | 56.57   | 59.47    | 61.74         | 57.95   | 57.07    |
| aime2025          | 69.48    | 64.38   | 63.96    | 73.44         | 64.79   | 66.67    |
| mmlu_pro          | 73.69    | 71.73   | 72.34    | 77.85         | 75.77   | 75.69    |
| LCBCodeGeneration | 50.86    | 44.10   | 46.95    | 56.67         | 50.86   | 49.24    |

For reproduction methods, please refer to [this](https://lmdeploy.readthedocs.io/en/latest/benchmark/evaluate_with_opencompass.html) document.
