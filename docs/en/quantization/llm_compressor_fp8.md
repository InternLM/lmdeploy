# llm-compressor-fp8 Support

This guide aims to introduce how to use LMDeploy's TurboMind inference engine to run models quantized by the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) tool.

Currently supported `llm-compressor-fp8` quantization types include:

- AWQ、GPTQ

These quantized models can run via the TurboMind engine on the following NVIDIA GPU architectures:

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

LMDeploy will continue to follow up and expand support for the `llm-compressor-fp8` project.

The remainder of this document consists of the following sections:

<!-- toc -->

- [Model Quantization](#model-quantization)
- [Model Deployment](#model-deployment)
- [Accuracy Evaluation](#accuracy-evaluation)

<!-- tocstop -->

## Model Quantization

`llm-compressor-fp8` provides a wealth of model quantization [examples](https://github.com/vllm-project/llm-compressor/tree/main/examples). Please refer to its tutorials to select a quantization algorithm supported by LMDeploy to complete your model quantization work.

LMDeploy also provides a built-in [script](https://github.com/InternLM/lmdeploy/blob/main/examples/lite/fp8/qwen3_30b_a3b_fp8.py) for FP8 quantization of **Qwen3-30B-A3B** using `llm-compressor-fp8` for your reference:

```shell
# Create conda environment
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# Install llm-compressor
pip install llmcompressor

# Clone lmdeploy source code and run the quantization example
git clone https://github.com/InternLM/lmdeploy
cd lmdeploy
python examples/lite/fp8/qwen3_30b_a3b_fp8.py --work-dir ./qwen3_30b_a3b_fp8

```

In the following sections, we will use this quantized model as an example to introduce model deployment and accuracy evaluation methods.

## Model Deployment

### Offline Inference

With the quantized model, offline batch processing can be implemented with just a few lines of code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_fp8", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

For a detailed introduction to the pipeline, please refer to [here](https://lmdeploy.readthedocs.io/en/latest/llm/pipeline.html).

### Online Serving

LMDeploy api_server supports encapsulating the model as a service with a single command. The provided RESTful APIs are compatible with OpenAI interfaces. Below is an example of starting the service:

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_fp8 --backend turbomind
```

The default service port is 23333. After the server starts, you can access the service via the OpenAI SDK. For command arguments and methods to access the service, please read [this](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html) document.

## Accuracy Evaluation

We deployed FP8-quantized models of Qwen3-8B (Dense) and Qwen3-30B-A3B (MoE) as services via LMDeploy, and evaluated them on several academic datasets using [opencompass](https://github.com/open-compass/opencompass). The results show that the accuracy gap between the FP8-quantized models and the BF16 models is not significant, which is in line with expectations.

| dataset           | Qwen3-8B |       | Qwen3-30B-A3B |       |
| ----------------- | -------- | ----- | ------------- | ----- |
|                   | bf16     | fp8   | bf16          | fp8   |
| ifeval            | 85.58    | 87.62 | 86.32         | 86.51 |
| hle               | 5.05     | 5.89  | 7.00          | 7.51  |
| gpqa              | 59.97    | 59.22 | 61.74         | 60.73 |
| aime2025          | 69.48    | 70.00 | 73.44         | 71.15 |
| mmlu_pro          | 73.69    | 73.54 | 77.85         | 77.50 |
| LCBCodeGeneration | 50.86    | 49.81 | 56.67         | 56.86 |

For reproduction methods, please refer to [this](https://lmdeploy.readthedocs.io/en/latest/benchmark/evaluate_with_opencompass.html) document.
