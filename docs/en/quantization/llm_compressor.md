# llm-compressor Support

This guide aims to introduce how to use LMDeploy's TurboMind inference engine to run models quantized by the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) tool.

Currently supported `llm-compressor` quantization types include:

- int4 quantization (e.g., AWQ, GPTQ)

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

LMDeploy will continue to follow up and expand support for the `llm-compressor` project.

The remainder of this document consists of the following sections:

<!-- toc -->

- [Model Quantization](#model-quantization)
- [Model Deployment](#model-deployment)
- [Accuracy Evaluation](#accuracy-evaluation)

<!-- tocstop -->

## Model Quantization

`llm-compressor` provides a wealth of model quantization [examples](https://github.com/vllm-project/llm-compressor/tree/main/examples). Please refer to its tutorials to select a quantization algorithm supported by LMDeploy to complete your model quantization work.

LMDeploy also provides a built-in [script](https://github.com/InternLM/lmdeploy/examples/lite/qwen3_30b_a3b_awq.py) for AWQ quantization of **Qwen3-30B-A3B** using `llm-compressor` for your reference:

```shell
# Create conda environment
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# Install llm-compressor
pip install llm-compressor

# Clone lmdeploy source code and run the quantization example
git clone https://github.com/InternLM/lmdeploy
cd lmdeploy
python examples/lite/qwen3_30b_a3b_awq.py --work-dir ./qwen3_30b_a3b_awq
```

In the following sections, we will use this quantized model as an example to introduce model deployment and accuracy evaluation methods.

## Model Deployment

### Offline Inference

With the quantized model, offline batch processing can be implemented with just a few lines of code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_4bit", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

For a detailed introduction to the pipeline, please refer to [here](https://lmdeploy.readthedocs.io/en/latest/llm/pipeline.html).

### Online Serving

LMDeploy api_server supports encapsulating the model as a service with a single command. The provided RESTful APIs are compatible with OpenAI interfaces. Below is an example of starting the service:

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_4bit --backend turbomind
```

The default service port is 23333. After the server starts, you can access the service via the OpenAI SDK. For command arguments and methods to access the service, please read [this](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html) document.

## Accuracy Evaluation

After deploying the AWQ symmetric and AWQ asymmetric quantized Qwen3-30B-A3B models as services via LMDeploy, we evaluated their accuracy on several academic datasets using [opencompass](https://github.com/open-compass/opencompass). Compared with the Qwen3-30B-A3B BF16 model, the accuracy differences are within an acceptable range:

| dataset           | bf16  | awq sym | awq asym |
| ----------------- | ----- | ------- | -------- |
| ifeval            | 86.32 | 84.10   | 84.29    |
| hle               | 7.00  | 5.47    | 5.65     |
| gpqa              | 61.74 | 57.95   | 57.07    |
| aime2025          | 73.44 | 64.79   | 66.67    |
| mmlu_pro          | 77.85 | 75.77   | 75.69    |
| LCBCodeGeneration | 56.67 | 50.86   | 49.24    |

For reproduction methods, please refer to [this](https://lmdeploy.readthedocs.io/en/latest/benchmark/evaluate_with_opencompass.html) document.
