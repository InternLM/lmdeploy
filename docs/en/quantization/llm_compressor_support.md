# llm-compressor Support

This guide aims to introduce how to use LMDeploy's TurboMind inference engine to run models quantized by the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) tool.

Currently supported `llm-compressor` quantization types include:

- 4-bit asymmetric quantization (e.g., AWQ, GPTQ)

These quantized models can run via the TurboMind engine on the following NVIDIA GPU architectures:

- **V100 (sm70)**: V100
- **Turing (sm75)**: 20 series, T4
- **Ampere (sm80, sm86)**: 30 series, A10, A16, A30, A100
- **Ada Lovelace (sm89)**: 40 series

LMDeploy will continue to follow up and expand support for the `llm-compressor` project.

The remainder of this document consists of the following sections:

<!-- toc -->

- [Model Quantization](#model-quantization)
- [Inference Deployment](#inference-deployment)
- [Accuracy Evaluation](#accuracy-evaluation)

<!-- tocstop -->

## Model Quantization

`llm-compressor` provides a wealth of model quantization examples. Please refer to its tutorials to select a quantization algorithm supported by LMDeploy to complete your model quantization work.

LMDeploy also provides a built-in script for AWQ quantization of **Qwen3-30B-A3B** using `llm-compressor` for your reference:

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

In the following sections, we will use this quantized model as an example to introduce inference deployment and accuracy evaluation methods.

## Inference Deployment

### Offline Inference

With the quantized model, offline batch processing can be implemented with just a few lines of code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_4bit", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

For a detailed introduction to the pipeline, please refer to [here](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/pipeline.html).

### Online Serving

LMDeploy api_server supports encapsulating the model as a service with a single command. The provided RESTful APIs are compatible with OpenAI interfaces. Below is an example of starting the service:

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_4bit --backend turbomind
```

The default service port is 23333. After the server starts, you can access the service via the OpenAI SDK. For command arguments and methods to access the service, please read [this](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/api_server.html) document.

## Accuracy Evaluation

After deploying the aforementioned quantized model as a service via LMDeploy, we evaluated its accuracy on several academic datasets using OpenCompass. Compared with the original model, the accuracy difference is within an acceptable range:

| dataset                                                                                                                                           | Qwen3-30B-A3B | Qwen3-30B-A3B awq | diff  |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------------- | ----- |
| core_average                                                                                                                                      | 43.63         | 42.83             | -0.8  |
| IFEval                                                                                                                                            | 85.77         | 85.77             | 0     |
| hle_llmjudge                                                                                                                                      | 2.18          | 1.95              | -0.23 |
| GPQA_diamond_repeat_4                                                                                                                             | 51.39         | 49.37             | -2.02 |
| aime2025_repeat_32                                                                                                                                | 18.02         | 19.58             | 1.56  |
| mmlu_pro                                                                                                                                          | 74.05         | 72.86             | -1.19 |
| lcb_code_generation_repeat_6                                                                                                                      | 30.38         | 27.43             | -2.95 |
| For reproduction methods, please refer to [this](https://lmdeploy.readthedocs.io/zh-cn/latest/benchmark/evaluate_with_opencompass.html) document. |               |                   |       |
