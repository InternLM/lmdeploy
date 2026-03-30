# Multi-Modal Model Evaluation Guide

This document describes how to evaluate multi-modal models' capabilities using VLMEvalKit and LMDeploy.

## Environment Setup

```shell
pip install lmdeploy

git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && pip install -e .
```

It is recommended to install LMDeploy and VLMEvalKit in separate Python virtual environments to avoid potential dependency conflicts.

## Evaluations

1. **Deploy Large Multi-Modality Models (LMMs)**

```shell
lmdeploy serve api_server <model_path> --server-port 23333 <--other-options>
```

2. **Config the Evaluation Settings**

Modify `VLMEvalKit/vlmeval/config.py`, add following LMDeploy API configurations in the `api_models` dictionary.

The `<task_name>` is a custom name for your evaluation task (e.g., `lmdeploy_qwen3vl-4b`). The `model` parameter should match the `<model_path>` used in the `lmdeploy serve` command.

```python
// filepath: VLMEvalKit/vlmeval/config.py
// ...existing code...
api_models = {
    # lmdeploy api
    ...,
    "<task_name>": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        model="<model_path>",
        retry=4,
        timeout=1200,
        temperature=0.7, # modify if needed
        max_new_tokens=16384, # modify if needed
    ),
    ...
}
// ...existing code...
```

3. **Start Evaluations**

```shell
cd VLMEvalKit
python run.py --data OCRBench --model <task_name> --api-nproc 16 --reuse --verbose --api 123
```

The `<task_name>` should match the one used in the above config file.

Parameter explanations:

- `--data`: Specify the dataset for evaluation (e.g., `OCRBench`).
- `--model`: Specify the model name, which must match the `<task_name>` in your `config.py`.
- `--api-nproc`: Specify the number of parallel API calls.
- `--reuse`: Reuse previous inference results to avoid re-running completed evaluations.
- `--verbose`: Enable verbose logging.
