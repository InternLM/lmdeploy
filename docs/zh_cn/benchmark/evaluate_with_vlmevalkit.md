# 多模态模型评测指南

本文档介绍如何使用 VLMEvalKit 和 LMDeploy 评测多模态模型能力。

## 环境准备

```shell
pip install lmdeploy

git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && pip install -e .
```

建议在不同的 Python 虚拟环境中分别安装 LMDeploy 和 VLMEvalKit，以避免潜在的依赖冲突。

## 评测

1. **部署大语言多模态模型 (LMMs)**

```shell
lmdeploy serve api_server <model_path> --server-port 23333 <--other-options>
```

2. **配置评测设置**

修改 `VLMEvalKit/vlmeval/config.py`，在 `api_models` 字典中添加以下 LMDeploy API 配置。

`<task_name>` 是您评测任务的自定义名称（例如 `lmdeploy_qwen3vl-4b`）。`model` 参数应与 `lmdeploy serve` 命令中使用的 `<model_path>` 保持一致。

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

3. **开始评测**

```shell
cd VLMEvalKit
python run.py --data OCRBench --model <task_name> --api-nproc 16 --reuse --verbose --api 123
```

`<task_name>` 应与上述配置文件中使用的名称保持一致。

参数说明：

- `--data`: 指定用于评测的数据集（例如 `OCRBench`）。
- `--model`: 指定模型名称，必须与您在 `config.py` 中设置的 `<task_name>` 匹配。
- `--api-nproc`: 指定并行的 API 调用数量。
- `--reuse`: 复用先前的推理结果，以避免重新运行已完成的评测。
- `--verbose`: 启用详细日志记录。
