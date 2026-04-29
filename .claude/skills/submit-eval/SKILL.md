---
name: submit-eval
description: Use when submitting a model eval task to the auto-eval platform
disable-model-invocation: true
---

# Submit Eval Task

Submit a model evaluation task to the auto-eval platform.

## Prerequisites

Read `~/.eval/config` and verify these required keys are present:

```
AUTO_EVAL_TOKEN
FEISHU_EVAL_WEBHOOK
USER
OPENAI_API_BASE
AUTO_EVAL_API_URL
```

If any are missing, stop and tell the user to populate `~/.eval/config`.

Also verify `~/.eval/models/model.yaml` exists. If missing, stop and tell the user to create it.

## 1. Gather inputs

Ask the user for:

- **model_abbr** — key in `~/.eval/models/model.yaml`
- **instances** — number of inference instances (integer, used to compute `end_num`)
- **datasets** — comma-separated dataset keys (looked up in `~/.eval/config`)
- **image** (optional) — Docker image for the eval container

## 2. Look up model config

Read `~/.eval/models/model.yaml`. Find the entry matching `model_abbr`. If not found, stop and list available model keys.

From the model entry, extract:

- `model_path` (required)
- `tp` (required)
- `backend` (required)
- `reasoning_parser` (optional)
- `tool_call_parser` (optional)

## 3. Resolve image

If the user provided an `image`, use it. Otherwise, invoke the `docker-build` skill to build and push an image from the current branch, then use the resulting image tag.

## 4. Resolve datasets

Parse the comma-separated `datasets` input. For each key, look up its value in `~/.eval/config`. If a key is not found, stop and list available dataset keys.

Combine the values into `subdataset`: `[*val1, *val2, ...]`

## 5. Compute derived fields

Compute the timestamp-padded model name:

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_ABBR_PADDED="${model_abbr}-${TIMESTAMP}"
```

Compute `infer_extra_params`:

```
--tp {tp} --backend {backend} --reasoning-parser {reasoning_parser} [--tool-call-parser {tool_call_parser}]
```

- Append `--tool-call-parser {tool_call_parser}` only if `tool_call_parser` is set.

Compute resource fields:

- `gpu_num` = `tp`
- `cpu` = `16 * tp`
- `memory` = `"128000 * tp"` (as string)
- `end_num` = `instances + 1`
- `tokenizer_path` = `model_path`
- `output_dir` = `"./{USER}/${MODEL_ABBR_PADDED}"`

## 6. Assemble model_infer_config

Build as a Python dict string with these fields:

```python
{
    'type': 'OpenAISDKStreaming',
    'key': 'sk-admin',
    'openai_api_base': ['{OPENAI_API_BASE}'],
    'query_per_second': 8,
    'batch_size': 32,
    'max_workers': 8,
    'temperature': 1,
    'tokenizer_path': '{model_path}',
    'retry': 50,
    'max_out_len': 128000,
    'max_seq_len': 128000,
    'extra_body': {
        'top_k': 20,
        'repetition_penalty': 1.0,
        'top_p': 0.95,
    },
    'verbose': True,
}
```

If `reasoning_parser` is set, add to `extra_body`:

```python
'chat_template_kwargs': {'enable_thinking': True},
```

And add at the top level:

```python
'pred_postprocessor': {'type': 'extract-non-reasoning-content'},
```

## 7. Compute model_infer_config_base64

```bash
echo -n '{model_infer_config}' | base64 -w 0
```

## 8. Assemble infer_backend_config

```json
{
    "end_num": {instances + 1},
    "gpu_num": {tp},
    "memory": "{128000 * tp}",
    "cpu": {16 * tp},
    "parallelism": "TP",
    "oc_cpu": "1",
    "oc_mem": 4000,
    "model": "{MODEL_ABBR_PADDED}",
    "model_path": "{model_path}",
    "image": "{image}",
    "infer_engine": "lmdeploy",
    "infer_extra_params": "{infer_extra_params}",
    "delete": "false",
    "start_infer": "true",
    "node_num": 1
}
```

## 9. Assemble full payload

Build the JSON body:

```json
{
    "job_name": "api_eval_v4",
    "param": {
        "cluster": "yidian",
        "workspace_id": "evalservice_gpu",
        "model_abbr": "{MODEL_ABBR_PADDED}",
        "user": "{USER}",
        "model_infer_config": "{model_infer_config as string}",
        "llm_judger_config": "",
        "infer_worker_nums": 8,
        "eval_nums": "15",
        "eval_type": "chat_objective",
        "auto_eval_version": "ld_0122_oc_0524d49_v2",
        "ocp_version": "fullbench_v2_0",
        "subdataset": "{subdataset}",
        "fast_infer": "true",
        "output_dir": "{output_dir}",
        "eval_only": "false",
        "cli_extra": "",
        "dataset_max_out_len": "128000",
        "feishu_token": "{FEISHU_EVAL_WEBHOOK}",
        "model_infer_config_base64": "{model_infer_config_base64}",
        "infer_backend_config": {infer_backend_config}
    }
}
```

## 10. Submit

Execute the curl command:

```bash
curl -s -X POST "${AUTO_EVAL_API_URL}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${AUTO_EVAL_TOKEN}" \
  -d @- <<'JSON'
{payload}
JSON
```

Report the HTTP status and response body to the user.
