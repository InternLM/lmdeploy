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

Also verify `~/.eval/model.yaml` exists. If missing, stop and tell the user to create it.

## 1. Gather inputs

Read `~/.eval/model.yaml` and present the list of available model keys to the user. Ask them to select one or more models. The model key (e.g. `Qwen3.5-35B-A3B`) is the `model_abbr`. User input is matched case-insensitively — `qwen3.5-35b-a3b` matches `Qwen3.5-35B-A3B`. The original casing from the YAML key is used in the payload.

Then ask for:

- **instances** — number of inference instances (integer, used to compute `end_num`)
- **datasets** — comma-separated dataset keys (looked up in `~/.eval/config`)
- **image** (optional) — Docker image for the eval container

If the user selected multiple models, repeat steps 3-10 for each model.

## 2. Look up model config

For each selected model, read its entry from `~/.eval/model.yaml`. Extract `model_path` and all other fields.

All fields except `model_path` are passed as CLI flags to `infer_extra_params`, mapping each key to `--{key} {value}`. For example:

```yaml
tp: 2
backend: turbomind
reasoning_parser: qwen-qwq
tool_call_parser: qwen
```

produces:

```
--tp 2 --backend turbomind --reasoning-parser qwen-qwq --tool-call-parser qwen
```

Keys with underscores are converted to hyphens for the CLI flag name.

## 3. Resolve image

If the user provided an `image`, use it. Otherwise, build and push an image automatically:

### 3a. Verify registry credentials

```bash
echo $LMDEPLOY_REGISTRY
echo $REGISTRY_USER
test -n "$REGISTRY_PASSWORD" && echo "<set>" || echo "<missing>"
```

If any are missing, stop and tell the user to set them.

### 3b. Compute image tag

```bash
BRANCH=$(git branch --show-current | sed 's/[^a-zA-Z0-9._-]/-/g')
SHA=$(git rev-parse --short=7 HEAD)
TAG="${BRANCH}-${SHA}"
IMAGE="${LMDEPLOY_REGISTRY}/lmdeploy:${TAG}"
```

### 3c. Build (patch mode)

```bash
docker build -f docker/Dockerfile_patch \
  --build-arg BASE_IMAGE=openmmlab/lmdeploy:v0.12.3.post2-cu12.8 \
  --build-arg BACKEND=pytorch \
  --build-arg http_proxy=${http_proxy:-} \
  --build-arg https_proxy=${https_proxy:-} \
  --build-arg no_proxy=${no_proxy:-} \
  -t "${IMAGE}" \
  .
```

### 3d. Push

```bash
echo "${REGISTRY_PASSWORD}" | docker login "${LMDEPLOY_REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
docker push "${IMAGE}"
```

## 4. Resolve datasets

Parse the comma-separated `datasets` input. For each key, look up its value in `~/.eval/config`. If a key is not found, stop and list available dataset keys.

Combine the values into `subdataset`: `[*val1, *val2, ...]`

## 5. Compute derived fields

Compute the timestamp-padded model name:

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_ABBR_PADDED="${model_abbr}-${TIMESTAMP}"
```

Compute `infer_extra_params` from all model.yaml fields except `model_path` (see step 2 for the mapping rule).

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

If the model has a `reasoning_parser` field, add to `extra_body`:

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
