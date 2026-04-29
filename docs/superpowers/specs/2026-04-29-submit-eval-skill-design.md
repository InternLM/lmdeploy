# submit-eval Skill Design

## Overview

A Claude Code skill that submits a model evaluation task to the auto-eval platform. The API URL is read from `~/.eval/config`. It reads model and dataset config from user-maintained files, computes derived fields, optionally builds a Docker image via the `docker-build` skill, and submits the request via curl.

## User Inputs

| Input        | Required | Example                                 | Notes                                                                             |
| ------------ | -------- | --------------------------------------- | --------------------------------------------------------------------------------- |
| `model_abbr` | Yes      | `Qwen3.5-35B`                           | Key in `~/.eval/models/model.yaml`; padded with `-yyyymmdd-hhmmss` at submit time |
| `instances`  | Yes      | `3`                                     | Number of inference instances; used to compute `end_num`                          |
| `datasets`   | Yes      | `mmlu_pro, ifeval, aime2026`            | Comma-separated keys (looked up in `~/.eval/config`)                              |
| `image`      | No       | `<your-registry>/lmdeploy:main-abc1234` | Docker image. If omitted, triggers `docker-build` skill                           |

## Config Files

### `~/.eval/config` (KEY=VALUE)

```
AUTO_EVAL_TOKEN=<your-token>
FEISHU_EVAL_WEBHOOK=<your-feishu-webhook-url>
USER=lvhan
OPENAI_API_BASE=<your-api-base-url>
AUTO_EVAL_API_URL=<your-auto-eval-api-url>
mmlu_pro=*mmlu_pro_datasets
ifeval=*ifeval_datasets
aime2026=*aime2026_datasets
```

### `~/.eval/models/model.yaml` (YAML)

```yaml
Qwen3.5-35B:
  model_path: /mnt/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6
  tp: 2
  backend: turbomind
  reasoning_parser: qwen-qwq
  # tool_call_parser: ...   (optional)

Qwen3-32B:
  model_path: /mnt/shared-storage-gpfs2/.../snapshots/...
  tp: 2
  backend: turbomind
  reasoning_parser: qwen-qwq
```

## Hardcoded Defaults

| Field               | Default                 |
| ------------------- | ----------------------- |
| `job_name`          | `api_eval_v4`           |
| `cluster`           | `yidian`                |
| `workspace_id`      | `evalservice_gpu`       |
| `eval_type`         | `chat_objective`        |
| `auto_eval_version` | `ld_0122_oc_0524d49_v2` |
| `ocp_version`       | `fullbench_v2_0`        |
| `fast_infer`        | `true`                  |
| `eval_only`         | `false`                 |
| `parallelism`       | `TP`                    |
| `infer_engine`      | `lmdeploy`              |
| `delete`            | `false`                 |
| `start_infer`       | `true`                  |
| `node_num`          | `1`                     |
| `oc_cpu`            | `1`                     |
| `oc_mem`            | `4000`                  |
| `infer_worker_nums` | `8`                     |
| `eval_nums`         | `15`                    |
| `llm_judger_config` | `""`                    |
| `cli_extra`         | `""`                    |

## Derived Fields

| Field                       | Derivation                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `model_abbr` (padded)       | `{model_abbr}-{yyyymmdd-hhmmss}` using current timestamp at submit time                                                  |
| `subdataset`                | From `datasets` input: look up each key in `~/.eval/config`, combine values as `[*val1, *val2, ...]`                     |
| `infer_extra_params`        | `--tp {tp} --backend {backend} --reasoning-parser {reasoning_parser}` + optional `--tool-call-parser {tool_call_parser}` |
| `gpu_num`                   | = `tp`                                                                                                                   |
| `cpu`                       | = `16 * tp`                                                                                                              |
| `memory`                    | = `"128000 * tp"` (as string, e.g. tp=2 → `"256000"`)                                                                    |
| `end_num`                   | = `instances + 1`                                                                                                        |
| `tokenizer_path`            | = `model_path`                                                                                                           |
| `output_dir`                | `./{user}/{model_abbr_padded}`                                                                                           |
| `model_infer_config`        | Assembled dict string (see below)                                                                                        |
| `model_infer_config_base64` | `echo -n '{model_infer_config}' \| base64`                                                                               |
| `infer_backend_config`      | Assembled dict (see below)                                                                                               |

### model_infer_config structure

Assembled as a Python dict string with:

- `type`: `OpenAISDKStreaming`
- `key`: `sk-admin`
- `openai_api_base`: `['{OPENAI_API_BASE}']` (from `~/.eval/config`)
- `query_per_second`: `8`
- `batch_size`: `32`
- `max_workers`: `8`
- `temperature`: `1`
- `tokenizer_path`: `{model_path}`
- `retry`: `50`
- `max_out_len`: `128000`
- `max_seq_len`: `128000`
- `extra_body`: `{top_k: 20, repetition_penalty: 1.0, top_p: 0.95, chat_template_kwargs: {enable_thinking: True}}`
  - If `reasoning_parser` is set, include `chat_template_kwargs.enable_thinking: True`; otherwise omit
- `pred_postprocessor`: `{type: 'extract-non-reasoning-content'}` (only if `reasoning_parser` is set)
- `verbose`: `True`

### infer_backend_config structure

Assembled as a dict with:

- `end_num`: `{instances + 1}`
- `gpu_num`: `{tp}`
- `memory`: `"{128000 * tp}"`
- `cpu`: `{16 * tp}`
- `parallelism`: `TP`
- `oc_cpu`: `1`
- `oc_mem`: `4000`
- `model`: `{model_abbr_padded}`
- `model_path`: `{model_path}`
- `image`: `{image}`
- `infer_engine`: `lmdeploy`
- `infer_extra_params`: `{infer_extra_params}`
- `delete`: `false`
- `start_infer`: `true`
- `node_num`: `1`

## Flow

1. **Read `~/.eval/config`** — verify `AUTO_EVAL_TOKEN`, `FEISHU_EVAL_WEBHOOK`, `USER`, `OPENAI_API_BASE`, `AUTO_EVAL_API_URL` are present. Stop if missing.
2. **Gather inputs** — ask user for `model_abbr`, `instances`, `datasets`, and optionally `image`.
3. **Look up model** — read `~/.eval/models/model.yaml`, find the entry for `model_abbr`. Stop if not found.
4. **Resolve image** — if user provided `image`, use it; else invoke the `docker-build` skill to build and push an image from the current branch.
5. **Resolve datasets** — parse comma-separated `datasets` input, look up each key in `~/.eval/config`, combine into `subdataset` value.
6. **Compute derived fields** — pad `model_abbr` with timestamp, assemble `infer_extra_params`, `gpu_num`, `cpu`, `memory`, `end_num`, `model_infer_config`, `model_infer_config_base64`, `infer_backend_config`, `output_dir`.
7. **Assemble payload** — build the full JSON body with defaults + user inputs + computed fields.
8. **Submit** — execute `curl -X POST` to `AUTO_EVAL_API_URL` with the payload and `AUTO_EVAL_TOKEN` as Bearer token. Report the response.

## Skill File Location

`/workspace/lmdeploy/.claude/skills/submit-eval/SKILL.md`

## Error Handling

- Missing `~/.eval/config` or incomplete keys → stop and tell user to create/populate it
- Model not found in `~/.eval/models/model.yaml` → stop and list available models
- Dataset key not found in `~/.eval/config` → stop and list available dataset keys
- curl failure → report the HTTP status and response body
