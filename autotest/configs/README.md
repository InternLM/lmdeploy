# autotest/configs

Per-model matrix configuration for autotest.

Each file maps to one HuggingFace model id:

- `configs/<org>/<model>.yml` -> `<org>/<model>`

## Add A New Model (Fast Path)

Use this workflow when adding any new model:

1. Create `autotest/configs/<org>/<model>.yml`.
2. Add header: `# HF model id: <org>/<model>`.
3. Start with one minimal row in one environment.
4. Add extra rows only when behavior differs.
5. Run:
   - `python autotest/tools/normalize_model_configs.py`
6. Verify model selection under your target `TEST_ENV` (and optional `DEPS_PROFILE`).

## Minimal Template

```yaml
# HF model id: org/model-name

a100:
- model_type: chat
  engine_config:
    tp: 1
  backends:
  - name: turbomind
    communicators: [nccl]
  - name: pytorch
    communicators: [nccl]
  test_coverage:
  - func
```

## When To Split Rows

Create separate rows only if one or more of these differ:

- `model_type` (`chat` / `vl` / `base`)
- `engine_config` layout (`tp`, `dp`+`ep`, `cp`+`tp`)
- `backends` or communicators
- `test_coverage`
- `interface`
- `quantization`
- `engine_config.extra`
- `gen_config`
- `deps`

If a row is truly identical for multiple profiles, use:

```yaml
model_type: [chat, vl]
```

## Entry Schema

Each top-level environment key contains a list of matrix rows.

| Field                 | Required | Description                                       |
| --------------------- | -------- | ------------------------------------------------- |
| `model_type`          | Yes      | `chat`, `vl`, `base`, or a list                   |
| `engine_config`       | Yes      | Parallel layout (`tp` / `dp` / `ep` / `cp`)       |
| `backends`            | Yes      | Backend list (explicit communicators recommended) |
| `test_coverage`       | Yes      | Enabled test groups                               |
| `interface`           | No       | Per-backend REST interface suites                 |
| `quantization`        | No       | Backend-specific quant flags                      |
| `engine_config.extra` | No       | Launch/runtime flags                              |
| `gen_config`          | No       | Request/eval sampling params                      |
| `deps`                | No       | Per-row dependency pins                           |

## Environment Keys

Common environment keys:

- `a100`
- `h`
- `3090`
- `5080`
- `ascend`
- `test` / `testascend` (fixtures)

Environment paths and devices are defined in `autotest/env_paths.yml`.

## `test_coverage` (What Tests Run)

Available keys:

- `func`
- `evaluate`
- `benchmark`
- `longtext_benchmark`
- `longtext_evaluate`
- `mllm_evaluate`
- `prefix_cache`
- `quantization`

Rules:

- Keep MTP (speculative decoding) on its own row with `speculative-algorithm` in `engine_config.extra`; use `func` and/or `evaluate` in `test_coverage`.
- Use `prefix_cache` in `test_coverage`; do not add `enable-prefix-caching` manually to `engine_config.extra`.
- Use `quantization` in `test_coverage` only for runtime weight-quant rows (`awq`, `gptq`, `w8a8`).

## `interface` (REST interface coverage)

Sibling of `test_coverage`. Drives `daily_ete_test` job `test_restful` via
`autotest/tools/gen_interface_matrix.py`.

Launch params use the **same dict shape as `engine_config.extra`** (kebab-case
keys; `null` => boolean CLI flag). Formatting goes through `get_cli_str`, shared
with tools/pipeline.

Recommended form — each backend is `{suites, extra}`, or a **list of launch
profiles** when different suites need **different** `api_server` extras.

Same launch command → one profile (merge suites; no extra restart):

```yaml
interface:
  pytorch:
    suites: [base, logprob, anthropic]
    extra:
      logprobs-mode: raw_logprobs
  turbomind:
    suites: [base, logprob, anthropic]
    extra:
      logprobs-mode: raw_logprobs
```

Different launch extras (e.g. tool/reasoning parsers must not be on the
anthropic server) → list of profiles:

```yaml
interface:
  pytorch:
  - suites: [base, logprob, experts, toolcall, reasoning]
    extra:
      logprobs-mode: raw_logprobs
      enable-return-routed-experts: true
      tool-call-parser: qwen3coder
      reasoning-parser: default
  - suites: [anthropic]
    extra:
      logprobs-mode: raw_logprobs
      enable-return-routed-experts: true
  turbomind:
  - suites: [base, logprob, toolcall, reasoning]
    extra:
      logprobs-mode: raw_logprobs
      tool-call-parser: qwen3coder
      reasoning-parser: default
  - suites: [anthropic]
    extra:
      logprobs-mode: raw_logprobs
```

(YAML cannot have two sibling `suites:` keys under the same mapping; use a
list of profiles when extras differ. Profiles with identical `extra` are
merged at normalize time into one api_server phase.)

Shorthand (single suite list; missing launch keys filled from suite defaults):

```yaml
interface:
  pytorch: [base, logprob, experts]
  turbomind: [base, logprob]
```

If shorthand mixes `anthropic` with `toolcall`/`reasoning`, anthropic is
split into its own empty-extra phase so it does not share a server that must
carry tool/reasoning parsers from yaml.

Legacy single-dict form `{suites, extra, anthropic: {extra: ...}}` is still
accepted and normalized into the two-profile shape.

Merge order for each profile's `api_server` command (later wins):

1. suite defaults (`logprobs-mode` for logprob/experts; `enable-return-routed-experts` for experts suite only)
2. row `engine_config.extra` (shared with tools on the same row)
3. that profile's `extra`

`tool-call-parser` / `reasoning-parser` are **not** suite-defaulted — set them
in profile `extra` (or `engine_config.extra`) whenever `toolcall` / `reasoning`
is enabled.

Suites:

- `base` — chat/completions basic cases; generate without logprob/experts
- `logprob` — generate logprob cases
- `experts` — generate routed-experts cases
- `anthropic` — Anthropic Messages HTTP + SDK smoke (`RESTFUL_MODEL_LIST`). Share a profile with `base`/`logprob` when `extra` matches; otherwise its own profile **without** `tool-call-parser` / `reasoning-parser`.
- `toolcall` — `interface/restful/tool_parser/` (requires `tool-call-parser` in yaml `extra`; add `enable-return-routed-experts: true` when toolcall includes `@experts` cases)
- `reasoning` — `interface/restful/reasoning_parser/` (requires `reasoning-parser` in yaml `extra`)

Notes:

- Prefer putting launch flags in profile `extra` so interface does not
  pollute tools when sharing a row with `test_coverage: func`.
- Each distinct `extra` is one api_server phase (generic; not hardcoded to anthropic).
  Add more `- suites: [...]` / `extra:` blocks only when launch flags differ.
- Put tool-call / reasoning parsers only on profiles that need them (yaml only).
- Suite default for `experts` includes `enable-return-routed-experts`; set it explicitly on `toolcall` profiles when needed;
  yaml `extra` can still override if needed.
- Chat/vl rows run `chat_completions_v1` + `generate` when any of base/logprob/experts is set.
- Base-only rows run `completions_v1`.
- Generate logprob/experts stay in one file and use pytest marks (`-m`).
- Put `interface` on the main functional row only; do not put it on MTP-only rows.
- Daily GPU-concurrent entrypoint (tools-style `gpu_num_*` + xdist):
  `autotest/interface/restful/test_restful_interface_llm.py`
  (covers chat/base/vl via config `interface`; chat+vl rows use the chat path).
  Each case self-starts `api_server` via `worker_id` port/GPU isolation.
- Backend filter env: `INTERFACE_BACKENDS=pytorch` / `turbomind` / `pytorch,turbomind` (workflow `test_restful` sets it from input `backend` / matrix).
- Nested protocol suite concurrency: `INTERFACE_SUITE_WORKERS` (default `20`, same as historical `-n 20`); set `0`/`1` to disable.

## `quantization`

Backend-scoped format:

```yaml
quantization:
  turbomind: [awq, kvint4, kvint8]
  pytorch: [w8a8, kvint4]
```

Supported flags: `awq`, `gptq`, `w8a8`, `kvint4`, `kvint8`, `kvint42`, `fp8`.

For pre-quantized checkpoints (`AWQ`, `GPTQ`, `Int4` in model id):

- Do not include `quantization` in `test_coverage`.
- Do not include runtime weight flags (`awq`, `gptq`, `w8a8`, `fp8`) in `quantization`.
- KV flags can still exist.

## Context parallel (`cp`)

When `engine_config` includes `cp`, use **TurboMind only** — do not list `pytorch` under `backends` or `quantization`. Export/normalize tooling strips pytorch automatically if present.

## `engine_config.extra` vs `gen_config`

Use `engine_config.extra` for launch/runtime settings:

- `session-len`
- `cache-max-entry-count`
- `max-batch-size`
- `model-format`
- speculative decoding launch options

Use `gen_config` for request/evaluation sampling params (kebab-case YAML keys):

```yaml
gen_config:
  temperature: 0.6
  reasoning-effort: high
  top-p: 0.95
  top-k: 50
  min-p: 0.0
  chat-template-kwargs:
    enable_thinking: true
```

OpenCompass scalar controls (`query_per_second`, `max_out_len`, `max_seq_len`, `batch_size`) come from preset tables, not from YAML `gen_config`.

## `deps` and `DEPS_PROFILE`

Per-row dependency pin:

```yaml
deps:
  transformers: "4.57.6"
```

Filtering behavior:

- `DEPS_PROFILE` empty/unset: rows without dependency pins
- `DEPS_PROFILE=pkg==ver` (or multi-key selector): exact `deps` match
- `DEPS_PROFILE=all`: no dependency filtering

`TEST_ENV` and `DEPS_PROFILE` are independent selectors.

## Validation

Run after any model edit:

```bash
python autotest/tools/normalize_model_configs.py
```

This keeps formatting stable and validates schema consistency.
