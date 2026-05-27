# autotest/configs

Per-model YAML: `configs/<org>/<model-name>.yml` → HF id `<org>/<model-name>`.

## 环境

| 环境     | 说明      | `deps.transformers` 默认                                                      |
| -------- | --------- | ----------------------------------------------------------------------------- |
| `a100`   | CUDA A100 | 镜像默认；legacy 矩阵为同环境下**另一条** list（`deps.transformers: 4.57.6`） |
| `h`      | H 集群    | 同上                                                                          |
| `3090`   | RTX 3090  | 同上                                                                          |
| `5080`   | RTX 5080  | 同上                                                                          |
| `ascend` | 昇腾 NPU  | 镜像默认                                                                      |

环境级文件：

- `../env_paths.yml`（`autotest/env_paths.yml`）— 各环境路径与 `device`

合并规则：

- **`deps` 只写在模型 YAML 条目上**；无 `deps` 表示使用 CI 镜像默认依赖
- **无 `a100_legacy` 顶层 key**：`config_legacy.yml` 等合并进 `a100` 的 list，与 `config.yml` 行靠 `deps` / `backends` / `test_coverage` 区分
- **`DEPS_PROFILE`**（环境变量）：按条目 `deps` 筛选矩阵；默认空=仅无 deps 条目（见下）

### `engine_config.extra`（服务启动 / CLI 参数）

直接写在模型 YAML 的 `engine_config.extra` 中，由 `config_utils` 在展开 run_config 时读入 `extra_params`。例如 `session-len`、`cache-max-entry-count`、`max-batch-size`、`model-format`、投机解码相关项等。

导出/规范化时，`export_model_configs.resolve_launch_extra_overrides()` 会按模型名、环境、`engine_config` 布局、`test_coverage` 合并历史 launch 规则（原在 loader 里对 `run_config['extra_params']` 打补丁的逻辑），写入各条 `engine_config.extra`。

```yaml
engine_config:
  dp: 8
  ep: 8
  extra:
    session-len: 65536
    cache-max-entry-count: 0.9
    max-batch-size: 256
```

### `gen_config`（请求 / 评测采样参数）

写在**模型 YAML 条目**上（kebab-case）。运行时 `get_eval_preset_config()` 优先使用 `run_config['gen_config']`。

- OC 评测参数（`query-per-second` / `max-out-len` / `max-seq-len` / `batch-size`）不应出现在模型 YAML 的 `gen_config`
- **模型相关** OpenAI / 采样字段（原 `openai_extra_kwargs` / `extra_body`）须写在本条 `gen_config`，例如：

```yaml
gen_config:
  temperature: 0.6
  reasoning-effort: high          # gpt-oss
  top-p: 0.95
  top-k: 50
  min-p: 0.0                      # Intern-S1-Pro
  chat-template-kwargs:
    enable_thinking: true         # Qwen3.5
```

**不要**把 `session-len` 写在 `gen_config`；**不要**把 `temperature` 写在 `engine_config.extra`。

## 单条 list 项

| 字段                  | 说明                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `model_type`          | **字符串或列表**：`chat` / `vl` / `base`；列表表示同一矩阵复用到多种 profile（展开为多条 run_config） |
| `engine_config`       | `tp` / `dp`+`ep` / `cp`+`tp`                                                                          |
| `backends`            | 支持两种写法：`[turbomind, pytorch]`；或冗余 communicator 的对象数组（见下）                          |
| `test_coverage`       | 见下表                                                                                                |
| `quantization`        | 各 backend 启用的量化：`awq` `gptq` `w8a8` `kvint4` `kvint8` `kvint42`                                |
| `engine_config.extra` | 服务启动 / CLI 参数（写在模型 YAML 中）                                                               |
| `gen_config`          | 请求/评测采样（`EVAL_CONFIGS` / `MLLM_EVAL_CONFIGS`）                                                 |
| `deps`                | 可选；Python 依赖覆盖，见下表                                                                         |

### `test_coverage` 与条目拆分

| function             | 对应用例                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| `func`               | pipeline / restful / chat 接口功能                                                 |
| `evaluate`           | API 评测                                                                           |
| `benchmark`          | 吞吐 / apiserver benchmark                                                         |
| `longtext_benchmark` | 长上下文 benchmark                                                                 |
| `mllm_evaluate`      | 多模态评测                                                                         |
| `mtp_evaluate`       | MTP 推测解码评测                                                                   |
| `quantization`       | 仅当条目含 **AWQ / GPTQ / W8A8** 时添加；KVINT 只写在 `quantization:` 块，不占此项 |

**已量化 HF 权重**（模型 id 含 `AWQ` / `GPTQ` / `Int4` 等）：`test_coverage` **不含** `quantization`；`quantization` 块**不写** `awq` / `gptq` / `w8a8` / `fp8`（仅保留 KV 类如 `kvint4` / `kvint8`）。导出与 `normalize_model_configs.py` 会自动清理。
| **`prefix_cache`**   | **前缀缓存**：`benchmark/test_prefixcache_*`、`test_*_prefix_cache_tp2` 等                                                                                |
| **`mtp_evaluate`**   | **MTP 推测解码评测**：须单独一条矩阵（`test_coverage` 仅含此项），`engine_config.extra` 含 `speculative-algorithm` 等；勿与 `evaluate`/`func` 合并 dedupe |

**拆分原则**（与现有 autotest 一致）：

1. `test_coverage` 列出该矩阵要跑的全部用例（可含 `prefix_cache` / `benchmark` / `func` / …）；**不要**在 `engine_config.extra` 里写 `enable-prefix-caching`（由 loader 根据 `prefix_cache` 注入 CLI）。
2. `prefix_cache` 与当前条目的 `engine_config` 一致（写在同一条 `test_coverage` 里），不再单独拆 `tp: 2` 行。
3. **不支持 prefix cache 的模型**（如含 `Qwen3.5` 且引擎限制）不要写 `prefix_cache`。

### `backends` 冗余 communicator

推荐在模型 yaml 中显式冗余 communicator，减少 loader 推断分支：

```yaml
backends:
  - name: turbomind
    communicators: [nccl, cuda-ipc]
  - name: pytorch
    communicators: [nccl]
```

兼容旧写法 `backends: [turbomind, pytorch]`；`config_utils.py` 会自动回退推断 communicator。

矩阵排除规则写在模型 YAML（`#` 注释掉不跑的 backend/communicator），**不要**在 `config_utils.py` 里硬编码：

| 规则                                                          | YAML 写法                                                                   |
| ------------------------------------------------------------- | --------------------------------------------------------------------------- |
| InternVL3 / InternVL2_5 / InternVL2-Llama3 + turbomind + tp>1 | `# - cuda-ipc`                                                              |
| Qwen2.5-VL / Qwen2-VL                                         | 注释掉整个 `turbomind` backend，保留 `pytorch`                              |
| vl + `mllm_evaluate` + tp>1                                   | 拆条：`vl` 条目仅 `pytorch`；`chat` 可保留 turbomind                        |
| Qwen3.5 + prefix_cache                                        | 文件头注释 + export `PREFIX_EXCLUDE_SUBSTR`（不写 `prefix_cache` function） |
| phi + vl_model                                                | 不提供 vl 矩阵（见 `microsoft/Phi-4-mini-instruct.yml` 注释）               |

### `model_type` 用 list 的时机

当 **同一环境** 下 chat / vl 的 `engine_config`、`backends`、`test_coverage`（除 profile 专属项外）、`quantization` 一致时，合并为一项：

```yaml
model_type: [chat, vl]
```

若 chat 与 vl 的 `test_coverage` 或 `backends` 不同，仍拆成两条（或 `model_type: chat` / `model_type: vl`）。

### `deps`（条目依赖）

写在模型 YAML 的 list 项上，展开 run_config 时进入 `run_config['deps']`。当前以 **`transformers`** 为主：

```yaml
deps:
  transformers: "4.57.6"
```

| 条目 `deps`                                                           | 含义               |
| --------------------------------------------------------------------- | ------------------ |
| 省略                                                                  | 使用 CI 镜像内版本 |
| `transformers: "4.57.6"`                                              | legacy 用例        |
| `transformers: "git+https://github.com/huggingface/transformers.git"` | Qwen3.5 等新架构   |

CI 示例：`pip install "${TRANSFORMERS_SPEC}"`（从 `run_config['deps']['transformers']` 读取）。

### `DEPS_PROFILE`（依赖包筛选）

| `DEPS_PROFILE`                | 行为                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| **未设置或空**                | 仅跑条目 **无 `deps` 块**（或 `deps` 里全是 `null`、无 `profile`）的矩阵            |
| **`transformers==4.57.6`** 等 | 跑条目 `deps` 与选择器 **完全一致** 的矩阵（pip 风格 `pkg==ver`；多键用空格或 `;`） |
| **`all`**                     | 不做 deps 筛选（调试用）                                                            |

```bash
# 默认 CI：镜像默认 transformers，条目不写 deps
export TEST_ENV=a100
unset DEPS_PROFILE

# legacy transformers 矩阵
export DEPS_PROFILE='transformers==4.57.6'
```

`TEST_ENV` 与 `DEPS_PROFILE` **独立**：

- `TEST_ENV`：硬件/集群环境（`a100` / `3090` / `5080`），对应 `autotest/env_paths.yml`
- `DEPS_PROFILE`：筛选模型矩阵（见上表）

CI 示例（`daily_ete_test*.yml`）：

```yaml
strategy:
  matrix:
    deps_profile: ['', 'transformers==4.57.6']
env:
  TEST_ENV: a100
  DEPS_PROFILE: ${{ matrix.deps_profile }}
```

矩阵值与 `pip install` 参数一致，CI 可直接：

```yaml
- name: Install pinned deps (DEPS_PROFILE)
  if: matrix.deps_profile != ''
  run: pip install ${{ matrix.deps_profile }}
```

## 批量生成

从根目录 `config*.yml` 导出全部模型 YAML：

```bash
python autotest/tools/export_model_configs.py
```

来源映射：

| 源文件                   | 模型 YAML 顶层 key                |
| ------------------------ | --------------------------------- |
| `config.yml`             | `a100`                            |
| `config_h.yml`           | `h`                               |
| `config_3090.yml`        | `3090`                            |
| `config_5080.yml`        | `5080`                            |
| `config_legacy.yml`      | 并入 `a100` list（条目带 `deps`） |
| `config_h_legacy.yml`    | 并入 `h` list                     |
| `config_3090_legacy.yml` | 并入 `3090` list                  |
| `config_5080_legacy.yml` | 并入 `5080` list                  |

同时更新 `autotest/env_paths.yml`。
