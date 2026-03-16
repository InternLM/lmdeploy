# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Linting:**

```bash
pre-commit run --all-files
```

Style: PEP8, max line length 120, double quotes, LF endings. C++ source under `src/` uses clang-format.

**Tests:**

```bash
pytest tests/test_lmdeploy                          # all unit tests
pytest tests/test_lmdeploy/test_model.py            # specific file
pytest tests/test_lmdeploy/test_lite/               # quantization tests
pytest tests/test_lmdeploy/test_vl/                 # vision-language tests
```

**Debug logging:**

```bash
LMDEPLOY_LOG_LEVEL=DEBUG python ...
```

**Build (TurboMind C++ extension):**

- Controlled via `setup.py` + CMake. Relevant env vars: `LMDEPLOY_TARGET_DEVICE` (default `cuda`), `DISABLE_TURBOMIND`, `CMAKE_BUILD_TYPE`, `CUDACXX`.
- Requirements split by device: `requirements/runtime_cuda.txt`, `runtime_ascend.txt`, etc.

## Architecture

### Two Backends, One Pipeline

`lmdeploy/pipeline.py` is the main user-facing entry point (`pipeline()` in `api.py`). It instantiates either the **PyTorch engine** (`lmdeploy/pytorch/`) or the **TurboMind engine** (`lmdeploy/turbomind/`) based on config.

### PyTorch Backend

**Model patching** is the core mechanism: HuggingFace models are loaded normally, then their layers are dynamically replaced with optimized LMDeploy implementations.

- `lmdeploy/pytorch/models/module_map.py` — registry mapping HF class names → LMDeploy replacement classes. Device-specific overrides in `DEVICE_SPECIAL_MODULE_MAP`.
- `lmdeploy/pytorch/models/patch.py` — applies the substitutions at runtime via `_get_rewrite_qualname()` / `_class_from_qualname()`.
- `lmdeploy/pytorch/models/` — 40+ per-model files (e.g., `llama.py`, `qwen.py`, `deepseek_v2.py`). Each reimplements attention, MLP, and embeddings using custom kernels.
- `lmdeploy/pytorch/nn/` — reusable optimized modules: `linear/` (AWQ, W8A8, blocked-FP8, LoRA variants), `attention.py`, `norm.py`, `rotary_embedding.py`, `moe/`.
- `lmdeploy/pytorch/kernels/` — Triton/CUDA kernels (e.g., `w8a8_triton_kernels.py`).
- `lmdeploy/pytorch/backends/` — kernel/operator dispatchers per quantization type (FP8, AWQ, CUDA).

**Engine execution flow:**

1. `engine.py` owns the main PyTorch engine.
2. `engine/executor/` — pluggable executors: `uni_executor.py` (single process), `mp_executor.py` (multiprocess), `ray_executor.py`.
3. `paging/scheduler.py` — schedules sequences into batches; manages prefill/decode phases, block eviction, prefix caching (`BlockTrie`).
4. `paging/block_manager/` — GPU/CPU KV-cache block allocation (physical ↔ logical mapping).
5. `engine/cache_engine.py` — KV cache lifecycle; supports 0/4/8-bit quantization policies.
6. `engine/engine_loop.py` — async inference loop.
7. `engine/input_process.py` / `inputs_maker.py` — tokenized inputs → model tensors.
8. `engine/logits_process.py` — sampling, temperature, stop-word filtering.
9. `model_inputs.py` — `StepContext` / `StepContextManager` passed to model on each forward step.

**Configuration dataclasses** (`lmdeploy/pytorch/config.py`): `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `BackendConfig`, `DistConfig`, `MiscConfig`.

### TurboMind Backend

- Python wrapper: `lmdeploy/turbomind/turbomind.py` (~800 lines). Bridges into `lmdeploy/lib/_turbomind` (pybind11 extension built from `src/turbomind/`).
- Tensor interop via `torch.from_dlpack()` / `_tm.from_dlpack()`.
- Config and model conversion: `lmdeploy/turbomind/deploy/config.py`, `supported_models.py`.
- Parallel config helpers: `update_parallel_config()`, `complete_parallel_config()` in `messages.py`.

### Lite / Quantization

Entrypoints in `lmdeploy/lite/apis/`: `calibrate.py` (main), `auto_awq.py`, `gptq.py`, `smooth_quant.py`.

**Flow:** load HF model → `CalibrationContext` collects activation statistics → scale computation (`lmdeploy/lite/quantization/`) → write quantized weights.

- `lite/quantization/awq.py` — AWQ (NORM_FCS_MAP, FC_FCS_MAP define per-model layer structure).
- `lite/quantization/weight/quantizer.py` — weight quantizer.
- `lite/quantization/activation/observer.py` — activation statistics.
- `lite/modeling/` — model-specific GPTQ implementations (e.g., `internlm2_gptq.py`).
- `lite/utils/cal_qparams.py` — quantization parameter calculation utilities.

Layer/norm/head mappings per model family are defined directly in `calibrate.py` and `awq.py`.

### Vision-Language Models

- `lmdeploy/vl/model/` — VLM preprocessing (InternVL, Qwen-VL, LLaVA, CogVLM, etc.).
- `lmdeploy/vl/media/` — image/video loaders and base classes.
- `lmdeploy/pytorch/multimodal/` — multimodal input handling for the PyTorch engine.
- Reference VLM implementation: `lmdeploy/vl/model/qwen3.py`.

### Other Key Files

- `lmdeploy/messages.py` — core types: `GenerationConfig`, `EngineConfig`, `TurbomindEngineConfig`, `SchedulerSequence`, `MessageStatus`.
- `lmdeploy/model.py` — chat templates; critical for correct conversation formatting.
- `lmdeploy/archs.py` — architecture registry mapping model arch names to runtime patches.
- `lmdeploy/tokenizer.py` — HuggingFace/SentencePiece tokenizer wrapper.
- `lmdeploy/serve/openai/` — OpenAI-compatible API server.

## Adding a New PyTorch Model

1. Create `lmdeploy/pytorch/models/<model>.py` implementing the patched attention/MLP layers.
2. Register the HF class names in `lmdeploy/pytorch/models/module_map.py`.
3. If quantization support is needed, add layer/norm/head mappings in `lmdeploy/lite/apis/calibrate.py` and `lmdeploy/lite/quantization/awq.py`.
4. For VLM support, add preprocessing in `lmdeploy/vl/model/<model>.py`.
5. Reference model: `lmdeploy/pytorch/models/qwen3.py`.
