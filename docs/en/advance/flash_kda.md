# GLM-5.3 FlashKDA prefill

Ordinary prefill remains available through official FLA 0.5.2 when the optional native FlashKDA checkpoint operator cannot be loaded or has an incompatible schema. On the supported configuration below, a compatible native operator is preferred for both ordinary and checkpoint-enabled prefill. Internal checkpoint export requires that operator; the validated implementation currently comes from the **vLLM fork of FlashKDA**. MoonshotAI/FlashKDA `master` at `7afb9f454f160a6c4bbc0999beca0a8c40a38934` does not provide it.

When the compatible native operator is available on the deployment configuration below, GLM-5.3 uses `torch.ops.flash_kda.fwd()` for KDA prefill recurrent computation. Ordinary prefill passes `None` for both checkpoint arguments; a caller requesting an internal recurrent checkpoint supplies both buffers to the same operator. Each call processes the complete prefill interval.

```text
Kda → CudaKdaImpl._prefill_recurrent
    → torch.ops.flash_kda.fwd  (compatible native operator available)
    → FLA chunk_kda           (ordinary prefill fallback)
```

This interface exports recurrent state only. Full IC scheduling and model-to-scheduler checkpoint publication are not enabled by installing these dependencies.

## Deployment scope

| Property             | Direct FlashKDA path                               |
| -------------------- | -------------------------------------------------- |
| Current model caller | GLM-5.3 (`glm5_next` KDA layers)                   |
| GPU                  | NVIDIA Hopper, compute capability 9.0              |
| Inputs               | BF16 Q, K, V, raw gate and raw beta                |
| Head dimensions      | K = V = 128                                        |
| Recurrent cache      | FP32, value-first `[N, H, V, K]`                   |
| Gate                 | FP32 `A_log` / `dt_bias`, lower bound in `[-5, 0]` |

Tensor/device properties and native operator availability select this path automatically, with hardware support cached by CUDA device index and optional operator availability cached per backend instance. The backend does not take a model identifier. GLM-5.3 is currently the only model using the shared `Kda` module; other models retain their existing operator paths. There is no new FLA/native prefill switch. `FLA_FLASH_KDA` retains its existing meaning inside FLA's fallback dispatch and does not control the direct native path. Ordinary prefill outside the supported tensor/device contract also retains FLA dispatch. FLA 0.5.2 remains required for convolution and gate operations. AR decode and MTP verification retain the shared TileLang recurrence and their existing state-ring behavior.

## FlashKDA dependency

`flash-linear-attention` and `FlashKDA` are separate packages. `requirements/runtime_cuda.txt` pins official FLA 0.5.2 for the existing FLA operations; it does not install FlashKDA or enable internal checkpoints. The [FLA 0.5.2 FlashKDA adapter](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/kda/backends/flash_kda.py) accepts extra keyword arguments but does not forward `checkpoint_state` or `checkpoint_offsets` to `flash_kda.fwd`. Setting `state_v_first=True` selects the state layout, and `output_final_state=True` requests the state at the end of the interval; neither exports an internal checkpoint.

The [MoonshotAI FlashKDA interface at `7afb9f4`](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/flash_kda/__init__.py) calls the `flash_kda_C` extension. Its [native interface](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/flash_kda.cpp) has no checkpoint arguments. This version cannot supply the native operator used by the LMDeploy path above. Upgrading FLA alone does not add that native capability.

### Validated fork revision

The native checkpoint ABI was validated using [vllm-project/FlashKDA commit `ee0be888cd0e972f9409bf53756f8c38c6652173`](https://github.com/vllm-project/FlashKDA/commit/ee0be888cd0e972f9409bf53756f8c38c6652173), merged into that fork's `dev` branch by [PR #7](https://github.com/vllm-project/FlashKDA/pull/7). The tested checkout had no local source modifications. This commit is from the vLLM fork, not MoonshotAI's `master` branch.

To reproduce the native-kernel validation with that fork, install the pinned revision explicitly. This separate installation is required only for the native path described here and is not added to LMDeploy's common CUDA requirements:

```bash
FLASH_KDA_CUDA_ARCHS=90a python -m pip install --no-build-isolation --no-deps \
    'git+https://github.com/vllm-project/FlashKDA.git@ee0be888cd0e972f9409bf53756f8c38c6652173'
```

LMDeploy loads this fork's `flash_kda._C` extension and checks that the native `torch.ops.flash_kda.fwd` schema includes `checkpoint_state` and `checkpoint_offsets`. It calls that operator directly for both ordinary and checkpoint-enabled prefill. The fork's public Python wrapper omits these arguments, so it is not used for recurrent prefill.

That native revision uses the PyTorch stable extension API and requires PyTorch 2.10 or newer, CUDA Toolkit 12.9 or newer, and a compatible C++ compiler, alongside LMDeploy's own dependency constraints. Set `CUDA_HOME` to the toolkit location when needed. Check the native schema in the environment used by every LMDeploy worker:

```bash
python - <<'PY'
import torch
from flash_kda import _C  # Load the extension and register its operators.

required = {'checkpoint_state', 'checkpoint_offsets'}
schema = torch.ops.flash_kda.fwd.default._schema
assert required <= {argument.name for argument in schema.arguments}, \
    'FlashKDA must provide native checkpoint support'
print(schema)
PY
```

LMDeploy checks checkpoint buffer metadata and the native schema before modifying a prefill cache. Ordinary prefill returns to the existing FLA dispatch when the extension cannot load or the checkpoint schema is unavailable. Explicit checkpoint requests fail before cache mutation in those cases. A native kernel execution error is propagated and never retried through FLA after computation has started. `CudaKdaImpl` allocates a contiguous uint8 workspace using `torch.ops.flash_kda.get_workspace_size()`, taking over the allocation normally performed by the Python wrapper. Output and final-state buffers are allocated in the same method; the caller owns checkpoint buffers.

## Recurrent checkpoint contract

`Kda.forward()` and `CudaKdaImpl.forward()` accept two optional keyword arguments:

- `checkpoint_state`: caller-owned, contiguous CUDA FP32 output of shape `[N, H, V, K]`.
- `checkpoint_offsets`: caller-owned, contiguous CUDA integer tensor of shape `[N]`, with the same dtype as `metadata.cu_seqlens` (`int32` or `int64`).

Provide both or neither. Before convolution updates the cache, the backend validates checkpoint dtype, shape, device and contiguity, as well as the corresponding `cu_seqlens` metadata. The lower-level fixed-batch recurrent interface requires int64 checkpoint offsets when `cu_seqlens` is absent. Each positive offset is the number of tokens processed within that sequence's current forward interval, divisible by 16 and no greater than the sequence length. Zero disables capture for a sequence and leaves its checkpoint buffer unchanged. The caller must validate offset values using CPU metadata before dispatch; the interface avoids reading GPU offsets on the CPU. Checkpoint output buffers must be separate from live input state. For an absolute checkpoint position `H` in a forward starting at `start`, the offset is `H - start`.

The normal output and final recurrent state still represent the complete interval. A checkpoint request on decode or outside the deployment scope fails explicitly. MTP callers must supply a contiguous checkpoint output buffer; normal runtime state continues to advance to the full-prefill end slot. A common slot view such as `ring[:, slot]` from a multi-request `[N, Ring, H, V, K]` cache is generally non-contiguous and is rejected before cache updates. Batched IC integration must supply a contiguous scratch buffer and scatter the results to the reserved request/ring slots. That allocation, scatter and checkpoint publication remain caller responsibilities.

The `main` baseline for this change does not include the IC scheduler from [PR #5048](https://github.com/InternLM/lmdeploy/pull/5048). These arguments export recurrent state only. The IC caller remains responsible for the matching convolution snapshot, reservation, publication, restoration and MTP lookahead checks. This change does not enable merged IC scheduling or alter prefix-cache policy.

## Verification

```bash
# From the LMDeploy checkout, with a compatible FlashKDA build installed:
python -m pytest tests/pytorch/test_kda_prefill.py -q
python -m pytest tests/pytorch/kernel/test_kda.py -q
```

The CPU tests cover tensor/device dispatch, device capability caching across backend instances, optional dependency fallback, native schema compatibility, checkpoint metadata rejection before cache mutation, workspace allocation, optional buffer forwarding, unchanged FLA flags and MTP ring addressing. Extension-loading failures and old schemas are simulated; these tests do not validate a separate CUDA 12.8 or PyTorch 2.9 installation. CUDA tests check that the real FLA fallback exactly matches direct FLA outputs/final states at 129 and 2048 tokens, and record cross-kernel RMS and maximum absolute differences from native FlashKDA as JUnit properties. They also cover ordinary versus checkpoint-enabled full prefill, exported state versus a prefix forward, disabled capture at offset zero, resumed suffixes, restored short prefills and partial MTP acceptance. The checkpoint tests reject calls to the public Python wrapper. Native H200 validation uses the vLLM fork revision above. These kernel tests do not establish full-model accuracy, TTFT, throughput or workspace peaks at long context lengths.
