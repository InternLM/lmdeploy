# GLM-5.3 FlashKDA prefill

GLM-5.3 uses the native `torch.ops.flash_kda.fwd()` operator for KDA prefill recurrent computation on the deployment configuration below. Ordinary prefill passes `None` for both checkpoint arguments; a caller requesting an internal recurrent checkpoint supplies both buffers to the same operator. Each call processes the complete prefill interval.

```text
Kda → CudaKdaImpl._prefill_recurrent → torch.ops.flash_kda.fwd
```

## Deployment scope

| Property             | Direct FlashKDA path                               |
| -------------------- | -------------------------------------------------- |
| Current model caller | GLM-5.3 (`glm5_next` KDA layers)                   |
| GPU                  | NVIDIA Hopper, compute capability 9.0              |
| Inputs               | BF16 Q, K, V, raw gate and raw beta                |
| Head dimensions      | K = V = 128                                        |
| Recurrent cache      | FP32, value-first `[N, H, V, K]`                   |
| Gate                 | FP32 `A_log` / `dt_bias`, lower bound in `[-5, 0]` |

Tensor and device properties select this path automatically, with hardware support cached by CUDA device index. The backend does not take a model identifier. GLM-5.3 is currently the only model using the shared `Kda` module; other models retain their existing operator paths. There is no FLA/native prefill switch, and `FLA_FLASH_KDA` does not control the direct path. KDA configurations outside the supported tensor/device contract retain their existing FLA prefill dispatch. FLA 0.5.2 remains required for convolution and gate operations. AR decode and MTP verification retain the shared TileLang recurrence and their existing state-ring behavior.

## FlashKDA dependency

Install a FlashKDA build whose native `torch.ops.flash_kda.fwd` schema includes `checkpoint_state` and `checkpoint_offsets`. LMDeploy loads the installed `flash_kda._C` extension to register the operator, then calls it directly for both ordinary and checkpoint-enabled prefill. The public Python wrapper does not need to expose these arguments. No FlashKDA source changes or patches are required.

The native checkpoint ABI was validated at the unmodified upstream [FlashKDA commit `ee0be888cd0e972f9409bf53756f8c38c6652173`](https://github.com/MoonshotAI/FlashKDA/commit/ee0be888cd0e972f9409bf53756f8c38c6652173). Its Python wrapper omits these arguments, which does not affect this path. Pin that revision when installing for the deployment configuration above:

```bash
FLASH_KDA_CUDA_ARCHS=90a python -m pip install --no-build-isolation --no-deps \
    'git+https://github.com/MoonshotAI/FlashKDA.git@ee0be888cd0e972f9409bf53756f8c38c6652173'
```

That native revision uses the PyTorch stable extension API and requires PyTorch 2.10 or newer, CUDA Toolkit 12.9 or newer, and a compatible C++ compiler. Set `CUDA_HOME` to the toolkit location when needed. Check the native schema in the environment used by every LMDeploy worker:

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

LMDeploy checks the native schema before modifying a prefill cache. A missing extension or an operator without checkpoint support produces an actionable error; there is no silent implementation fallback on an in-scope deployment. `CudaKdaImpl` allocates a contiguous uint8 workspace using `torch.ops.flash_kda.get_workspace_size()`, taking over the allocation normally performed by the Python wrapper. Output and final-state buffers are allocated in the same method; the caller owns checkpoint buffers.

## Recurrent checkpoint contract

`Kda.forward()` and `CudaKdaImpl.forward()` accept two optional keyword arguments:

- `checkpoint_state`: caller-owned, contiguous CUDA FP32 output of shape `[N, H, V, K]`.
- `checkpoint_offsets`: caller-owned, contiguous CUDA integer tensor of shape `[N]`, with the same dtype as `metadata.cu_seqlens` (`int32` or `int64`).

Provide both or neither. Each positive offset is the number of tokens processed within that sequence's current forward interval, divisible by 16 and no greater than the sequence length. Zero disables capture for a sequence and leaves its checkpoint buffer unchanged. The caller must validate values before dispatch; the interface avoids reading GPU offsets on the CPU. Checkpoint output buffers must be separate from live input state. For an absolute checkpoint position `H` in a forward starting at `start`, the offset is `H - start`.

The normal output and final recurrent state still represent the complete interval. A checkpoint request on decode or outside the deployment scope fails explicitly. MTP callers supply a contiguous view of the reserved checkpoint ring slot; normal runtime state continues to advance to the full-prefill end slot.

The `main` baseline for this change does not include the IC scheduler from [PR #5048](https://github.com/InternLM/lmdeploy/pull/5048). These arguments export recurrent state only. The IC caller remains responsible for the matching convolution snapshot, reservation, publication, restoration and MTP lookahead checks. This change does not enable merged IC scheduling or alter prefix-cache policy.

## Verification

```bash
# From the LMDeploy checkout, with a compatible FlashKDA build installed:
python -m pytest tests/pytorch/test_kda_prefill.py -q
python -m pytest tests/pytorch/kernel/test_kda.py -q
```

The CPU tests cover tensor/device dispatch, device capability caching across backend instances, native schema compatibility, workspace allocation, optional buffer forwarding, unchanged FLA flags and MTP ring addressing. CUDA tests cover ordinary versus checkpoint-enabled full prefill, exported state versus a prefix forward, disabled capture at offset zero, resumed suffixes, restored short prefills and partial MTP acceptance. The checkpoint tests reject calls to the public Python wrapper. Numerical tests require a compatible GPU and FlashKDA build; API contract tests alone do not establish model-level accuracy or performance.
