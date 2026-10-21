# GLM-5.3 FlashKDA prefill

GLM-5.3 uses the public `flash_kda.fwd()` API for KDA prefill recurrent computation on the deployment configuration below. Ordinary prefill passes `None` for both checkpoint arguments; a caller requesting an internal recurrent checkpoint supplies both buffers to the same API. Each call processes the complete prefill interval.

```text
Kda → CudaKdaImpl → flash_kda.fwd → torch.ops.flash_kda.fwd
```

## Deployment scope

| Property | Direct FlashKDA path |
|---|---|
| Model | GLM-5.3 (`glm5_next` KDA layers) |
| GPU | NVIDIA Hopper, compute capability 9.0 |
| Inputs | BF16 Q, K, V, raw gate and raw beta |
| Head dimensions | K = V = 128 |
| Recurrent cache | FP32, value-first `[N, H, V, K]` |
| Gate | FP32 `A_log` / `dt_bias`, lower bound in `[-5, 0]` |

Model and device properties select this path automatically. There is no FLA/native prefill switch. `FLA_FLASH_KDA` does not control the direct path. Other models and configurations retain their existing FLA prefill dispatch. FLA 0.5.2 remains required for convolution and gate operations. AR decode and MTP verification retain the shared TileLang recurrence and their existing state-ring behavior.

## Install the public checkpoint API

The native checkpoint ABI is available in FlashKDA commit `ee0be888cd0e972f9409bf53756f8c38c6652173`, but that commit's Python wrapper omits the checkpoint arguments. Apply the accompanying [public API patch](../../../requirements/patches/flash_kda_checkpoint_api.patch) before building it. The patch only extends the Python wrapper, its documentation and CPU contract tests; CUDA/C++ sources remain unchanged.

This revision uses the PyTorch stable extension API and requires PyTorch 2.10 or newer, CUDA Toolkit 12.9 or newer, and a compatible C++ compiler. Run the installation in the environment used by the LMDeploy workers:

```bash
# Set this to the absolute path of the LMDeploy checkout containing this patch.
LMDEPLOY_SRC=/path/to/lmdeploy
git clone https://github.com/MoonshotAI/FlashKDA.git
cd FlashKDA
git checkout ee0be888cd0e972f9409bf53756f8c38c6652173
git apply "$LMDEPLOY_SRC/requirements/patches/flash_kda_checkpoint_api.patch"
git submodule update --init --recursive
FLASH_KDA_CUDA_ARCHS=90a pip install --no-build-isolation --no-deps .
```

LMDeploy checks the public function signature before modifying a prefill cache. A missing package or an older wrapper produces an actionable error. It does not bypass the wrapper with a private operator call or silently switch implementations on an in-scope deployment.

## Recurrent checkpoint contract

`Kda.forward()` and `CudaKdaImpl.forward()` accept two optional keyword arguments:

- `checkpoint_state`: caller-owned, contiguous CUDA FP32 output of shape `[N, H, V, K]`.
- `checkpoint_offsets`: caller-owned, contiguous CUDA integer tensor of shape `[N]`, with the same dtype as `metadata.cu_seqlens` (`int32` or `int64`).

Provide both or neither. Each positive offset is the number of tokens processed within that sequence's current forward interval, divisible by 16 and no greater than the sequence length. Zero disables capture for a sequence and leaves its checkpoint buffer unchanged. The caller must validate values before dispatch; the interface avoids reading GPU offsets on the CPU. Checkpoint output buffers must be separate from live input state. For an absolute checkpoint position `H` in a forward starting at `start`, the offset is `H - start`.

The normal output and final recurrent state still represent the complete interval. A checkpoint request on decode or outside the deployment scope fails explicitly. MTP callers supply a contiguous view of the reserved checkpoint ring slot; normal runtime state continues to advance to the full-prefill end slot.

The `main` baseline for this change does not include the IC scheduler from [PR #5048](https://github.com/InternLM/lmdeploy/pull/5048). These arguments export recurrent state only. The IC caller remains responsible for the matching convolution snapshot, reservation, publication, restoration and MTP lookahead checks. This change does not enable merged IC scheduling or alter prefix-cache policy.

## Verification

```bash
# From the LMDeploy checkout, with the patched FlashKDA installed:
python -m pytest tests/pytorch/test_kda_prefill.py -q
python -m pytest tests/pytorch/kernel/test_kda.py -q

# From the patched FlashKDA checkout (does not require a GPU):
python -m pytest tests/test_public_api.py -q
```

The CPU tests cover dispatch scope, public API compatibility, optional buffer forwarding, unchanged FLA flags and MTP ring addressing. CUDA tests cover ordinary versus checkpoint-enabled full prefill, exported state versus a prefix forward, resumed suffixes, restored short prefills and partial MTP acceptance. Numerical tests require a compatible GPU and the patched extension; API contract tests alone do not establish model-level accuracy or performance.
