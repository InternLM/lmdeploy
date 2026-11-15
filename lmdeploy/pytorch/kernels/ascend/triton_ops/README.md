# Triton Operators for Qwen3.5 Inference on Ascend NPU

This directory contains custom Triton kernels optimized for running Qwen3.5 (linear attention variant) inference on Huawei Ascend NPUs.

## Directory Structure

```
triton_ops/
├── __init__.py                    # Operators are imported explicitly from their submodules
├── causal_conv1d.py               # Causal 1D convolution (prefill + decode)
├── rms_norm_gated.py              # Gated RMSNorm with SiLU or sigmoid activation
├── triton_utils.py                # NPU device property helpers
└── fla/
    ├── __init__.py                # FLA submodule entry
    ├── chunk.py                   # Chunked gated delta rule (prefill)
    └── sigmoid_gating.py          # Fused sigmoid-gated delta rule (decode)
```

## Exported Operators

| Operator                                 | File                    | Phase   | Description                                     |
| ---------------------------------------- | ----------------------- | ------- | ----------------------------------------------- |
| `causal_conv1d_fn`                       | `causal_conv1d.py`      | Prefill | Varlen causal conv1d using PyTorch              |
| `causal_conv1d_update_npu`               | `causal_conv1d.py`      | Decode  | Tiled stateful causal conv1d Triton kernel      |
| `chunk_gated_delta_rule`                 | `fla/chunk.py`          | Prefill | Chunked gated delta rule attention              |
| `fused_sigmoid_gating_delta_rule_update` | `fla/sigmoid_gating.py` | Decode  | Fused sigmoid-gated delta rule recurrent update |
| `rmsnorm_fn`                             | `rms_norm_gated.py`     | Both    | Gated RMSNorm kernel                            |

______________________________________________________________________

## Operator Details

### 1. `causal_conv1d_fn` & `causal_conv1d_update_npu`

**File**: `causal_conv1d.py`

**Source**:

- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/mamba/causal_conv1d.py
- Original: https://github.com/Dao-AILab/causal-conv1d

**Purpose**:
Implements causal 1D convolution for the convolutional path in Qwen3.5's linear attention mechanism. The causal nature ensures each output position only depends on current and past inputs (no future leakage).

**Two Functions**:

1. `causal_conv1d_fn` (Prefill): Pure PyTorch implementation using `F.conv1d`, handles variable-length sequences via `query_start_loc`
2. `causal_conv1d_update_npu` (Decode): Tiled Triton kernel (`_causal_conv1d_update_kernel_npu_tiled`) for single-token or multi-token generation with state management

**Decode Kernel Features** (`causal_conv1d_update_npu`):

- **Tiled execution**: `B_TILE` × `BLOCK_N` tiling for efficient NPU utilization (targeting 2×40 AI cores)
- **Variable-length sequences** (`IS_VARLEN`): driven by `query_start_loc`
- **Speculative decoding** (`IS_SPEC_DECODING`): handles multiple accepted tokens via `num_accepted_tokens`
- **Automated Page Caching** (`IS_APC_ENABLED`): supports non-contiguous state management via `block_idx_last_scheduled_token` and `initial_state_idx`
- **Kernel widths 1–6**: compile-time unrolled convolution loop
- **Optional SiLU activation**

______________________________________________________________________

### 2. `chunk_gated_delta_rule`

**File**: `fla/chunk.py`

**Source**:

- Kernel: https://gitcode.com/Ascend/triton-ascend-kernels/blob/master/src/triton_ascend_kernels/attention/fla/
- Original: https://github.com/fla-org/flash-linear-attention (MIT)

**Purpose**:
Implements the core linear attention operation for prefill phase using chunked parallel computation. The "gated delta rule" is a variant of linear attention that:

- Uses gating to control information flow
- Applies delta rule for state updates (subtracting projected keys from values)
- Processes sequences in chunks for parallelism

**Wrapper Role**:
`fla/chunk.py` is a thin wrapper that handles input validation, dtype conversion (to bfloat16), optional Q/K L2 normalization (`use_qk_l2norm_in_kernel`) via `triton_ascend_kernels.norm.l2norm.l2norm_fwd`, and delegates to `triton_ascend_kernels.attention.fla.chunk_gated_delta_rule_fwd`.

______________________________________________________________________

### 3. `fused_sigmoid_gating_delta_rule_update`

**File**: `fla/sigmoid_gating.py`

**Source**:

- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/sigmoid_gating.py
- Original: https://github.com/fla-org/flash-linear-attention (MIT)
- Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

**Purpose**:
Implements the decode-phase recurrent update for the gated delta rule using a fully fused Triton kernel. This replaces a separate torch_npu op with a native Triton kernel that runs the entire recurrent loop in a single pass.

**Two Kernels**:

1. `fused_recurrent_gated_delta_rule_fwd_kernel`: General recurrent kernel supporting continuous batching, speculative decoding, and variable-length sequences
2. `fused_sigmoid_gating_delta_rule_update_kernel`: Decode-optimized kernel that fuses sigmoid gating computation with the recurrent delta rule update

**Key Operations (per token step)**:

1. Compute gate: `g = -exp(A_log) * softplus(a + dt_bias)` (numerically stable)
2. Compute beta: `beta = sigmoid(b)`
3. Apply L2 norm to Q/K if `use_qk_l2norm_in_kernel`
4. Decay hidden state: `h *= exp(g)`
5. Delta rule: `v -= sum(h * k, dim=0)`
6. Update state: `h += k[:, None] * (v * beta)[None, :]`
7. Output: `o = sum(h * q, dim=0)`

**Features**:

- **Variable-length sequences** (`IS_VARLEN`): driven by `cu_seqlens`
- **Continuous batching**: reads/writes hidden states via `h0_indices` for in-place state management
- In-place state update: the final hidden state is written back to `initial_state_source` indexed by `h0_indices`

______________________________________________________________________

### 4. `rmsnorm_fn`

**File**: `rms_norm_gated.py`

**Source**:

- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/layernorm_guard.py
- Original: https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py

**Purpose**:
Implements RMSNorm (Root Mean Square Normalization) with optional SiLU (`silu`/`swish`) or sigmoid gating.
The public `lmdeploy.pytorch.nn.RMSNormGated` layer owns the weight and selects an implementation through
`RMSNormGatedBuildSpec`; the Ascend backend calls this kernel. Qwen3-Next/Qwen3.5 use SiLU gating,
while GLM5-Next uses sigmoid gating.

**Why Required**:

- Qwen3.5 uses gated normalization in its architecture
- Standard PyTorch normalization doesn't support gating
- Fused kernel is more efficient than separate norm + gate + mul

**Formula**:

- Without gate: `y = (x / sqrt(mean(x^2) + eps)) * weight`
- With gate after normalization: `y = norm(x) * activation(z)` (`norm_before_gate=True`, used by the public layer)
- With gate before normalization: `y = norm(x * activation(z))` (`norm_before_gate=False`)
- The kernel also preserves optional bias, grouped normalization, and the ungated path.

______________________________________________________________________

### 5. `triton_utils.py` (Helper)

**Source**:

- Adapted from: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/triton_utils.py

**Purpose**:
Provides NPU device property queries for Triton kernel tuning:

- `init_device_properties_triton()`: Initialize device properties
- `get_aicore_num()`: Get number of AI cores
- `get_vectorcore_num()`: Get number of vector cores

**Why Required**:

- Ascend NPUs have different core counts (AI cores vs vector cores)
- Some kernels use this information for optimal grid sizing
- Called during backend initialization

______________________________________________________________________

## Data Flow in Qwen3.5 Inference

### Prefill Phase

```
Input tokens
    ↓
[Causal Conv1D] ← causal_conv1d_fn
    ↓
[Linear Attention] ← chunk_gated_delta_rule (triton_ascend_kernels)
    ↓
[Gated RMSNorm] ← RMSNormGated → rmsnorm_fn
    ↓
Output logits
```

### Decode Phase

```
New token(s)
    ↓
[Causal Conv1D Update] ← causal_conv1d_update_npu
    ↓
[Sigmoid-Gated Delta Rule Update] ← fused_sigmoid_gating_delta_rule_update
    ↓
[Gated RMSNorm] ← RMSNormGated → rmsnorm_fn
    ↓
Output logits
```

______________________________________________________________________

## External Dependencies

| Package                 | Usage                                                          |
| ----------------------- | -------------------------------------------------------------- |
| `triton`                | Kernel DSL and JIT compilation                                 |
| `triton_ascend_kernels` | Official Ascend-optimized kernels (prefill attention, L2 norm) |
| `torch`                 | Tensor operations, NPU device management                       |
| `torch_npu`             | Ascend NPU backend for PyTorch                                 |

______________________________________________________________________

## Installation

### Installing `triton-ascend-kernels`

The core attention kernels (`chunk_gated_delta_rule_fwd`, `l2norm_fwd`) are provided by the official `triton-ascend-kernels` package.

**LMDeploy runtime requirements**:

Use `requirements/runtime_ascend.txt` and `requirements/ascend_kernels.txt`
from the repository root. The native backend requires matching torch/torch_npu
major.minor versions, torch >= 2.8.0, torch_npu >= 2.8.0.post1, and
`torch.npu.NPUGraph.update` (`backends/ascend/version.py:23`). The old 2.6
installation recipe is not supported by this checkout. Record the actual
`triton-ascend-kernels` Git commit in the A3 validation environment; the imported
chunk/L2Norm kernels are an external dependency, not code supplied by dlinfer.

See the [Ascend installation guide](../../../../../docs/en/get_started/ascend/get_started.md#native-a3-installation).

**Install from source**:

```bash
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
cd triton-ascend-kernels
pip install -e .
```

**Verify installation**:

```python
from triton_ascend_kernels.attention.fla import chunk_gated_delta_rule_fwd
from triton_ascend_kernels.norm.l2norm import l2norm_fwd

print("triton-ascend-kernels installed successfully!")
```

For more details, see: https://gitcode.com/Ascend/triton-ascend-kernels

______________________________________________________________________

## License

All kernels are Apache-2.0 licensed. Original sources are attributed in individual file headers.
