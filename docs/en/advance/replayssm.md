# ReplaySSM for Gated DeltaNet

ReplaySSM is an inference optimization that caches recent state-space-model
(SSM) **inputs** and replays them from an older checkpoint, instead of writing
the full recurrent state to HBM after every token. The method was introduced in
the [ReplaySSM article][replayssm-blog] and
[reference implementation][replayssm-code]. LMDeploy applies it to the
[Gated DeltaNet][gdn-paper] layers used by Qwen3-Next and Qwen3.5-family
models.

## Status

ReplaySSM is an experimental, opt-in PyTorch CUDA feature and is disabled by
default. Supported standard-decode and `qwen3_5_mtp` configurations can be
enabled with `enable_replayssm=True`. Executor construction validates the
backend, engine role, ring geometry, speculative method, GPU architecture, and
model family, and fails early for unsupported configurations. With
`enable_replayssm=False`, executor startup and the existing baseline path are
unchanged.

Current validation is limited to:

- the PyTorch CUDA backend on SM90 GPUs;
- colocated `Hybrid` engines;
- Qwen3-Next and Qwen3.5-family Gated DeltaNet models; and
- standard autoregressive decoding or `qwen3_5_mtp` speculative decoding.

TurboMind, non-CUDA devices, P/D-disaggregated state transfer, external cache
connectors, and other recurrent architectures are not supported by this
checkpoint.

## Basic principle

### Baseline recurrent decode

A Gated DeltaNet layer maintains a large recurrent matrix `S`. In simplified
form, one token applies decay, computes a delta-rule correction, updates the
matrix, and reads the output:

```text
S_decay = exp(g) * S
d       = beta * (v - S_decay * k)
S_next  = S_decay + d * k^T
y       = S_next * q
```

The exact Gated DeltaNet formulation is described in the
[Gated DeltaNet paper][gdn-paper]. During conventional decode, every layer and
token reads and writes the full `S[HV,V,K]` matrix. This state traffic can
dominate a one-token decode step even though each token contributes only the
compact vectors `(d,k,g)`.

### Standard ReplaySSM

Standard ReplaySSM represents the logical recurrent state as:

```text
logical state = checkpoint S0 + committed (d,k,g) ring prefix + cursor
```

A normal step appends `(d,k,g)` to a length-`L` circular buffer and computes
the output from `S0` plus the buffered history without materializing and
storing a new full state. When the buffer becomes full, or when the cache
lifecycle requests a safe boundary, the buffered transitions are folded into
`S0` once and the ring is cleared. Full-state writes are therefore amortized
over `L` decode steps.

This is primarily a memory-traffic optimization, not a state-capacity
optimization: standard ReplaySSM retains one full checkpoint and adds a ring.
The extra ring can reduce the number of state-cache slots available under a
fixed GPU-memory budget.

### MTP ReplaySSM

Conventional Gated DeltaNet speculative verification may retain one full
recurrent-state snapshot for every verify position so rejection can restore
the last accepted position. With
`Tmax = num_speculative_tokens + 1`, that storage grows with `Tmax`.

MTP ReplaySSM instead keeps one FP32 checkpoint, compact transformed and raw
input rings, and a device cursor. Candidate records are tentative. After
rejection sampling, the cursor advances only by the accepted target-input
count; rejected suffix records remain invisible and are overwritten later.
When a flush is needed, LMDeploy replays committed raw inputs in their original
order to advance the checkpoint exactly. This replaces per-position full-state
snapshots and makes rollback a cursor operation. The ring-and-cursor design and
lifecycle constraints follow the original ReplaySSM work.

ReplaySSM does not change model weights, tokenization, sampling, or attention
KV-cache semantics. However, output reconstruction can use a different
floating-point accumulation order, so accuracy and MTP acceptance must be
qualified rather than assumed to be bitwise identical.

### Cached values and shapes

Use the following notation for one request slot and one Gated DeltaNet layer:

- `H`: query/key heads on the current TP rank;
- `HV`: value heads on the current TP rank;
- `K` and `V`: query/key and value head dimensions;
- `L`: ReplaySSM ring length; and
- `Tmax = num_speculative_tokens + 1` for MTP verification.

Each value head owns a state `S[HV,V,K]`. A group of `HV/H` value heads shares
one normalized key/query head. The tensors below are logical per-slot,
per-layer shapes; physical state-cache allocation also has slot and layer
dimensions. The packed cursor is instead one `[3]` record per slot shared by
all GDN layers.

Standard ReplaySSM keeps:

| Value | Shape | Dtype | Meaning |
|---|---:|---|---|
| checkpoint `S0` | `[HV,V,K]` | SSM-state dtype | State at the last flush boundary |
| `d_ring` | `[HV,L,V]` | SSM-state dtype | Delta-rule correction vectors `d_i` |
| `k_ring` | `[H,L,K]` | SSM-state dtype | L2-normalized keys shared by value-head groups |
| `g_ring` | `[HV,L]` | FP32 | Log-decays, with `alpha_i = exp(g_i)` |
| cursor | `[3]` | int32 | `[write_pos, cache_base, is_flush]` |

`cache_base` and `is_flush` are zero in the ordinary standard path. A normal
non-flush token appends `(d_i,k_i,g_i)`; it does not cache `q_i`, `v_i`, or
`beta_i`, because `q_i` only reads the state and `v_i/beta_i` have already
been absorbed into `d_i`.

MTP ReplaySSM additionally keeps raw inputs so a persistent checkpoint can be
folded with the original closed-loop recurrence:

| Value | Shape | Dtype | Meaning |
|---|---:|---|---|
| checkpoint `S0` | `[HV,V,K]` | FP32 | Last materialized committed state |
| `d_ring` | `[HV,L,V]` | activation dtype | Fast reconstruction correction vectors |
| `k_ring` | `[H,L,K]` | activation dtype | Fast reconstruction normalized keys |
| `g_ring` | `[HV,L]` | FP32 | Log-decays |
| `raw_v_ring` | `[HV,L,V]` | activation dtype | Raw value inputs for exact fold |
| `raw_k_ring` | `[H,L,K]` | activation dtype | Keys before L2 normalization |
| `beta_ring` | `[HV,L]` | FP32 | Writing strengths used to recompute `d_i` |
| cursor | `[3]` | int32 | Committed length, circular base, and next-step flush flag |

The transformed rings make normal verification/reconstruction fast. The raw
rings are the authority for checkpoint write-back. Queries are not cached
because they affect only `y_i`, not the transition from `S_{i-1}` to `S_i`.
Candidate MTP records may be written into the ring before rejection sampling,
but only the prefix covered by the committed cursor is visible.

`k_ring` is intentionally retained even though `raw_k_ring` exists. The hot
non-flush path consumes the already normalized `(d_ring,k_ring,g_ring)` closed
form on every verify step. The cold exact-fold path consumes
`(raw_v_ring,raw_k_ring,g_ring,beta_ring)` only when materializing a persistent
checkpoint. Replacing `k_ring` with `raw_k_ring` would require renormalizing
every cached key during every verify step, or adding an equivalent normalized
scratch buffer. It is therefore a possible memory/performance redesign, not a
free removal. The same hot/cold split explains `d_ring` versus `raw_v_ring`.

For [Qwen3.5-35B-A3B][qwen35-model] TP1, `H=16`, `HV=32`, and `K=V=128`,
with 30 GDN layers.
The qualified standard `L=16` shapes per layer/slot are
`S0[32,128,128]`, `d[32,16,128]`, `k[16,16,128]`, and `g[32,16]`.
For MTP3 with `L=8`, the corresponding shapes are `S0[32,128,128]`,
`d/raw_v[32,8,128]`, `k/raw_k[16,8,128]`, and `g/beta[32,8]`.
The existing causal-convolution state is separate: `[8192,4]` in standard
mode and `[8192,7]` for MTP3.

### Reconstructing and writing back `S`

For processed Qwen3.5 inputs, let

```text
k_hat_i = L2Norm(k_raw_i)
q_hat_i = L2Norm(q_raw_i) / sqrt(K)
alpha_i = exp(g_i)
S_i^-   = alpha_i * S_(i-1)
d_i     = beta_i * (v_i - S_i^- * k_hat_i)
S_i     = S_i^- + d_i * k_hat_i^T
y_i     = S_i * q_hat_i
```

This is the Gated DeltaNet recurrence. Here `d_i` is also called `u_i` in the
ReplaySSM article.

Suppose `S0` is the checkpoint and the standard ring contains `m` committed
records. Define `G_j = sum_(r=1)^j g_r`. Every intermediate logical state can
be reconstructed without the raw values:

```text
S_j = exp(G_j) * S0
      + sum_(i=1)^j exp(G_j - G_i) * d_i * k_hat_i^T,
      1 <= j <= m.
```

The standard kernel normally needs only contractions of this expression with
the current `q` and `k`, so it avoids storing an intermediate full matrix.
For the current token `c`:

```text
d_c = beta_c * (v_c - alpha_c * S_m * k_hat_c)
y_c = alpha_c * (S_m * q_hat_c)
      + d_c * (k_hat_c^T * q_hat_c)
```

On a full or forced flush, it materializes exactly one full matrix and writes:

```text
S0 <- alpha_c * S_m + d_c * k_hat_c^T
```

It then resets the standard ring cursor. Thus the full state is written once
per flush, rather than once per token.

MTP checkpoint write-back deliberately does **not** trust the low-precision
transformed `d_ring`. It reads only the committed raw records in circular-ring
order and performs a closed-loop FP32 fold:

```text
S <- S0
for i in committed_ring_order:
    k_hat_i = L2Norm(raw_k_i)
    S^-     = exp(g_i) * S
    d_i     = beta_i * (raw_v_i - S^- * k_hat_i)
    S       = S^- + d_i * k_hat_i^T
S0 <- S
```

The intermediate matrices exist only in kernel registers; only the final
`S0` is stored to HBM. During a verify window, the same recurrence produces
the candidate intermediate states and outputs in order, but its ring writes
remain tentative. After rejection sampling, the device commit kernel advances
the cursor by the accepted target-input count. Rejected suffix records are
therefore ignored. If a flush is pending, only the previously committed raw
prefix is folded before the new speculative window is written.

## Mode selection

LMDeploy selects the ReplaySSM protocol from the existing speculative
configuration; there is no separate MTP ReplaySSM switch.

| Configuration | Behavior |
|---|---|
| `enable_replayssm=False` | Existing baseline path |
| Replay enabled, no speculative config | Standard ReplaySSM |
| Replay enabled with `qwen3_5_mtp` | MTP ReplaySSM |
| Replay enabled with another speculative method | Startup error |

## Basic usage

### Command line

Standard-decode interface:

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 16
```

Qwen3.5 MTP interface with three draft tokens:

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 8 \
  --speculative-algorithm qwen3_5_mtp \
  --speculative-num-draft-tokens 3
```

### Python API

Standard decode:

```python
from lmdeploy import PytorchEngineConfig, pipeline

def main():
    backend_config = PytorchEngineConfig(
        enable_replayssm=True,
        replayssm_buffer_len=16,
    )
    pipe = pipeline('/path/to/qwen3.5-model', backend_config=backend_config)
    # Use pipe(...), then close it when finished.
    pipe.close()


if __name__ == '__main__':
    main()
```

MTP decode:

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

def main():
    backend_config = PytorchEngineConfig(
        enable_replayssm=True,
        replayssm_buffer_len=8,
    )
    speculative_config = SpeculativeConfig(
        method='qwen3_5_mtp',
        num_speculative_tokens=3,
    )
    pipe = pipeline(
        '/path/to/qwen3.5-model',
        backend_config=backend_config,
        speculative_config=speculative_config,
    )
    # Use pipe(...), then close it when finished.
    pipe.close()


if __name__ == '__main__':
    main()
```

### Choosing the buffer length

`replayssm_buffer_len` accepts `8`, `16`, or `32` and defaults to `16`.

- A larger ring flushes less often but reads more history per decode step and
  consumes more state-cache memory.
- A smaller ring reduces ring memory and replay work but materializes the full
  checkpoint more frequently.
- For MTP, LMDeploy requires
  `L >= 2 * (num_speculative_tokens + 1)`. MTP3 therefore requires at least
  `L=8`.
- The current H200 qualification uses `L=16` for standard decode and `L=8` for
  MTP3. These are validated starting points, not universal optima.

LMDeploy logs the selected mode, aligned bytes per state slot, total state-cache
bytes, and the effective cache geometry during initialization. Compare these
values with the flag-off configuration before choosing deployment concurrency.

### Rollback

Set `enable_replayssm=False` or remove `--enable-replayssm` to return to the
existing recurrent-state implementation. The flag-off path remains the
correctness and performance control.

## References

- [ReplaySSM: Cache SSM Inputs, Not State][replayssm-blog]
- [ReplaySSM reference implementation][replayssm-code]
- [Gated Delta Networks: Improving Mamba2 with Delta Rule][gdn-paper]
- [Qwen3.5-35B-A3B model configuration][qwen35-model]

[replayssm-blog]: https://dao-lab.ai/blog/2026/replayssm/
[replayssm-code]: https://github.com/Johnny-Liou/ReplaySSM
[gdn-paper]: https://arxiv.org/abs/2412.06464
[qwen35-model]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
