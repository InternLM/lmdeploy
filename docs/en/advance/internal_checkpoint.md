# GLM-5.3 native KDA prefill backend

The default `LMDEPLOY_KDA_PREFILL_BACKEND=fla` keeps the existing FLA entry
point. FLA may dispatch to its own FlashKDA backend when its conditions are met.
Set `LMDEPLOY_KDA_PREFILL_BACKEND=flash_kda` to explicitly select LMDeploy's
native FlashKDA adapter instead. This does not change the decode backend.

The native adapter requires BF16 inputs, head dimension 128, FP32 recurrent
state, and a gate lower bound. The installed FlashKDA operator must expose
`checkpoint_state` and `checkpoint_offsets`; incompatible installations fail
the capability check instead of silently switching backends.

The validated FlashKDA source revision is
`ee0be888cd0e972f9409bf53756f8c38c6652173`, with Torch 2.10.0+cu128 and
flash-linear-attention 0.5.2 on H200. No FlashKDA, FLA, convolution, KPool, or
TileLang decode kernel source is modified by this integration.

For controlled comparisons, select the same prefill backend in both arms.
Use `FLA_FLASH_KDA=0` when an explicit FLA Triton reference is required.
Selecting the native adapter does not by itself enable internal checkpoints.

## Internal prefill checkpoints

Set `LMDEPLOY_INTERNAL_PREFILL_CHECKPOINT=1` together with the native backend
and local prefix caching to enable IC. IC is disabled by default.

The scheduler first reserves the same safe checkpoint H as the original
split-prefill path. If the remaining prompt fits the existing prefill budget,
it can expand that forward to the prompt end E while retaining its reservation
at H. FlashKDA exports the recurrent state directly into that checkpoint slot;
the backend copies the preceding convolution-input window into its conv state.
The normal runtime state still advances to E. An endpoint copy must not
overwrite the internal checkpoint.

The capture description contains the input start/end, checkpoint position, and
source/destination slots. Public engine components transport this description
and retain existing reservation, pinning, publication, and cleanup mechanisms.
GLM-specific code maps slots to per-layer KDA output views. KPool runtime tails
remain excluded from checkpoint copies, as defined by PR #5037.

Initial support is non-MTP, text-only, exclusive single-sequence prefill with
one internal save. A non-MTP prompt end that also requires an aligned checkpoint
keeps the original split rather than dropping either save. Connector/Mooncake,
multimodal, scoring, mixed local decode/prefill, insufficient capacity, and
unsupported capture layouts do not enter the IC path. Invalid enabled-backend
configurations fail validation; per-request ineligibility preserves the
ordinary split path.

Merged intervals within `piecewise_cudagraph_max_tokens` retain the original
path. Current IC runs outside prefill PCG; it does not disable decode graphs.
`prefill_save_alignment` constrains absolute checkpoint positions, including
prompt-end saves, but does not guarantee a save at every aligned position.

Comparisons must hold the backend, prefill budget, PCG cap, alignment, and
request set fixed. Export/copy work and a larger single forward can consume
time and memory; IC is not a zero-cost operation or a universal speedup.
