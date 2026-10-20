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
