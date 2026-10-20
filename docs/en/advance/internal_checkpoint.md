# GLM-5.3 internal prefill checkpoints

IC avoids a model forward used only to reach a checkpoint boundary. It is
disabled by default and supports non-MTP AR and `deepseek_mtp`.

## Enablement

Use local prefix caching (`--enable-prefix-caching`) and set:

```bash
export LMDEPLOY_KDA_PREFILL_BACKEND=flash_kda
export LMDEPLOY_INTERNAL_PREFILL_CHECKPOINT=1
```

The default backend remains the FLA entry, which may dispatch to FlashKDA.
Native selection alone does not enable IC or change decode. The native adapter
requires BF16 inputs, head dimension 128, FP32 recurrent state, a gate lower
bound, and operator outputs `checkpoint_state` / `checkpoint_offsets`.
Incompatible enabled configurations fail validation rather than silently fall back.

Validated FlashKDA revision: `ee0be888cd0e972f9409bf53756f8c38c6652173`,
Torch 2.10.0+cu128 and FLA 0.5.2 on H200. No kernel sources are modified.

## Execution and safety

The scheduler retains the original reservation at H and, when capacity permits,
expands execution to prompt end E within the existing prefill budget:

```text
OFF: forward [start,H) -> save S_H -> forward [H,E)
 ON: forward [start,E), exporting S_H internally; runtime advances to S_E
```

FlashKDA writes recurrent state into the checkpoint slot. Conv state comes from
the preceding projected-input window. Existing pinning, publication and cleanup
are reused; endpoint copying must not overwrite the internal state. KPool raw
tails remain runtime-only.

For MTP K, operator offset is `H - start`, recurrent destination is `H % (K+1)`,
and conv window `[H-width,H)` is placed in the `width+K` ring, wrapping if needed.
Draft shifting, known-prompt lookahead checks and accepted-history recovery are
unchanged. Predicting a next token does not make a prompt-end MTP checkpoint
publishable; MTP decode checkpoint saving remains disabled.

## Limits and validation

- Only exclusive single-sequence text prefill with one internal save is supported.
- Required non-MTP endpoint saves retain the original split; no checkpoint is dropped.
- Connector/Mooncake, multimodal, scoring/routed-expert output, mixed local
  decode/prefill, insufficient capacity and unsupported layouts do not enter IC.
- Intervals at or below `piecewise_cudagraph_max_tokens` retain the original path.
  IC currently runs outside prefill PCG; decode graphs remain enabled.
- `prefill_save_alignment` filters absolute internal and endpoint save positions,
  including with IC OFF; aligned positions may be skipped. Decode intervals are unchanged.
- Comparisons must fix backend, budget, PCG cap, alignment and requests. Export
  work and larger forwards cost time/memory; IC is not a universal speedup.
- Tests cover alignment, fallback, restore and K=1/3/5 rings/partial acceptance;
  model checks cover K=3 at TP1/DP8/EP8, not full IC accuracy or production stress.
