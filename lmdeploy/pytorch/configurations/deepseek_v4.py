# Copyright (c) OpenMMLab. All rights reserved.
# DeepSeek-V4-Flash model config builder.
#
# V4 attention is NOT standard MLA (no kv_lora_rank): single shared KV head
# (num_key_value_heads=1, head_dim=512), q_lora/o_lora projections, o_groups,
# and a DSA (Dynamic Sparse Attention) KV cache.  Therefore build() does NOT
# reuse DeepseekV2ModelConfigBuilder.build (which assumes kv_lora_rank); it
# constructs ModelConfig directly with V4-correct head dims.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder
from .deepseek_v2 import DeepseekV2ModelConfigBuilder


def _check_env_v4(device: str = 'cuda'):
    """V4 on Ascend relies on dlinfer-loaded _C_ascend kernels. Nothing to
    assert here; op registration is checked at runtime."""
    return


class DeepseekV4ModelConfigBuilder(DeepseekV2ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return getattr(hf_config, 'model_type', None) == 'deepseek_v4'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, is_draft_model: bool = False,
              spec_method: str = None, tp: int = 1, **kwargs):
        """build V4 ModelConfig directly (no V2 MLA head_dim)."""
        num_attention_heads = hf_config.num_attention_heads
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads', 1) or 1
        head_dim = getattr(hf_config, 'head_dim', 512)
        num_layers = hf_config.num_hidden_layers

        # V4 has a single shared KV head (num_key_value_heads=1, is_tp=False on
        # wkv).  Under TP it is replicated: bump num_key_value_heads to tp so
        # the engine's head-sharding assertion (num_kv_heads % world_size == 0)
        # holds and each rank keeps one replica of the swa_kv cache.
        num_key_value_heads = cls.update_num_kv_heads(hf_config, tp, num_key_value_heads)

        model_paradigm = 'ar'
        if is_draft_model:
            num_layers = hf_config.num_nextn_predict_layers
            hf_config.architectures[0] = 'DeepseekMTPModel'
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map
            model_paradigm = 'ar_spec'
        if spec_method is not None:
            assert spec_method == 'deepseek_mtp'
            model_paradigm = 'ar_spec'

        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=head_dim,
            k_head_dim=head_dim,
            v_head_dim=head_dim,
            vocab_size=hf_config.vocab_size,
            use_flash_mla=False,
            model_paradigm=model_paradigm,
            # Propagate the SWA sliding window (config.json: sliding_window=128)
            # so ExecutorBase sets cache_config.window_size>0. This selects
            # WindowBlockManager (real eviction: old swa_kv blocks are freed,
            # only the last `window` tokens stay resident) instead of the
            # append-only DefaultBlockManager, and trips the unlimited branch
            # in _get_max_session_len -> session_len is no longer capped to the
            # (tiny) swa_kv pool capacity (~9k tok) but to the configured
            # --session-len. Without this, long-thinking decodes >9k are
            # truncated (GPQA 49/198 cut at ~9000 -> acc 72.7% vs ~87%).
            # Safe because npu_sparse_attn_sharedkv in ori_mask_mode=4 only
            # reads window blocks (proven by vllm-ascend AscendSlidingWindowMLASpec
            # evicting swa_kv and running); the compressor recurses on state_cache
            # (not swa_kv), and the indexer's top-k draws from compress_kv.
            # FIXED (0903): sliding_window eviction is now SAFE.
            # 0903 root cause: compress_kv AND indexer_k REUSED swa_block_table
            # (v4_dsa L1783/1791) -- they shared the swa physical blocks + the
            # EVICTED block table. When swa freed block P (cross-request reuse),
            # compress_kv[P] still held the prior request's compressed KV; the
            # indexer's sparse top-k read (NOT bounded by seqused_kv) read it ->
            # foreign text bled into output (GPQA batch-16 11.11%). vllm-ascend
            # avoids this via SEPARATE caches (Compress/Indexer/SWA each own a
            # KVCacheSpec+pool).
            # FIX: compress_kv/indexer_k now own a SEPARATE non-evicted pool +
            # slot-stable full-history block table (v4_dsa dsa_kv_bt, a
            # _V4StateAlloc with state_window=0 -- absolute-position-indexed,
            # own pool tensors, swa eviction never touches them). swa_kv keeps
            # its windowed evicted pool/table (SWA only needs the window).
            # CRITICAL: the compressor path (state_bt + dsa_kv_bt) is absolute-
            # position-indexed, so start_pos / indexer_kvlens MUST be the FULL
            # (pre-offset, absolute) kv lens -- the offset (window-relative)
            # kv_seqlens would index the nulled/early region. v4_dsa now derives
            # these from kv_lens_cpu (the scheduler-side FULL list) in the eager
            # path; the FULL-graph in-graph path is Stage 1b (run EAGER first
            # to prove correctness: V4_FULL_GRAPH_DECODE=0). swa keeps the offset
            # kv_lens / seq_lens_l (windowed swa_block_table + seqused_kv mask).
            # Verified: marker repro 0/5 (was 1/5 graph+sync / 2/5 eager). See
            # dsv4-swa-eviction-corrupts-kv.
            # BIsect-2 (0903): sw=128 (target config) re-enabled. The lazy-LRU
            # state allocator fix (no free-on-preemption-absence) eliminates
            # the preemption full-realloc spike. The dsa_kv separate pool
            # (marker-validated) prevents swa-eviction cross-request bleed.
            # 0904: chunked-prefill root cause FIXED. The LAST remaining
            # needle-retrieval failure (any chunked prompt, pt>max_prefill_token_num)
            # was NOT eviction -- it was step_context.seq_ids flipping None(chunk1)
            # ->[id](chunk2) across chunks, which flipped _V4StateAlloc's keying
            # scheme mid-request so chunk 2 re-allocated FRESH dsa_kv blocks
            # instead of retaining chunk-1's, orphaning chunk-1's compress_kv.
            # Fix: create_model_inputs_long_context now sets seq_ids=[seq.seq_id]
            # so all chunks are seq_id-keyed (chunk 2 retains chunk-1's blocks).
            # sw=128 re-enabled: eviction contamination (separate dsa_kv pool) AND
            # chunked-prefill (seq_ids consistency) are both structurally fixed.
            # 0908 sw=128 DRIFT BUG FIXED (NULL-PAD). Root cause: WindowBlockManager
            # COMPACTS the swa block_table (drops evicted blocks, renumbers), but
            # the opaque npu_sparse_attn_sharedkv kernel reads ABSOLUTE block
            # indices [0, seqused_kv/bs] and shares ONE seqused_kv to bound BOTH
            # the ori (swa) read and the cmp (compress_kv) read. Compaction forced
            # seqused_kv=OFFSET (else FULL OOBs the compacted table) -> cmp bound
            # = OFFSET/cmp_ratio starved the question after it scrolled out of the
            # SWA window (~128 decode tok) -> drift/ramble into few-shot (GSM8K
            # ~83% vs sw=-1 98%; vllm-ascend sw=128 = 100% -> port-specific).
            # vllm NULL-PADS (full-extent table, evicted->null_block, frees
            # physical blocks). Fix (v4_dsa.py build_v4_decode_meta_in_graph):
            # reconstruct a FULL-EXTENT null-padded swa_block_table in-graph
            # (evicted slots -> 0, masked by ori_win_left) + seqused_kv=FULL
            # (position_id+1). swa POOL still evicts (memory saved); only the
            # cheap int32 table grows. VALIDATED: drift eliminated (idx2 exemplar
            # fixed, all questions clean+EOS), 20q=90% raw / 95% under fair eval;
            # residual 2 misses are SW-AGNOSTIC (idx12=12 vs gold 13 -- sw=-1
            # ALSO picks 12; idx14=60% format vs 60 -- math identical). Matches
            # sw=-1 behavior. See dsv4-sw128-compaction-vs-nullpad-rootcause.
            sliding_window=128,  # 0908 NULL-PAD fix VALIDATED: drift eliminated

        )

        # The 6-tuple DSA KV cache (compress_kv / swa_kv / state / indexer_*)
        # is allocated per-layer by the ascend backend (see
        # backends/dlinfer/ascend/v4_dsa.py) because cache_shapes cannot encode
        # the per-layer heterogeneous block sizes (state block 8/32, dim
        # 1024/2048).  The engine paged k_cache is reused as swa_kv, so the
        # cache block_size MUST equal MLA_BS=128 (--cache-block-size 128).
        config.check_env_func = _check_env_v4
        # swa_kv is the engine's std paged k_cache (reused in v4_dsa.build_*),
        # so the standard k/v cache descs must be allocated; the remaining 5
        # DSA caches come back via get_custom_cache_descs.
        config.use_standard_kv_cache = True
        return config
