# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek V4 model construction + weight loading (Task A).

This module ports the model construction and the checkpoint weight-name
mapping of DeepSeek V4 Flash to lmdeploy. The forward methods intentionally
raise ``NotImplementedError`` because the DSA (Deepseek Sparse Attention)
forward path is Task B.

Architecture summary (from the bf16 checkpoint dump):
  - 43 layers, hidden=4096, n_heads=64, head_dim=512, qk_rope_head_dim=64,
    num_kv_heads=1 (single shared KV), vocab=129280.
  - Attention: wq_a (4096->q_lora 1024) -> q_norm -> wq_b (1024->64*512);
    wkv (4096->512); kv_norm; wo_a (4096->8192) / wo_b (8192->4096) with
    o_groups=8; attn_sink (F32 [n_heads]).
  - Per-layer attention variant decided by ``compress_ratios``:
      * 0  -> SWA (sliding window, no indexer/compressor)
      * 4  -> CSA with Indexer + Compressor
      * 128 -> HCA with Compressor only
  - Indexer (c4 only): wq_b (1024->64*128), weights_proj (4096->64) and a
    nested Compressor(head_dim=128).
  - Compressor (c4/c128): ape (F32), wkv, wgate, norm (RMSNorm).
  - MoE: 256 routed experts (w1/w2/w3, moe_intermediate=2048), 1 shared
    expert, scoring_func=sqrtsoftplus, topk_method=noaux_tc,
    routed_scaling_factor=1.5, swiglu_limit=10.0. First 3 layers use hash
    routing (tid2eid I64 [vocab, 6]) instead of the score-correction bias.
  - mHC (multi-head highway): hc_mult=4, hc_pre/hc_post hooks around attn
    and ffn; hc_head before lm_head. hc_* params are F32.
  - RoPE: YaRN (factor=16, original_max=65536, beta_fast=32/slow=1) with
    complex-exponential interleave partial rotary. compress_rope_theta is
    used for compressed layers (handled in Task B forward).
"""
from collections.abc import Iterable
from typing import Any

import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn

import dlinfer.ops.llm as dllm


from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank, all_reduce
from lmdeploy.pytorch.kernels.dlinfer import DlinferMoECommType
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import (
    ApplyRotaryEmb,
    Attention,
    ParallelEmbedding,
    RMSNorm,
    RopeType,
    build_rotary_embedding,
    build_rotary_params,
)
from lmdeploy.pytorch.nn.linear import (
    build_colwise_linear,
    build_o_proj,
    build_rowwise_linear,
)
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_parameters, get_rope_theta
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_v2 import DeepseekV2MLP
from .utils.cudagraph import CudaGraphMixin

# Fused grouped MoE (dlinfer ``fused_moe_naive``) is used only when the token
# count fed to the EP-sharded experts is at decode scale. ``_grouped_mlp``
# upcasts the gate/up projection to float32 for the swiglu clamp, whose peak
# is ~[active_tokens, 2*ffn] float32 with active_tokens = num_tokens * top_k;
# at prefill scale (e.g. 2k ctx -> 16 ranks x ~2k x 8 top_k active rows) that
# blows the NPU's free memory, while at decode scale (ep_size * batch, a few
# hundred) it is tiny. Above this count we fall back to the per-expert dense
# loop, which computes one expert at a time (small peak) and is OOM-safe. The
# threshold covers decode (incl. graph padding_batch_size: ep16 * max-batch)
# with headroom; real prefills (>= a few dozen tokens * ep16) exceed it.
V4_FUSED_MOE_MAX_TOKENS = int(os.environ.get('V4_FUSED_MOE_MAX_TOKENS', '1024'))

# MoE EP dispatch backend (env-toggled so the verified ``naive`` version is the
# zero-risk default/fallback):
#   naive (default) -- manual EP-spanning all_gather + ``fused_moe_naive`` (each
#       rank computes its local experts on the replicated full token set, then
#       EP all_reduce sums partials). Decode ~110ms/step, graph-capturable,
#       correctness verified. This is the workaround for lmdeploy internal DP
#       leaving an idle DP group (a real alltoall/MC2 dispatch over mismatched
#       per-rank token counts would hang).
#   mc2 (Path B) -- standard dlinfer ``fused_moe_mc2`` (A2's alltoall-equivalent):
#       each rank dispatches ONLY its own tokens to the expert-owning ranks via
#       ``npu_moe_distribute_dispatch_v2`` / ``combine_v2`` (fused NPU ops, no
#       host sync -> graph-capturable). Idle ranks are handled natively by a
#       uniform EP-wide pad + ``x_active_mask`` (real tokens True, padding
#       False), so the EP collective is shape-balanced without replicating
#       tokens. Eliminates the spanning gather + EP all_reduce. Only engaged on
#       decode steps (``moe_comm_type == MC2``); prefill/mixed steps fall through
#       to the naive path (``ALLGATHER``), so the collective is always balanced
#       across ranks. Keeps the vllm-identical DP2xTP8xEP16 topology + graph mode.
V4_MOE_BACKEND = os.environ.get('V4_MOE_BACKEND', 'naive')


class SiluAndMulWithClamp(nn.Module):
    """SwiGLU with gate/up clamping (vllm ``SiluAndMulWithClamp``).

    Computes ``silu(clamp(gate, max=L)) * clamp(up, -L, L)`` where the input
    is split ``[gate, up]`` along the last dim. Used by V4's routed and
    shared experts (``swiglu_limit`` config field); without the clamp the
    MoE output drifts enough to scramble generation.
    """

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


def _apply_dsa_q_rms(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS norm on q after wq_b, before RoPE (vllm-ascend
    ``DeviceOperator.apply_dsa_q_rms`` non-A5 fallback). Normalizes over the
    last (head_dim) axis per (token, head). Without it q's scale differs from
    the reference and the sparse-attn softmax saturates the wrong way,
    producing plausible-magnitude but wrong tokens.
    """
    dtype = q.dtype
    qf = q.float()
    variance = qf.square().mean(-1, keepdim=True)
    return (qf * torch.rsqrt(variance + eps)).to(dtype)


def _get_compress_ratio(config: Any, layer_idx: int) -> int:
    """Return the per-layer compress ratio (0/4/128)."""
    compress_ratios = getattr(config, 'compress_ratios', None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0
    return int(compress_ratios[layer_idx])


# DSA compressor slot-mapping format (mirrors vllm DSA_COMPRESSOR_SLOT_MAPPING_*).
_DSA_SLOT_MAPPING_BLOCK_OFFSET = 2


def _make_hadamard(dim: int) -> torch.Tensor:
    """Build a fixed Sylvester Hadamard matrix of size ``dim`` (power of two).

    Mirrors vllm ``AscendDSAMetadataBuilder.hadamard`` (scipy.linalg.hadamard)
    used by ``rotate_activation`` for the c4 indexer. Returned as bf16.
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, f'dim must be power of two, got {dim}'
    h = torch.tensor([[1]], dtype=torch.float32)
    while h.size(0) < dim:
        n = h.size(0)
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
    return h.contiguous().to(torch.bfloat16)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Hadamard transform over the last dim (mirrors vllm rotate_activation)."""
    hidden_size = x.size(-1)
    return _hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)


def _hadamard_transform_ref(x: torch.Tensor, hadamard: torch.Tensor, scale: float = 1.0):
    """Mirrors vllm hadamard_transform_ref (F.linear with padding)."""
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard) * scale
    return out[..., :dim].reshape(*x_shape)


class Compressor(nn.Module):
    """DeepSeek V4 Compressor.

    Maps a chunk of hidden states into compressed KV/score states. The
    ``head_dim`` differs between the main compressor (``config.head_dim``)
    and the indexer's compressor (``config.index_head_dim``). ``coff`` is
    ``1 + (compress_ratio == 4)`` to account for the overlap buffer used on
    c4 layers.
    """

    def __init__(self,
                 config: Any,
                 compress_ratio: int,
                 head_dim: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap
        self.norm_eps = config.rms_norm_eps

        # ape: (compress_ratio, coff * head_dim), float32
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        # wkv / wgate: (coff * head_dim, hidden), replicated (no TP)
        self.wkv = build_colwise_linear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.wgate = build_colwise_linear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.norm = RMSNorm(self.head_dim, self.norm_eps, dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('DeepseekV4 Compressor forward: Task B (DSA) pending')


class Indexer(nn.Module):
    """DeepSeek V4 Indexer (c4 layers only).

    Produces top-k indices over compressed KV states. Holds wq_b,
    weights_proj and a nested Compressor(head_dim=index_head_dim).
    """

    def __init__(self,
                 config: Any,
                 compress_ratio: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio
        # c4 indexer extras (mirrors vllm Indexer / dsa_v1 _indexer_qkv_prepare)
        self.softmax_scale = self.head_dim**-0.5
        self.rotate = True

        self.wq_b = build_colwise_linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.weights_proj = build_colwise_linear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.compressor = Compressor(
            config,
            compress_ratio=compress_ratio,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )
        # Fixed Hadamard matrix used by rotate_activation (non-trainable buffer).
        self.register_buffer(
            'hadamard',
            _make_hadamard(self.head_dim).to(device=device),
            persistent=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('DeepseekV4 Indexer forward: Task B (DSA) pending')


class DeepseekV4Attention(nn.Module):
    """DeepSeek V4 attention.

    wq_a -> q_norm -> wq_b ; wkv ; kv_norm ; wo_a / wo_b (o_groups) ;
    attn_sink. Optionally an Indexer (c4) and a Compressor (c4/c128).
    The forward (DSA) is Task B.
    """

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        # FULL-graph C1 step2b: the captured attention forward builds the
        # sparse_attn_sharedkv / lightning_indexer metadata ops IN-GRAPH when
        # V4_META_IN_GRAPH=1 (build_v4_dsa_inputs defers them, passing None +
        # the baked max_kv_ceiling). hf_config is needed by the shared backend
        # helpers (_build_sas_metadata / _build_qli_metadata) for tp-aware head
        # counts / index_topk / index_head_dim.
        self._hf_config = config
        # FULL-graph C1 step2b call-once cache: the in-graph metadata op is
        # built ONCE at graph capture (capturing=True) and its output tensor
        # held persistently here so the recorded kernel writes a STABLE address
        # each replay and the attention op reads that same address. Mirrors
        # vllm-ascend dsa_v1 `decode_ratio_to_sas_metadata[layer_name]`.
        # Warmup (capturing=False) builds fresh and must NOT populate these --
        # else capture would skip the op (cache hit) and never record it.
        self._in_graph_sas_map = None
        self._in_graph_qli_map = None
        quantization_config = getattr(config, 'quantization_config', None)
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # attn_sink (F32, [n_heads] in checkpoint); TP-sharded to [heads_per_rank]
        # at load time, so the param holds only this rank's local heads.
        try:
            _attn_ws, _ = get_tp_world_rank('attn')
            _n_local_heads = self.n_heads // _attn_ws if _attn_ws > 1 else self.n_heads
        except Exception:
            _n_local_heads = self.n_heads
        self.n_local_heads = _n_local_heads
        # V4 o-proj shards the GROUPS (not heads) by attn TP: each rank keeps
        # n_groups//attn_tp groups, each with the full n_heads/n_groups heads
        # intact. So attn TP must be <= n_groups (o_groups). Mirrors
        # vllm-ascend deepseek_v4.py: `n_local_groups = n_groups // tp_size`.
        self.n_local_groups = max(1, self.n_groups // _attn_ws) if _attn_ws else self.n_groups
        self.attn_sink = nn.Parameter(
            torch.empty(_n_local_heads, dtype=torch.float32, device=device), requires_grad=False)

        # q low-rank projection
        self.wq_a = build_colwise_linear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps, dtype=dtype, device=device)
        self.wq_b = build_colwise_linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )

        # shared single-KV projection
        self.wkv = build_colwise_linear(
            self.dim,
            self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps, dtype=dtype, device=device)

        # output projection (grouped, o_lora low-rank)
        wo_a_in = (self.n_heads * self.head_dim) // self.n_groups
        wo_a_out = self.n_groups * self.o_lora_rank
        self.wo_a = build_colwise_linear(
            wo_a_in,
            wo_a_out,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )
        self.wo_b = build_o_proj(
            self.n_groups * self.o_lora_rank,
            self.dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention variant
        self.compress_ratio = _get_compress_ratio(config, layer_idx)
        self.indexer: Indexer | None = None
        self.compressor: Compressor | None = None
        if self.compress_ratio > 1:
            self.compressor = Compressor(
                config,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                dtype=dtype,
                device=device,
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(
                    config,
                    compress_ratio=self.compress_ratio,
                    dtype=dtype,
                    device=device,
                )

        # Placeholder attention impl. The real DSA forward (Task B) bypasses
        # this standard paged-attention path; it is kept so the module has a
        # valid Attention instance for weight/layout introspection.
        num_kv_heads = getattr(config, 'num_key_value_heads', 1)
        self.attn_fwd = Attention(
            self.n_heads,
            self.head_dim,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            v_head_size=self.head_dim,
            sliding_window=self.window_size,
        )

    def _forward_o_proj(self, o_proj_input: torch.Tensor) -> torch.Tensor:
        """Output projection: wo_a (batch transpose matmul) -> wo_b.

        Mirrors dsa_v1.py _forward_o_proj default bf16 path
        (non-A5, non-OTP, non-olora, TP=1).
        """
        num_tokens = o_proj_input.shape[0]
        # V4 o-proj: shard GROUPS by attn TP (n_local_groups), keeping every
        # group's full n_heads/n_groups heads intact on each rank. Mirrors
        # vllm-ascend dsa_v1.py _forward_o_proj (bf16 non-A5/OTP/olora path):
        # wo_a.weight is passed directly to npu_transpose_batchmatmul (it is
        # the per-rank ColumnParallel weight [n_groups*o_lora_rank/tp,
        # group_hidden_dim]); the perms handle the batched matmul, so NO manual
        # view/transpose of the (sharded) weight.
        group_hidden_dim = (o_proj_input.shape[1] * o_proj_input.shape[2]
                            // self.n_local_groups)
        o_proj_input = o_proj_input.view(
            num_tokens, self.n_local_groups, group_hidden_dim)
        # wo_a.weight is ColumnParallel [n_local_groups*o_lora_rank,
        # group_hidden_dim] (2D, standard [out,in]). npu_transpose_batchmatmul
        # treats x2 as [B, K, N] (standard bmm — it does NOT internally
        # transpose, and rejects a 3-tuple perm on a 2D tensor), so we must
        # present it as [n_local_groups, group_hidden_dim, o_lora_rank] =
        # [B, K, N]. view group-major (sharding is by groups, so the per-rank
        # weight holds n_local_groups contiguous groups) then transpose the
        # last two axes. perm_x2=(0,1,2) is then identity, matching
        # vllm-ascend dsa_v1.py's bf16 _forward_o_proj call.
        wo_a_w = self.wo_a.weight.view(
            self.n_local_groups, self.o_lora_rank, group_hidden_dim
        ).transpose(1, 2).contiguous()
        o = torch_npu.npu_transpose_batchmatmul(
            o_proj_input,
            wo_a_w,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
        o = o.reshape(num_tokens, -1)
        return self.wo_b(o)

    def _build_sas_meta_in_graph(self, query_start_loc, seq_lens):
        """Build sparse_attn_sharedkv metadata IN the captured forward
        (FULL-graph C1 step2b). Reuses the backend _build_sas_metadata helper
        so the op kwargs stay byte-identical to the eager path -- no sizing /
        semantics divergence (V4 metadata sizing is delicate, see
        dsv4-state-bt-original-token-indexing / dsv4-ascend-v4attention-forward).
        max_seqlen_q=1 (decode constant); max_seqlen_kv = the baked session_len
        ceiling stashed on _DECODE_META by the eager pre-step. in_graph=True so
        the backend feeds a REAL decode cu_seqlens_ori_kv (cat([0, cumsum]),
        vllm A5-style) instead of an empty -- the empty is replay-unsafe in
        dlinfer's graph runner (0-elem storage reclaimed -> 507011/MTE)."""
        from lmdeploy.pytorch.backends.dlinfer.ascend.v4_dsa import \
            _build_sas_metadata, _DECODE_META
        _ceiling = getattr(_DECODE_META, 'max_kv_ceiling', None) or 8192
        return _build_sas_metadata(
            self._hf_config, self.compress_ratio, 1,
            query_start_loc, seq_lens, _ceiling,
            query_start_loc.device, is_decoding=True,
            max_seqlen_q=1, max_seqlen_kv=_ceiling,
            in_graph=True)

    def _build_qli_meta_in_graph(self, query_start_loc, seq_lens):
        """Build lightning_indexer metadata IN the captured forward (C1 step2b),
        c4 layers only. Same shared _build_qli_metadata helper as the eager
        pre-step path; max_seqlen_q=1, max_seqlen_k = the baked ceiling."""
        from lmdeploy.pytorch.backends.dlinfer.ascend.v4_dsa import \
            _build_qli_metadata, _DECODE_META
        _ceiling = getattr(_DECODE_META, 'max_kv_ceiling', None) or 8192
        return _build_qli_metadata(
            self._hf_config, query_start_loc, seq_lens, 1, _ceiling,
            query_start_loc.device)

    def _in_graph_meta(self, kind, query_start_loc, seq_lens):
        """Per-capture-batch-size cache for the in-graph metadata op (C1 step2b).

        Each captured graph (one per cudagraph_capture_batch_size) must record
        its OWN metadata-op kernel -- otherwise only the first (largest) graph
        records the op and smaller graphs read a STALE output from the largest
        graph's capture (aicore MTE OOB at replay, because the stale metadata
        was computed for a different batch shape). Mirrors vllm-ascend dsa_v1
        ``decode_ratio_to_sas_metadata``, but keyed by capture batch size (the
        op output is held persistently per graph so the recorded kernel writes a
        STABLE address each replay and the attention op reads that same address
        -- no stale-pointer 507011).

        - capture (``AscendGraphRunner.capturing == True``): first call FOR
          THIS BATCH SIZE builds + caches into the per-bs map; the op kernel is
          recorded into that batch size's aclgraph. Subsequent captures of a
          DIFFERENT batch size get their own entry (their own op kernel).
        - warmup / eager (``capturing == False``): build fresh each step
          (NOT cached, so capture still sees no entry and records the op).
        - replay: the model's Python forward is NOT re-run
          (AscendSingleGraphRunner.forward calls ``self._graph.replay()``), so
          this method is not re-entered; only the recorded op kernel for the
          replayed batch size re-runs, recomputing the metadata from the
          pinned, per-step-refreshed query_start_loc / seq_lens / sas_seqused_kv
          graph inputs (address-stable).
        """
        from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
            import AscendGraphRunner
        _capturing = getattr(AscendGraphRunner, 'capturing', False)
        if _capturing:
            _bs = max(query_start_loc.numel() - 1, 1)
            _attr = '_in_graph_sas_map' if kind == 'sas' else '_in_graph_qli_map'
            _map = getattr(self, _attr, None)
            if _map is None:
                _map = {}
                setattr(self, _attr, _map)
            if _bs not in _map:
                if kind == 'sas':
                    _map[_bs] = self._build_sas_meta_in_graph(
                        query_start_loc, seq_lens)
                else:
                    _map[_bs] = self._build_qli_meta_in_graph(
                        query_start_loc, seq_lens)
            return _map[_bs]
        if kind == 'sas':
            return self._build_sas_meta_in_graph(query_start_loc, seq_lens)
        return self._build_qli_meta_in_graph(query_start_loc, seq_lens)

    def forward(self,
                hidden_states: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                swa_kv_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                block_table: torch.Tensor,
                seq_lens: torch.Tensor,
                query_start_loc: torch.Tensor,
                sas_metadata,
                dsa_caches=None,
                dsa_meta=None,
                is_decoding: bool = False,
                ) -> torch.Tensor:
        """Attention forward.

        Mirrors dsa_v1.py AscendDSAImpl._forward_prefill (non-multistream,
        bf16) + forward (o_proj rotary + _forward_o_proj).

        - compress_ratio <= 1: SWA (paged swa_kv_cache only).
        - compress_ratio == 128: HCA (compressor + sparse_attn, no indexer).
        - compress_ratio == 4: CSA (compressor + lightning_indexer +
          sparse_attn). ``dsa_caches`` is the 6-tuple paged KV cache and
          ``dsa_meta`` carries the compressor/indexer metadata; both are
          ignored by the SWA branch.
        """
        if self.compress_ratio > 1:
            return self._forward_c4c128(
                hidden_states, cos, sin, swa_kv_cache, slot_mapping,
                block_table, seq_lens, query_start_loc, sas_metadata,
                dsa_caches, dsa_meta, is_decoding=is_decoding)

        # FULL-graph C1 step2b (SWA path): when V4_META_IN_GRAPH=1 + decoding,
        # build_v4_dsa_inputs defers the metadata op (sas_metadata=None
        # sentinel). Build it HERE in the captured forward so the op kernel is
        # recorded at capture and replayed each step (0 host dispatch) reading
        # the pinned, refreshed query_start_loc/seq_lens graph inputs. The
        # c4/c128 path builds it via the same gate inside _forward_c4c128
        # (which receives sas_metadata too).
        if sas_metadata is None and is_decoding and \
                os.environ.get('V4_META_IN_GRAPH', '0') == '1':
            sas_metadata = self._in_graph_meta('sas', query_start_loc, seq_lens)

        # vllm-ascend: prefill op gets cu_seqlens_ori_kv = query cumsum
        # (==kv cumsum, since q_len==kv_len); decode op gets NO
        # cu_seqlens_ori_kv kwarg and the metadata's is empty
        # (dsa_v1.py:466/2386). PA_ND paged decode reads KV via block_table
        # + seqused_kv. Passing query_start_loc ([0,1]) at decode tells the
        # op each req has 1 KV token, corrupting the read -> token-2 drift.
        cu_seqlens_ori_kv = (query_start_loc if not is_decoding
                             else torch.empty(0, dtype=torch.int32,
                                              device=hidden_states.device))

        # ---- MLA prolog (dsa_v1.py _forward_prefill, bf16) ----
        # q: wq_a -> q_norm -> wq_b
        q_a = self.wq_a(hidden_states)          # [tok, q_lora]
        qr = self.q_norm(q_a)                   # [tok, q_lora]
        q = self.wq_b(qr)                       # [tok, n_local_heads * head_dim]
        q = q.view(-1, self.n_local_heads, self.head_dim)  # [tok, n_local_heads, head_dim]
        # weightless q RMS norm (vllm apply_dsa_q_rms, non-A5 fallback)
        q = _apply_dsa_q_rms(q, self.eps)

        # partial rotary on q (inplace, interleave)
        dllm.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # kv: wkv -> kv_norm
        kv = self.wkv(hidden_states)            # [tok, head_dim]
        kv = self.kv_norm(kv)                   # [tok, head_dim]
        kv = kv.view(-1, 1, self.head_dim)      # [tok, 1, nope+rope]

        # partial rotary on kv
        dllm.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # scatter kv into swa_kv_cache (dsa_kv_compress_scatter, non-A5).
        # Use _paged_scatter (index_copy_) instead of the raw
        # npu_scatter_nd_update_v2 op: on A3 the raw aclnn op lacks tiling and
        # either silently no-ops (leaving the cache zero) or stack-smashes for
        # larger seq_len. The c4/c128 path already uses _paged_scatter for the
        # same reason; SWA must too.
        self._paged_scatter(swa_kv_cache, slot_mapping, kv)

        # ---- sparse attention (sparse_attn_sharedkv, SWA) ----
        attn_output = dllm.sparse_attn_sharedkv(
            q,
            ori_kv=swa_kv_cache,
            ori_block_table=block_table,
            cu_seqlens_q=query_start_loc,
            seqused_kv=seq_lens,
            sinks=self.attn_sink,
            metadata=sas_metadata,
            softmax_scale=self.scale,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q='TND',
            layout_kv='PA_ND',
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        )[0]

        # CHUNK-2+ SWA output probe: SWA layers run BEFORE c4/c128. If the swa
        # window read is wrong on a continuation chunk (q=24 + history), the
        # hidden states fed to c4/c128 are garbage -> retrieval fails. Dump
        # finiteness + the op's seqused_kv/cu_seqlens_ori_kv to detect it.
        # CHUNK-2+ SWA output probe: after the 0904 seqused_kv=offset fix
        # (v4_dsa cr<=1 dict), confirm the SWA op no longer OOBs on chunk-2
        # prefill (windowed swa_block_table). Fires on continuation chunks
        # (small q <= 50 but kv > 50 = a chunk-2+ with history).
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and query_start_loc.numel() >= 2
                and int(query_start_loc[-1].item()) <= 50
                and int(seq_lens[0].item()) > 50):
            _sr = (dist.get_rank() if (dist.is_available()
                   and dist.is_initialized()) else 0)
            if _sr == 0:
                _sfin = bool(attn_output.isfinite().all().item())
                _samx = float(attn_output.abs().max().item())
                print(f'[V4-SWA-C2] rank={_sr} L{self.layer_idx} '
                      f'seqused_kv={seq_lens.tolist()} '
                      f'bt_shape={tuple(block_table.shape)} '
                      f'cu_ori_kv={cu_seqlens_ori_kv.tolist() if cu_seqlens_ori_kv.numel()<=8 else tuple(cu_seqlens_ori_kv.shape)} '
                      f'win={self.window_size} '
                      f'attn_out fin={_sfin} absmax={_samx:.4e}',
                      flush=True)

        # ---- o_proj rotary (dsa_v1.py forward, after attn) ----
        dllm.inplace_partial_rotary_mul(
            attn_output.unsqueeze(1), cos, -sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # ---- o_proj (wo_a batch transpose matmul -> wo_b) ----
        return self._forward_o_proj(attn_output)

    # ---- c4/c128 forward (dsa_v1.py AscendDSAImpl._forward_prefill) ----

    def _mla_prolog(self, hidden_states, cos, sin):
        """MLA prolog shared by c4/c128 (dsa_v1.py _forward_prefill, bf16).

        q: wq_a -> q_norm -> wq_b ; kv: wkv -> kv_norm ; partial rotary.
        Returns (q[tok,n_heads,head_dim], qr[tok,q_lora], kv[tok,1,head_dim]).
        """
        _pen = os.environ.get('V4_DEBUG_MOE', '0') == '1'
        if _pen:
            from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
                import AscendGraphRunner
            _pen = not getattr(AscendGraphRunner, 'capturing', False)
            _rk = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            def _ps(tag, t):
                if not _pen:
                    return
                torch.npu.synchronize()
                print(f'[V4-PROLOG] rank={_rk} L{self.layer_idx} {tag} '
                      f'sh={tuple(t.shape) if t is not None else "-"}', flush=True)
        else:
            def _ps(tag, t=None):
                pass
        q_a = self.wq_a(hidden_states)              # [tok, q_lora]
        _ps('post-wq_a', q_a)
        qr = self.q_norm(q_a)                        # [tok, q_lora]
        _ps('post-q_norm', qr)
        # wq_b is ColumnParallel (is_tp=True) -> per-rank output is
        # n_local_heads*head_dim (n_heads//attn_tp heads), so view with
        # n_local_heads, not the full n_heads. (kv is replicated: wkv
        # is_tp=False, so its view(-1,1,head_dim) uses the full head_dim.)
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        _ps('post-wq_b', q)
        # weightless q RMS norm (vllm apply_dsa_q_rms, non-A5 fallback)
        q = _apply_dsa_q_rms(q, self.eps)
        _ps('post-q_rms', q)
        dllm.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim])
        _ps('post-q_rotary', q)
        kv = self.kv_norm(self.wkv(hidden_states)).view(-1, 1, self.head_dim)
        _ps('post-wkv', kv)
        dllm.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim])
        _ps('post-kv_rotary', kv)
        return q, qr, kv

    @staticmethod
    def _compressor_call(hidden, comp, state_cache, cos, sin, block_table,
                         cu_seqlens, start_pos, cmp_ratio, rope_head_dim):
        """Wrap the compressor op call (mirrors dsa_v1.py inline compressor)."""
        if os.environ.get('V4_DEBUG_MOE', '0') == '1':
            # Dump the LONG-prefill compressor call (num_cmp > 100) for vllm
            # cross-check. The first-call dump missed the 1336 prefill. One-shot
            # per rank via _comp_dumped_long; fsync so it survives the crash.
            _long = (cu_seqlens.numel() >= 2
                     and int(cu_seqlens[-1].item()) > 1000)
            if _long and not getattr(DeepseekV4Attention,
                                     '_comp_dumped_long', False):
                import torch.distributed as _d
                _rk = _d.get_rank() if (_d.is_available() and _d.is_initialized()) else 0
                if _rk == 0:
                    def _cdlog(msg):
                        print(msg, flush=True)
                        try:
                            with open('/tmp/v4_comp_dump.log', 'a') as _f:
                                _f.write(msg + '\n'); _f.flush()
                                os.fsync(_f.fileno())
                        except Exception:
                            pass
                    def _d_(nm, t):
                        if t is None:
                            return f'{nm}=None'
                        s = t.stride() if hasattr(t, 'stride') else '-'
                        return (f'{nm} shape={tuple(t.shape)} dtype={t.dtype} '
                                f'stride={tuple(s) if s != "-" else "-"} '
                                f'fin={bool(t.isfinite().all().item()) if t.numel() and t.dtype.is_floating_point else "-"} '
                                f'absmax={float(t.abs().max().item()) if t.numel() and t.dtype.is_floating_point else "-"}')
                    sc = state_cache.squeeze(-2)
                    _cdlog('[V4-COMP-DUMP] compressor args (long prefill):')
                    _cdlog(_d_('hidden', hidden))
                    _cdlog(_d_('wkv', comp.wkv.weight))
                    _cdlog(_d_('wgate', comp.wgate.weight))
                    _cdlog(_d_('state_cache', sc))
                    _cdlog(_d_('ape', comp.ape))
                    _cdlog(_d_('norm', comp.norm.weight))
                    _sin = sin.view(-1, sin.shape[-1])
                    _cos = cos.view(-1, cos.shape[-1])
                    _cdlog(_d_('sin', _sin))
                    _cdlog(_d_('cos', _cos))
                    _cdlog(_d_('state_bt', block_table))
                    _cdlog(_d_('cu_seqlens', cu_seqlens))
                    _cdlog(_d_('start_pos', start_pos))
                    _cdlog(f'  ints: rope_head_dim={rope_head_dim} cmp_ratio={cmp_ratio} '
                           f'coff={comp.coff} norm_eps={comp.norm_eps} '
                           f'rotary_mode=2 cache_mode=1')
                    if block_table.numel():
                        _cdlog(f'  state_bt[0,:16]={block_table[0,:16].tolist()} '
                               f'bt_max={int(block_table.max().item())}')
                    if cu_seqlens.numel() <= 8:
                        _cdlog(f'  cu_seqlens.tolist()={cu_seqlens.tolist()}')
                    if start_pos.numel() <= 8:
                        _cdlog(f'  start_pos.tolist()={start_pos.tolist()}')
                    if comp.ape.numel() <= 64:
                        _cdlog(f'  ape[0,:8]={comp.ape.flatten()[:8].tolist()}')
                    if _cos.numel() <= 1024:
                        _cdlog(f'  cos[0,:8]={_cos[0,:8].tolist()}')
                DeepseekV4Attention._comp_dumped_long = True
        return dllm.compressor(
            hidden, comp.wkv.weight, comp.wgate.weight,
            state_cache.squeeze(-2), comp.ape, comp.norm.weight,
            sin.view(-1, sin.shape[-1]),
            cos.view(-1, cos.shape[-1]),
            state_block_table=block_table, cu_seqlens=cu_seqlens,
            seqused=None, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio,
            coff=comp.coff, norm_eps=comp.norm_eps,
            rotary_mode=2, cache_mode=1)

    @staticmethod
    def _paged_scatter(cache, slot_mapping, update):
        """Scatter update rows into a paged [num_blocks, block_size, 1, dim]
        cache. ``slot_mapping`` is either flat 1D [N] or block-offset 2D
        [N,2] (= [block_idx, offset]). Uses index_copy_ on a flat view because
        aclnnScatterNdUpdateV2 lacks tiling on A3. int8 caches (indexer_k) are
        scattered via a float16 staging copy since index_copy_ silently no-ops
        on int8. Mirrors vllm dsa_kv_compress_scatter."""
        block_size = cache.shape[1]
        if slot_mapping.dim() == 2:
            flat = slot_mapping[:, 0].long() * block_size + slot_mapping[:, 1].long()
        else:
            flat = slot_mapping.long()
        if update.dim() == 2:
            update = update.unsqueeze(1)
        dim = update.shape[-1]
        # compressor_metadata emits slot_mapping == -1 for PADDING rows
        # (partial chunks that produce no complete compressed token, e.g. a
        # 5-token c128/c4 prefill). vllm's npu_scatter_nd_update_v2 bounds-
        # checks and skips -1 natively, but aclnnScatterNdUpdateV2 is not in
        # this libopapi so we cannot use it. torch index_copy_ has NO bounds
        # check -- on NPU a -1 index WRAPS to the last cache slot, writing the
        # padding row's uninitialized (at::empty) compressor output into a live
        # KV slot -> wrong, run-to-run-nondeterministic greedy output.
        #
        # Graph-safe filter. The previous `if not valid.all(): flat = flat[
        # valid]` branched on a host sync (.all()->__bool__->.item()), which
        # breaks NPU graph capture ("Not allow to synchronize captured-stream",
        # error 107027) -- the sync fires UNCONDITIONALLY, even when there is no
        # padding, so decode capture could never succeed. Instead: redirect -1
        # rows to the LAST cache slot and ZERO their update, so the redirected
        # write is a no-op into a slot that holds no live KV during decode (the
        # sequence occupies far fewer slots than the cache holds). Real rows
        # have unique slots (one token per slot) so they never collide; the only
        # collision is padding->last_slot, which writes zero. When there is no
        # padding the mask is all-true and this is an exact no-op, so decode
        # capture at max_tokens=1 stays exact.
        valid = flat >= 0
        scratch = torch.full((), cache.shape[0] * cache.shape[1] - 1,
                             dtype=flat.dtype, device=flat.device)
        flat = torch.where(valid, flat, scratch)
        # update is [N,1,dim] (forced 3D above); broadcast the [N] mask across
        # the trailing (1,dim) axes -> [N,1,1]. (valid.unsqueeze(-1) would be
        # [N,1] and broadcast [N,1,dim]*[N,1] -> [N,N,dim] -- a shape bug.)
        update = update * valid.view(-1, 1, 1).to(update.dtype)
        if cache.dtype == torch.int8:
            staging = cache.to(torch.float16)
            staging.view(-1, 1, dim).index_copy_(0, flat, update.to(torch.float16))
            cache.copy_(staging.to(torch.int8))
        else:
            cache.view(-1, 1, dim).index_copy_(0, flat, update)

    def _forward_c4c128(self, hidden_states, cos, sin, swa_kv_cache,
                        slot_mapping, block_table, seq_lens,
                        query_start_loc, sas_metadata,
                        dsa_caches, dsa_meta, is_decoding=False):
        """c128 (HCA) and c4 (CSA) attention forward.

        Mirrors dsa_v1.py AscendDSAImpl._forward_prefill (compress_ratio>1,
        non-multistream bf16 path). ``dsa_caches`` is the 6-tuple:
        (compress_kv, swa_kv, state, indexer_state, indexer_k, indexer_scale).
        """
        # 0904 chunk2 crash pinpoint: [V4-C2-LN] (input_layernorm) passed but
        # [V4-C2-ENT] (.tolist D2H) raised -> the .tolist catches a fault the
        # default-stream sync missed. This torch.npu.synchronize() at the VERY
        # entry of _forward_c4c128 decides: if it RAISES -> the fault is
        # delayed from input_layernorm/hc_pre (cross-stream); if it passes ->
        # the fault is a later NPU op in _forward_c4c128 (MLA prolog / swa
        # scatter / compressor). fsync -> LAST tag = fault point.
        if (not is_decoding and self.layer_idx < 3
                and query_start_loc.numel() >= 2
                and int(query_start_loc[-1].item()) < 50
                and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'):
            _vr = (dist.get_rank() if (dist.is_available()
                    and dist.is_initialized()) else 0)
            if _vr == 0:
                torch.npu.synchronize()
                _ve = (f'[V4-C2-VE] L{self.layer_idx} cr={self.compress_ratio} '
                       f'_FORWARD_C4C128 ENTRY SYNC OK '
                       f'qsl[-1]={int(query_start_loc[-1].item())}')
                print(_ve, flush=True)
                try:
                    with open('/tmp/v4_dbg.log', 'a') as _f:
                        _f.write(_ve + '\n'); _f.flush(); os.fsync(_f.fileno())
                except Exception:
                    pass
        cr = self.compress_ratio
        (compress_kv_cache, _swa, state_cache,
         indexer_state_cache, indexer_k_cache,
         indexer_scale_cache) = dsa_caches
        m = dsa_meta

        # CHUNK-2+ ENTRY PROBE (ungated except rank0 + prefill): confirms the
        # function IS reached on chunk-2 prefill + the real start_pos value
        # (was the start_pos>1000 gate False because start_pos is 0/None?).
        if not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1':
            _er = (dist.get_rank() if (dist.is_available()
                   and dist.is_initialized()) else 0)
            if _er == 0 and query_start_loc.numel() >= 2:
                _esp = (m.start_pos.tolist() if m is not None
                        and getattr(m, 'start_pos', None) is not None
                        and m.start_pos.numel() else None)
                _esl = seq_lens.tolist() if seq_lens is not None else None
                print(f'[V4-C2-ENT] L{self.layer_idx} cr={cr} '
                      f'start_pos={_esp} seqused_kv={_esl} '
                      f'qsl[-1]={int(query_start_loc[-1].item())}', flush=True)
        # CHUNK-2 RETENTION PROBE: read compress_kv_cache at the physical
        # block the dsa_kv_bt maps compressed-block-0 to, BEFORE chunk-2's
        # compressor scatter runs (or, for chunk 1, before its own scatter).
        # Block 0 holds chunk-1's compressed KV (tokens 0..127 incl. the 4271
        # needle). On chunk 2 (start_pos>0) it must be NON-zero (chunk 1 wrote
        # it). Fires for chunk 1 (qsl>1000) AND chunk 2 (start_pos>1000).
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None
                and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and (int(m.start_pos[0].item()) > 1000
                     or (query_start_loc.numel() >= 2
                         and int(query_start_loc[-1].item()) > 1000))):
            _rr = (dist.get_rank() if (dist.is_available()
                   and dist.is_initialized()) else 0)
            if _rr == 0:
                _AGR = None
                try:
                    from dlinfer.framework.lmdeploy_ext.cudagraph. \
                        ascend_cudagraph import AscendGraphRunner as _AGR
                except Exception:
                    pass
                if not getattr(_AGR, 'capturing', False):
                    torch.npu.synchronize()
                    _bt = m.compress_kv_block_table
                    if _bt is not None and _bt.numel():
                        _b0 = int(_bt[0, 0].item())
                        _b1 = int(_bt[0, 1].item()) \
                            if _bt.shape[1] > 1 else -1
                        _b2 = int(_bt[0, 2].item()) \
                            if _bt.shape[1] > 2 else -1
                        # chunk-2 first compressed block col:
                        _sp = int(m.start_pos[0].item())
                        _c2col = (_sp // cr) // 128
                        _c2b = int(_bt[0, _c2col].item()) \
                            if _c2col < _bt.shape[1] else -1
                        _ck0 = compress_kv_cache[_b0]
                        _ck0fin = bool(_ck0.isfinite().all().item()) \
                            if _ck0.numel() else True
                        _ck0mx = float(_ck0.abs().max().item()) \
                            if _ck0.numel() else 0.0
                        print(f'[V4-RET] L{self.layer_idx} cr={cr} '
                              f'sp={_sp} qsl={int(query_start_loc[-1].item())} '
                              f'bt[0,0:3]={_b0},{_b1},{_b2} '
                              f'c2col={_c2col} c2block={_c2b} '
                              f'block0 fin={_ck0fin} absmax={_ck0mx:.4e} '
                              f'shape={tuple(compress_kv_cache.shape)}',
                              flush=True)
        # FULL-graph C1 step2b (c4/c128 path): when V4_META_IN_GRAPH=1 +
        # decoding, build_v4_dsa_inputs deferred both metadata ops (None
        # sentinels on m.sas_metadata / m.qli_metadata). Build them HERE in the
        # captured forward so their kernels are recorded at capture and
        # replayed each step (0 host dispatch), reading the pinned, refreshed
        # query_start_loc/seq_lens. Built BEFORE any op consumes them; m.qli
        # is consumed by the lightning-indexer op, m.sas by sparse_attn.
        if is_decoding and os.environ.get('V4_META_IN_GRAPH', '0') == '1':
            if m is not None and m.sas_metadata is None:
                m.sas_metadata = self._in_graph_meta(
                    'sas', query_start_loc, seq_lens)
            if cr == 4 and m is not None and getattr(m, 'qli_metadata', None) is None:
                m.qli_metadata = self._in_graph_meta(
                    'qli', query_start_loc, seq_lens)

        # sub-op diagnostic: sync+print after each c4/c128 op (eager only;
        # host sync during npu.graph capture -> 107027). Pinpoints which op
        # in the c4 path faults on long single-request prefill.
        _dbg_en = os.environ.get('V4_DEBUG_MOE', '0') == '1'
        if _dbg_en:
            from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
                import AscendGraphRunner
            _cap = getattr(AscendGraphRunner, 'capturing', False)
            _dbg_en = not _cap
            _rk = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

            # crash-surviving logger: writes a per-line fsync'd append so the
            # probe values are not lost when the worker dies on a 5070xx
            # aicore fault. Prefill only (decode would flood fsync).
            def _flog(msg):
                try:
                    with open('/tmp/v4_dbg.log', 'a') as _f:
                        _f.write(msg + '\n')
                        _f.flush()
                        os.fsync(_f.fileno())
                except Exception:
                    pass

            def _dbg(tag, t=None):
                if not _dbg_en:
                    return
                torch.npu.synchronize()
                shp = (tuple(t.shape) if t is not None else '-')
                _m = (f'[V4-C4] rank={_rk} L{self.layer_idx} cr={cr} '
                      f'{tag} hs={shp}')
                print(_m, flush=True)
                if not is_decoding:
                    _flog(_m)
        else:
            def _dbg(tag, t=None):
                pass

        # cu_seqlens_ori_kv: prefill -> query cumsum (==kv cumsum); decode ->
        # empty (vllm-ascend dsa_v1.py:466/2386 passes empty + no op kwarg).
        cu_seqlens_ori_kv = (query_start_loc if not is_decoding
                             else torch.empty(0, dtype=torch.int32,
                                              device=hidden_states.device))

        # ---- MLA prolog (dsa_v1.py _forward_prefill, bf16) ----
        # enter-c4 sync probe: if this sync (before any L2 op) raises, the
        # fault is async from L0/L1 in-flight ops (SWA scatter / MoE); if it
        # passes and post-mla-prolog raises, the fault is in the MLA prolog.
        _dbg('enter-c4', hidden_states)
        if _dbg_en and not is_decoding:
            torch.npu.synchronize()
            _hfin = bool(hidden_states.isfinite().all().item())
            _hmx = float(hidden_states.abs().max().item())
            print(f'[V4-ENTER] rank={_rk} L{self.layer_idx} cr={cr} '
                  f'hs={tuple(hidden_states.shape)} fin={_hfin} '
                  f'absmax={_hmx:.4e} ptr={hidden_states.data_ptr()}',
                  flush=True)
        q, qr, kv = self._mla_prolog(hidden_states, cos, sin)
        _dbg('post-mla-prolog', q)

        # scatter kv into swa_kv_cache (ori_kv, dsa_kv_compress_scatter)
        if _dbg_en and not is_decoding:
            torch.npu.synchronize()
            _sm = slot_mapping
            _sm_max = int(_sm.max().item()) if _sm.numel() else -1
            _sm_min = int(_sm.min().item()) if _sm.numel() else -1
            _cap = swa_kv_cache.shape[0] * swa_kv_cache.shape[1]
            print(f'[V4-SLOT] rank={_rk} L{self.layer_idx} cr={cr} '
                  f'slot_mapping shape={tuple(_sm.shape)} '
                  f'smin={_sm_min} smax={_sm_max} '
                  f'cap={_cap} OOB={_sm_max >= _cap} '
                  f'kv_shape={tuple(swa_kv_cache.shape)}',
                  flush=True)
        self._paged_scatter(swa_kv_cache, slot_mapping, kv)
        _dbg('post-swa-scatter', kv)
        # 0904 chunk2-only crash pinpoints: capture-safe (not capturing),
        # warmup-safe (start_pos>0 excludes warmup prefill), chunk1-safe
        # (start_pos>0 excludes chunk1 sp=0). fsync -> survives the 507057
        # aicore fault so the LAST tag printed = the op that faulted.
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None
                and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and int(m.start_pos[0].item()) > 0):
            _c2rk = (dist.get_rank() if (dist.is_available()
                      and dist.is_initialized()) else 0)
            if _c2rk == 0:
                _AGR2 = None
                try:
                    from dlinfer.framework.lmdeploy_ext.cudagraph. \
                        ascend_cudagraph import AscendGraphRunner as _AGR2
                except Exception:
                    pass
                if not getattr(_AGR2, 'capturing', False):
                    torch.npu.synchronize()
                    _tag = (f'[V4-C2] L{self.layer_idx} cr={cr} '
                            f'sp={int(m.start_pos[0].item())} '
                            f'qsl={int(query_start_loc[-1].item())} '
                            f'POST-SWA-SCATTER OK')
                    print(_tag, flush=True)
                    try:
                        with open('/tmp/v4_dbg.log', 'a') as _f:
                            _f.write(_tag + '\n'); _f.flush(); os.fsync(_f.fileno())
                    except Exception:
                        pass

        # ---- compressor (dsa_v1.py compressor_metadata + compressor) ----
        # compressor_metadata indexes the compress_kv cache (block_size =
        # MLA), producing the slot_mapping used to scatter compressed_kv.
        compress_cos, compress_sin, compress_slot_mapping = \
            dllm.compressor_metadata(
                m.full_compress_cos, m.full_compress_sin,
                query_start_loc, m.start_pos, m.compress_kv_block_table,
                m.compress_kv_block_size,
                _DSA_SLOT_MAPPING_BLOCK_OFFSET, cr,
                m.num_compressed_tokens, m.num_reqs_actual)
        _dbg('post-compressor-metadata', compress_slot_mapping)
        # 0904 chunk2-only crash pinpoint: did compressor_metadata return valid
        # slot_mapping? If this prints but POST-SWA-SCATTER also printed, the
        # fault is IN compressor_metadata (or _compressor_call/scatter next).
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None
                and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and int(m.start_pos[0].item()) > 0):
            _c2rk2 = (dist.get_rank() if (dist.is_available()
                       and dist.is_initialized()) else 0)
            if _c2rk2 == 0:
                _AGR3 = None
                try:
                    from dlinfer.framework.lmdeploy_ext.cudagraph. \
                        ascend_cudagraph import AscendGraphRunner as _AGR3
                except Exception:
                    pass
                if not getattr(_AGR3, 'capturing', False):
                    torch.npu.synchronize()
                    _csmx = int(compress_slot_mapping[:, 0].max().item()) \
                        if compress_slot_mapping.numel() else -1
                    _csmn = int(compress_slot_mapping[:, 0].min().item()) \
                        if compress_slot_mapping.numel() else -1
                    _cpool = compress_kv_cache.shape[0]
                    _tag2 = (f'[V4-C2] L{self.layer_idx} cr={cr} '
                             f'sp={int(m.start_pos[0].item())} '
                             f'qsl={int(query_start_loc[-1].item())} '
                             f'POST-COMPRESSOR-META OK '
                             f'csm=[{_csmn},{_csmx}] cpool={_cpool} '
                             f'OOB={_csmx >= _cpool}')
                    print(_tag2, flush=True)
                    try:
                        with open('/tmp/v4_dbg.log', 'a') as _f:
                            _f.write(_tag2 + '\n'); _f.flush(); os.fsync(_f.fileno())
                    except Exception:
                        pass
        # pre-compressor value check: is the fault a computation bug (Inf/NaN
        # in hidden or a corrupted state_cache) or a platform/op tolerance
        # issue (large-but-finite hidden)? L3 c128 (num_cmp=2) succeeds but L9
        # c128 (num_cmp=2, same shapes) faults -> value-dependent, so inspect
        # the input magnitudes right before the op. Eager only (host sync).
        if _dbg_en:
            torch.npu.synchronize()
            _hfin = bool(hidden_states.isfinite().all().item())
            _hmax = float(hidden_states.abs().max().item())
            _sfin = bool(state_cache.isfinite().all().item())
            _smax = float(state_cache.abs().max().item())
            _sbt = m.state_block_table
            _sbt_max = int(_sbt.max().item()) if _sbt.numel() else -1
            _pool = state_cache.shape[0] if state_cache.dim() >= 1 else 0
            _m = (f'[V4-PRECOMP] rank={_rk} L{self.layer_idx} cr={cr} '
                  f'h_fin={_hfin} h_absmax={_hmax:.4e} '
                  f's_fin={_sfin} s_absmax={_smax:.4e} '
                  f'num_cmp={compress_slot_mapping.shape[0]} '
                  f'state_bt={tuple(m.state_block_table.shape)} '
                  f'sbt_max={_sbt_max} pool={_pool} '
                  f'ccos={tuple(compress_cos.shape)} '
                  f'csin={tuple(compress_sin.shape)} '
                  f'csl={tuple(compress_slot_mapping.shape)} '
                  f'csl_max={int(compress_slot_mapping[:, 0].max().item()) if compress_slot_mapping.numel() else -1} '
                  f'cu={query_start_loc.tolist() if query_start_loc.numel()<=8 else query_start_loc.shape} '
                  f'sp={m.start_pos.tolist() if m.start_pos.numel()<=8 else tuple(m.start_pos.shape)}')
            print(_m, flush=True)
            if not is_decoding:
                _flog(_m)
        # compressor writes the state_cache in place (state_block_table indexes
        # the state cache whose block_size is baked into its shape).
        compressed_kv = self._compressor_call(
            hidden_states, self.compressor, state_cache,
            compress_cos, compress_sin, m.state_block_table,
            query_start_loc, m.start_pos, cr, self.rope_head_dim)
        if _dbg_en and not is_decoding and query_start_loc.numel() >= 2 \
                and int(query_start_loc[-1].item()) > 1000:
            torch.npu.synchronize()
            _ckvfin = bool(compressed_kv.isfinite().all().item()) if compressed_kv.numel() else True
            _ckvmx = float(compressed_kv.abs().max().item()) if compressed_kv.numel() else 0.0
            _ckvm = (f'[V4-CKV] rank={_rk} L{self.layer_idx} cr={cr} '
                     f'ckv shape={tuple(compressed_kv.shape)} '
                     f'stride={tuple(compressed_kv.stride())} '
                     f'fin={_ckvfin} absmax={_ckvmx:.4e} '
                     f'ptr={compressed_kv.data_ptr()}')
            print(_ckvm, flush=True)
            _flog(_ckvm)
        _dbg('post-compressor', compressed_kv)
        if compressed_kv.shape[0] > 0:
            self._paged_scatter(compress_kv_cache, compress_slot_mapping,
                                compressed_kv)
            _dbg('post-compress-scatter', compressed_kv)
        # POST-SCATTER PROBE: where did the compressor WRITE, and is block 0
        # (dsa_kv_bt[0,0], chunk-1's 4271 block) now non-zero? slotmap_blk is
        # the [min,max] block_idx column of compress_slot_mapping (the physical
        # compress_kv blocks the scatter touched THIS step). On chunk 1 we
        # expect block0_after_scatter to become non-zero; on chunk 2 we expect
        # it to STAY non-zero (chunk 1 wrote it). If chunk 2 block0 == 0,
        # chunk 1 never wrote to the phys block chunk 2 reads.
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None
                and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and (int(m.start_pos[0].item()) > 1000
                     or (query_start_loc.numel() >= 2
                         and int(query_start_loc[-1].item()) > 1000))):
            _scr = (dist.get_rank() if (dist.is_available()
                    and dist.is_initialized()) else 0)
            if _scr == 0:
                _AGR = None
                try:
                    from dlinfer.framework.lmdeploy_ext.cudagraph. \
                        ascend_cudagraph import AscendGraphRunner as _AGR
                except Exception:
                    pass
                if not getattr(_AGR, 'capturing', False):
                    torch.npu.synchronize()
                    _bt = m.compress_kv_block_table
                    _b0 = int(_bt[0, 0].item())
                    _ck0 = compress_kv_cache[_b0]
                    _ck0mx = float(_ck0.abs().max().item()) \
                        if _ck0.numel() else 0.0
                    if compress_slot_mapping.numel():
                        _sm_blk = compress_slot_mapping[:, 0]
                        _sm_min = int(_sm_blk.min().item())
                        _sm_max = int(_sm_blk.max().item())
                    else:
                        _sm_min = _sm_max = -1
                    print(f'[V4-SCAT] L{self.layer_idx} cr={cr} '
                          f'qsl={int(query_start_loc[-1].item())} '
                          f'sp={int(m.start_pos[0].item())} '
                          f'slotmap_blk=[{_sm_min},{_sm_max}] '
                          f'bt[0,0]={_b0} '
                          f'block0_after_scatter absmax={_ck0mx:.4e} '
                          f'ckv_rows={compressed_kv.shape[0]}',
                          flush=True)

        # ---- c4 indexer (dsa_v1.py indexer_select_qli) ----
        compress_topk_idxs = None
        if cr == 4:
            idx = self.indexer
            # indexer q: wq_b(qr) -> rotary -> hadamard (dsa_v1 _indexer_qkv_prepare)
            iq = idx.wq_b(qr).view(-1, idx.n_heads, idx.head_dim)
            dllm.inplace_partial_rotary_mul(
                iq.unsqueeze(1), cos, sin,
                rotary_mode='interleave',
                partial_slice=[idx.head_dim - self.rope_head_dim, idx.head_dim])
            iq = rotate_activation(iq, idx.hadamard)
            # indexer compressor -> compressed indexer kv (writes indexer_state_cache)
            ic_cos, ic_sin, ic_slot_mapping = dllm.compressor_metadata(
                m.full_compress_cos, m.full_compress_sin,
                query_start_loc, m.start_pos, m.indexer_k_block_table,
                m.indexer_k_block_size,
                _DSA_SLOT_MAPPING_BLOCK_OFFSET, cr,
                m.num_compressed_tokens, m.num_reqs_actual)
            # pre-indexer-compressor discriminating sync: 829 (post-compress-
            # scatter) passed, so a fault surfaced here isolates it to
            # indexer_metadata (843) or the indexer compressor (849). If THIS
            # sync raises -> metadata faulted; if it passes and
            # post-indexer-compressor raises -> the indexer compressor faults.
            if _dbg_en:
                torch.npu.synchronize()
                _isbt = m.indexer_state_block_table
                _isbt_max = int(_isbt.max().item()) if _isbt.numel() else -1
                _isbt_min = int(_isbt.min().item()) if _isbt.numel() else -1
                _ipool = indexer_state_cache.shape[0]
                _irows = indexer_state_cache.shape[1] if indexer_state_cache.dim() >= 2 else 1
                _idim = indexer_state_cache.shape[-1]
                _ifin = bool(indexer_state_cache.isfinite().all().item())
                _ismx = float(indexer_state_cache.abs().max().item())
                _icsl_max = int(ic_slot_mapping[:, 0].max().item()) if ic_slot_mapping.numel() else -1
                _icsl_min = int(ic_slot_mapping[:, 0].min().item()) if ic_slot_mapping.numel() else -1
                _iccos_max = float(ic_cos.abs().max().item()) if ic_cos.numel() else 0.0
                _hfin = bool(hidden_states.isfinite().all().item())
                _hmx = float(hidden_states.abs().max().item())
                _m = (f'[V4-PRE-IDX] rank={_rk} L{self.layer_idx} cr={cr} '
                      f'isbt={tuple(_isbt.shape)} isbt=[{_isbt_min},{_isbt_max}] '
                      f'ipool={_ipool} irows={_irows} idim={_idim} '
                      f'i_fin={_ifin} i_absmax={_ismx:.4e} '
                      f'icsl=[{_icsl_min},{_icsl_max}] '
                      f'iccos_absmax={_iccos_max:.4e} '
                      f'icc={tuple(ic_cos.shape)} ics={tuple(ic_sin.shape)} '
                      f'csl={tuple(ic_slot_mapping.shape)} '
                      f'h_fin={_hfin} h_absmax={_hmx:.4e} '
                      f'num_cmp={ic_slot_mapping.shape[0]} '
                      f'cu={query_start_loc.tolist() if query_start_loc.numel()<=8 else query_start_loc.shape} '
                      f'sp={m.start_pos.tolist() if m.start_pos.numel()<=8 else tuple(m.start_pos.shape)}')
                print(_m, flush=True)
                if not is_decoding:
                    _flog(_m)
            ikv = self._compressor_call(
                hidden_states, idx.compressor, indexer_state_cache,
                ic_cos, ic_sin, m.indexer_state_block_table,
                query_start_loc, m.start_pos, cr, self.rope_head_dim)
            if _dbg_en and not is_decoding and query_start_loc.numel() >= 2 \
                    and int(query_start_loc[-1].item()) > 1000:
                torch.npu.synchronize()
                _ikvfin = bool(ikv.isfinite().all().item()) if ikv.numel() else True
                _ikvmx = float(ikv.abs().max().item()) if ikv.numel() else 0.0
                _ikvm = (f'[V4-IKV] rank={_rk} L{self.layer_idx} cr={cr} '
                         f'ikv shape={tuple(ikv.shape)} stride='
                         f'{tuple(ikv.stride())} fin={_ikvfin} '
                         f'absmax={_ikvmx:.4e} ptr={ikv.data_ptr()}')
                print(_ikvm, flush=True)
                _flog(_ikvm)
            _dbg('post-indexer-compressor', ikv)
            if ikv.numel() > 0 and idx.rotate:
                ikv = rotate_activation(ikv, idx.hadamard)
            # weights (dsa_v1.py indexer_select_qli)
            weights = idx.weights_proj(hidden_states) * (
                idx.softmax_scale * idx.n_heads**-0.5)
            # quant + scatter (dsa_v1.py _indexer_quant_scatter, non-A5)
            iq_q, iq_scale = torch_npu.npu_dynamic_quant(iq, dst_type=torch.int8)
            iq_scale = iq_scale.to(torch.float16)
            if ikv.numel() > 0:
                ikv_q, ikv_scale = torch_npu.npu_dynamic_quant(
                    ikv, dst_type=torch.int8)
                ikv_scale = ikv_scale.unsqueeze(-1).to(torch.float16)
                if ikv_scale.ndim < 4:
                    ikv_scale = ikv_scale.unsqueeze(-1)
                self._paged_scatter(indexer_k_cache, ic_slot_mapping, ikv_q)
                self._paged_scatter(indexer_scale_cache, ic_slot_mapping,
                                    ikv_scale)
            # lightning_indexer (dsa_v1.py _indexer_qli)
            compress_topk_idxs, _ = dllm.lightning_indexer(
                query=iq_q, key=indexer_k_cache,
                weights=weights.to(torch.float16),
                query_dequant_scale=iq_scale,
                key_dequant_scale=indexer_scale_cache.squeeze(-2).to(torch.float16),
                query_quant_mode=0, key_quant_mode=0,
                actual_seq_lengths_query=query_start_loc[1:],
                actual_seq_lengths_key=m.indexer_kvlens,
                block_table=m.indexer_k_block_table, metadata=m.qli_metadata,
                layout_query='TND', layout_key='PA_BSND',
                sparse_count=idx.index_topk, sparse_mode=3,
                pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
                cmp_ratio=4, return_value=False)
            _dbg('post-lightning-indexer', compress_topk_idxs)

        # CHUNK-2+ PREFILL DIAGNOSTIC: start_pos>1000 means this is a
        # continuation chunk of a long prompt (chunk1 start_pos=0). Dumps the
        # lightning_indexer topk index range + the cmp/ori bounds the
        # sparse_attn will use, to settle WHY pt>max_prefill_token_num
        # retrieval fails despite seqused_kv=FULL (Fix A) + indexer_kvlens=FULL.
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and int(m.start_pos[0].item()) > 1000):
            _crk = (dist.get_rank() if (dist.is_available()
                    and dist.is_initialized()) else 0)
            if _crk == 0:
                _sp = m.start_pos.tolist()
                _ikvl = (m.indexer_kvlens.tolist()
                         if getattr(m, 'indexer_kvlens', None) is not None
                         and m.indexer_kvlens.numel() else None)
                _ccmp = m.cu_seqlens_cmp_kv
                if cr == 4 and compress_topk_idxs is not None \
                        and compress_topk_idxs.numel():
                    _ti_min = int(compress_topk_idxs.min().item())
                    _ti_max = int(compress_topk_idxs.max().item())
                    _ti_sh = tuple(compress_topk_idxs.shape)
                else:
                    _ti_min = _ti_max = -1; _ti_sh = None
                _ckv_bt = (tuple(m.compress_kv_block_table.shape)
                           if getattr(m, 'compress_kv_block_table', None)
                              is not None else None)
                print(f'[V4-C2] rank={_crk} L{self.layer_idx} cr={cr} '
                      f'start_pos={_sp} seqused_kv={seq_lens.tolist()} '
                      f'indexer_kvlens={_ikvl} '
                      f'cu_seqlens_cmp_kv={_ccmp} '
                      f'topk_idx=[{_ti_min},{_ti_max}] shape={_ti_sh} '
                      f'cmp_bt={_ckv_bt} '
                      f'cu_ori_kv={cu_seqlens_ori_kv.tolist() if cu_seqlens_ori_kv.numel()<=8 else tuple(cu_seqlens_ori_kv.shape)} '
                      f'qsl={query_start_loc.tolist() if query_start_loc.numel()<=8 else tuple(query_start_loc.shape)}',
                      flush=True)

        # ---- sparse attention (dsa_v1.py npu_sparse_attn_sharedkv) ----
        # sw=128 drift root cause (0908, runtime-confirmed): the opaque
        # npu_sparse_attn_sharedkv kernel in PA_ND decode uses seqused_kv for
        # BOTH the ori (swa) read AND the cmp (compress_kv) read bound
        # (=seqused_kv/cmp_ratio when cu_seqlens_cmp_kv is empty/None). It does
        # NOT honor a non-empty cu_seqlens_cmp_kv in PA_ND decode -- verified
        # by Candidate B (passing a FULL compressed cumsum as cu_seqlens_cmp_kv
        # while keeping seqused_kv=OFFSET): GSM8K 10q stayed at 80% (vs ~83%
        # baseline, within noise), drift failure mode unchanged. vllm-ascend
        # avoids the drift because it keeps seqused_kv=FULL for decode (its swa
        # block_table is full-extent, NOT compacted); lmdeploy's
        # WindowBlockManager COMPAcTs the swa table (drops evicted
        # logical_blocks), so a FULL seqused_kv OOBs the ori read -> garbage
        # from step 1 (verified: setting seq_lens=position_id+1 produced
        # "package package" garbage). So lmdeploy's eviction-based sw=128 is
        # structurally incompatible with this kernel's PA_ND decode path: the
        # ori read needs OFFSET (compacted table) but the cmp read needs FULL,
        # and the kernel shares one seqused_kv for both -- no decouple knob.
        # Production therefore ships sw=-1 (98% GSM8K, full-history KV). To
        # make sw=128 viable would require either (a) a kernel that honors a
        # separate cmp seqused / cu_seqlens_cmp_kv in PA_ND decode (C++ work in
        # _C_ascend.so, beyond lmdeploy Python), or (b) matching vllm by
        # disabling swa eviction (mask-only windowing = sw=-1 memory for the
        # swa pool, defeating the eviction's purpose). cu_seqlens_cmp_kv stays
        # None here (matching the pre-fix state); seqused_kv stays the engine's
        # OFFSET kv_seqlens (ori-safe). See memory dsv4-sw128-question-scrollout-drift.
        attn_kwargs = dict(
            ori_kv=swa_kv_cache, cmp_kv=compress_kv_cache,
            ori_block_table=block_table, cmp_block_table=m.compress_kv_block_table,
            cu_seqlens_q=query_start_loc, seqused_kv=seq_lens,
            sinks=self.attn_sink, metadata=m.sas_metadata,
            softmax_scale=self.scale, cmp_ratio=cr,
            ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=self.window_size - 1, ori_win_right=0,
            layout_q='TND', layout_kv='PA_ND',
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=m.cu_seqlens_cmp_kv)
        if cr == 4:
            attn_kwargs['cmp_sparse_indices'] = compress_topk_idxs
        attn_output = dllm.sparse_attn_sharedkv(q, **attn_kwargs)[0]
        _dbg('post-sparse-attn', attn_output)
        # CHUNK-2+ finiteness: NaN/inf here means the attention op produced
        # garbage (compress_kv read OOB / cmp bound wrong / indexer indices
        # invalid) even though the bounds above looked correct.
        if (not is_decoding and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and m is not None and getattr(m, 'start_pos', None) is not None
                and m.start_pos.numel()
                and int(m.start_pos[0].item()) > 1000):
            _crk2 = (dist.get_rank() if (dist.is_available()
                     and dist.is_initialized()) else 0)
            if _crk2 == 0:
                _fin = bool(attn_output.isfinite().all().item())
                _amx = float(attn_output.abs().max().item())
                print(f'[V4-C2-OUT] rank={_crk2} L{self.layer_idx} cr={cr} '
                      f'attn_out fin={_fin} absmax={_amx:.4e} '
                      f'shape={tuple(attn_output.shape)}', flush=True)

        # ---- o_proj rotary + o_proj (same as SWA) ----
        dllm.inplace_partial_rotary_mul(
            attn_output.unsqueeze(1), cos, -sin,
            rotary_mode='interleave',
            partial_slice=[self.nope_head_dim, self.head_dim])
        return self._forward_o_proj(attn_output)


class DeepseekV4MoEGate(nn.Module):
    """DeepSeek V4 MoE gate.

    Holds the router weight (F32) and, per layer, either the noaux_tc
    score-correction bias (regular layers) or the hash-routing tid2eid table
    (the first ``num_hash_layers`` layers). ``scoring_func='sqrtsoftplus'``
    is stored for Task B; the real routing/topk logic is not wired here.
    """

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim), dtype=torch.float32, device=device),
            requires_grad=False)

        self.num_hash_layers = getattr(config, 'num_hash_layers', 0)
        self.hash = layer_idx < self.num_hash_layers
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.int64, device=device),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        else:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32, device=device), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor,
                input_ids: torch.Tensor = None):
        """sqrtsoftplus scoring + noaux_tc topk.

        Mirrors vllm FusedMoE routing for V4. For the first
        ``num_hash_layers`` layers, routing is hash-based: the expert ids come
        from a precomputed ``tid2eid`` lookup table indexed by token id (see
        vllm-ascend ``moe_gating_top_k_hash`` / ``experts_selector``); the
        weights are still the sqrtsoftplus router scores of the selected
        experts, normalized/scaled the same way as the regular path.
        TODO: switch to dlinfer moe_gating_top_k for fused routing.
        """
        # router logits: [tok, n_routed_experts]
        router_logits = F.linear(hidden_states.float(), self.weight)
        # sqrtsoftplus: scores = sqrt(softplus(x))
        scores = torch.sqrt(F.softplus(router_logits))
        if self.hash:
            # hash routing: expert ids pre-assigned per token id.
            assert input_ids is not None, \
                'hash-routing layers require input_ids (tid2eid lookup)'
            # mHC may feed the MoE tok*hc_mult rows (one per hc-mult copy)
            # while input_ids is per-token; expand to match (token-major
            # flatten order == repeat_interleave).
            n = hidden_states.shape[0]
            if input_ids.numel() != n:
                factor = n // input_ids.numel()
                input_ids = input_ids.repeat_interleave(factor)
            topk_ids = self.tid2eid[input_ids.long()].to(torch.int32)  # [n, k]
            # weights = the scored router logits of the selected experts
            topk_weights = scores.gather(1, topk_ids.long())          # [n, k]
        else:
            # noaux_tc: the correction bias affects expert SELECTION only,
            # not the weight (vllm gathers weights from the UNBIASED scores;
            # using biased scores as weights flattens the distribution when
            # the bias is near-uniform, which DSv4-Flash's is).
            if self.e_score_correction_bias is not None:
                scores_for_choice = scores + self.e_score_correction_bias
            else:
                scores_for_choice = scores
            _, topk_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            # weights = UNBIASED sqrtsoftplus scores of the selected experts
            topk_weights = scores.gather(1, topk_ids.long())
        # vllm sqrtsoftplus: normalize THEN multiply by routed_scaling_factor
        # (both apply when norm_topk_prob -- not xor). The scaling lives in the
        # gate; MoE.forward adds the unscaled shared expert on top.
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


class DeepseekV4MoE(nn.Module):
    """DeepSeek V4 MoE block: gate + fused routed experts + shared expert."""

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # SwiGLU gate/up clamp (vllm SiluAndMulWithClamp): the gate is clamped
        # to [-inf, +L] and the up projection to [-L, +L] BEFORE silu/mul.
        # Applied to both routed experts (manual dispatch) and the shared
        # expert; without it the MoE output drifts enough to scramble logits.
        self.swiglu_limit = getattr(config, 'swiglu_limit', None)
        self.renormalize = self.top_k > 1 and self.norm_topk_prob
        self.layer_idx = layer_idx

        self.gate = DeepseekV4MoEGate(config, layer_idx, dtype=dtype, device=device)

        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=False,
            quant_config=quantization_config,
            layer_idx=layer_idx,
        )

        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config,
                intermediate_size=intermediate_size,
                dtype=dtype,
                device=device,
                is_shared_expert=True,
            )
            # The shared expert is TP-sharded (is_tp=True, all_reduce=False,
            # set in DeepseekV2MLP for dp==1 OR dp>1-with-ep>1) and reduced
            # over its mlp_tp group in MoE.forward -- NOT folded into the EP
            # all_reduce. Under dp1 mlp_tp==ep==16 (groups coincide); under
            # dp2 mlp_tp=8<ep=16 (groups differ), so folding would sum the
            # shared partial over the wrong rank count. Either way the
            # shared linears would otherwise inherit the global mlp_tp_mode
            # (DP_TP when attn_tp<mlp_tp, e.g. the dp1 asymmetric topology),
            # whose gather/reduce_scatter operates on the *token* axis --
            # that gathers tokens then, because all_reduce=False, never
            # scatters back, yielding a token-count mismatch with the
            # routed-expert output. The deferred-reduce design needs a plain
            # sharded matmul (partial over the original tokens), so force
            # DEFAULT tp_mode: the usual 1/mlp_tp weight shard, no
            # gather/scatter, partial summed by the mlp_tp all_reduce.
            for _lin in (self.shared_experts.gate_up_proj, self.shared_experts.down_proj):
                _lin.tp_mode = TPMode.DEFAULT
                _lin.dp_gather = False
            # The shared expert's SwiGLU must also clamp to swiglu_limit
            # (vllm builds the V4 shared MLP with SiluAndMulWithClamp).
            # DeepseekV2MLP ships a plain SiluAndMul; swap in a clamped one
            # so the shared path matches the routed path's activation.
            if self.swiglu_limit is not None:
                self.shared_experts.act_fn = SiluAndMulWithClamp(self.swiglu_limit)

    def forward(self, hidden_states: torch.Tensor,
               input_ids: torch.Tensor = None):
        """MoE forward: gate + manual expert dispatch + shared expert.

        Fallback: manual dispatch instead of dlinfer grouped_matmul_swiglu_quant
        (task: switch to dlinfer grouped_matmul_swiglu_quant_v2).
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # Path B: standard MC2 EP dispatch (``V4_MOE_BACKEND=mc2``), engaged only
        # on decode steps where op_backend selected MC2. Prefill / mixed steps
        # (ALLGATHER on A2) fall through to the naive path below so the EP
        # collective stays balanced across ranks. MC2 is the A2 alltoall-
        # equivalent: each rank dispatches its OWN tokens to the expert-owning
        # ranks (no spanning replication), combine routes them back. Idle DP
        # groups are handled by a uniform EP-wide pad + x_active_mask (built in
        # _forward_mc2), so the fused dispatch/combine NPU ops (no host sync)
        # stay graph-capturable.
        # NOTE (prefill, 2026-08-19): the naive spanning path below is correct
        # for SMALL prefill but catastrophic for large (>=~512-token) chunks:
        # its per-MoE-layer full-hidden dist.all_gather across ALL EP ranks
        # (61x ~hundreds-of-MB each) hangs HCCL (~48s) and degrades the stream
        # (507015 on every subsequent request). Routing prefill through MC2 is
        # WORSE: the A3 aclnnMoeDistributeDispatchV4 op has a HARD xDim0<=512
        # tiling limit (prefill chunks exceed it -> EZ1008 -> 561002 -> lingering
        # async 507015), then an HCCL_BUFFSIZE wall (needs ~4.35MB*maxBs), then
        # an HBM activation OOM. The CORRECT prefill path is the standard
        # dlinfer fused_moe ALLTOALL dispatch (fused_moe_all2all: histc +
        # all_to_all_single, no xDim0/HCCL walls), already wired in
        # dlinfer/vendor/ascend/torch_npu_ops.py fused_moe() L675, but NOT yet
        # hooked into this custom MoE.forward (needs expert_ids_per_ep_rank set
        # + internal-DP2 idle-group handling verified on device). See memory
        # dsv4-a3-prefill-moe-broken.
        if V4_MOE_BACKEND == 'mc2' and getattr(self.experts, 'ep_size', 1) > 1:
            _mmd = getattr(get_step_ctx_manager().current_context(), 'moe_metadata', None)
            if _mmd is not None:
                if _mmd.moe_comm_type == DlinferMoECommType.MC2:
                    return self._forward_mc2(hidden_states, input_ids, orig_shape)
                # ALLTOALL = A3 prefill (>mc2_token_capacity). Route through the
                # fused_moe_all2all path (``_forward_all2all``) instead of the
                # naive spanning all_gather below, which hangs HCCL on large
                # prefill and degrades the stream (507015). Gated by
                # V4_PREFILL_ALL2ALL so the naive path remains the default.
                if (_mmd.moe_comm_type == DlinferMoECommType.ALLTOALL
                        and os.environ.get('V4_PREFILL_ALL2ALL', '0') == '1'):
                    return self._forward_all2all(hidden_states, input_ids, orig_shape)
        # EP-spanning all_gather (DP>1 with spanning EP, i.e. ep_size > mlp_tp).
        # The EP all_reduce below sums each rank's routed-expert partial
        # position-wise across ALL EP ranks; that is correct only if every
        # EP rank computed its experts on the SAME token set. lmdeploy's
        # internal DP feeds each DP group its own tokens (group0 real,
        # group1 idle/dummy), so without this gather the two groups'
        # different tokens mix per-position -> non-deterministic garbage
        # (verified on device: same prompt/temp=0 gave different trajectories
        # run-to-run). Gather hidden_states AND input_ids across the EP group
        # so every rank sees the concatenated token set; the gate (hash
        # routing via input_ids) then routes the full set consistently on
        # every rank. MLA attention has no cross-DP collective, so it stays
        # per-DP-group with its own KV cache (8x, not 16x) -- only the MoE
        # input is replicated here. dp1: mlp_tp == ep_size -> no gather
        # (each rank already holds the full set). After the EP all_reduce we
        # slice back to this rank's own tokens.
        ep = self.experts
        ep_size = getattr(ep, 'ep_size', 1)
        mlp_ws, _ = (get_tp_world_rank('mlp') if ep_size > 1 else (1, 0))
        gather_ep = ep_size > 1 and mlp_ws < ep_size
        my_n = hidden_states.shape[0]
        ep_rank = 0
        max_n = my_n
        if gather_ep:
            _dctx = get_dist_manager().current_context()
            ep_rank = _dctx.ep_rank
            _ep_group = _dctx.ep_gpu_group
            if os.environ.get('V4_DEBUG_MOE', '0') == '1':
                print(f'[V4-MOE-IN] rank={dist.get_rank()} ep_rank={ep_rank} '
                      f'my_n={my_n} hs={tuple(hidden_states.shape)} '
                      f'iid={None if input_ids is None else tuple(input_ids.shape)}',
                      flush=True)
            # Pad my_n to the EP-group max before all_gather. lmdeploy's
            # internal DP leaves the idle/sleeping DP group with a token count
            # that differs from the active group's AND changes across requests
            # (req1: 1 dummy token; a later request's sleeping group: 0).
            # A naive all_gather of [my_n, hidden] then mismatched shapes across
            # ranks -> HCCL dispatch timeout / engine hang (verified: 3rd
            # request hung 13min). vllm's external DP never has idle ranks, so
            # its naive_dp_ep all_gather always sees matching shapes. all_reduce
            # the per-rank count (MAX), zero-pad this rank to max_n, gather, and
            # after the EP combine slice this rank's REAL tokens out of its own
            # padded slice.
            #
            # Graph capture: the engine sets meta.padding_batch_size =
            # max(all_num_tokens) for dp>1 decode (agent.py), and the graph key
            # (ascend_cudagraph.get_graph_key) captures at that uniform size on
            # EVERY EP rank. So under capture every rank's num_tokens is
            # already identical -> max_n == my_n, pad == 0. The
            # all_reduce(MAX).item() below is a host sync that breaks npu graph
            # capture (107027/107030), so skip it entirely while capturing --
            # the all_gather alone is balanced and shape-uniform. Eager decode
            # keeps the dynamic all_reduce path (the idle group's real token
            # count still varies across requests). Capturing is a class flag set
            # only inside the torch.npu.graph capture context (engine steps all
            # ranks in lockstep, so every rank toggles it the same step -> the
            # skipped collective stays balanced across ranks).
            from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
                import AscendGraphRunner
            _capturing = getattr(AscendGraphRunner, 'capturing', False)
            if _capturing:
                max_n = my_n          # uniform across EP ranks under capture
                _pad = 0
            else:
                _cnt = torch.tensor([my_n], device=hidden_states.device,
                                    dtype=torch.int64)
                if os.environ.get('V4_DEBUG_MOE', '0') == '1':
                    _mi = torch.npu.mem_get_info()
                    print(f'[V4-MOE] rank={dist.get_rank()} ep_rank={ep_rank} '
                          f'my_n={my_n} hs={tuple(hidden_states.shape)} '
                          f'iid={None if input_ids is None else tuple(input_ids.shape)} '
                          f'cap={_capturing} '
                          f'free={_mi[0]//1048576}MB total={_mi[1]//1048576}MB',
                          flush=True)
                dist.all_reduce(_cnt, op=dist.ReduceOp.MAX, group=_ep_group)
                max_n = int(_cnt.item())
                _pad = max_n - my_n
                if os.environ.get('V4_DEBUG_MOE', '0') == '1':
                    print(f'[V4-MOE] rank={dist.get_rank()} max_n={max_n} '
                          f'pad={_pad}', flush=True)
            if _pad > 0:
                _hpad = hidden_states.new_zeros(_pad, hidden_states.shape[1])
                hidden_states = torch.cat([hidden_states, _hpad], dim=0)
                if input_ids is not None:
                    _ii = input_ids.view(-1)
                    _ipad = _ii.new_zeros(_pad, dtype=_ii.dtype)
                    input_ids = torch.cat([_ii, _ipad], dim=0)
            _gh = [torch.empty_like(hidden_states) for _ in range(ep_size)]
            dist.all_gather(_gh, hidden_states, group=_ep_group)
            hidden_states = torch.cat(_gh, dim=0)
            if input_ids is not None:
                _ii = input_ids.view(-1)
                _gi = [torch.empty_like(_ii) for _ in range(ep_size)]
                dist.all_gather(_gi, _ii, group=_ep_group)
                input_ids = torch.cat(_gi, dim=0)
        topk_weights, topk_ids = self.gate(hidden_states, input_ids=input_ids)
        num_tokens = hidden_states.shape[0]

        # EP-aware manual expert dispatch. The FusedMoE weights are EP-sharded:
        # each rank holds only its local experts (self.experts.expert_list =
        # the global expert ids on this rank; gate_up/down.weight is indexed
        # by LOCAL id). The gate is replicated, so every rank computes the same
        # global topk_ids. Each rank computes only its local experts'
        # contributions for all tokens, then we all-reduce across the EP group
        # so every token accumulates the full sum over its top_k experts.
        exp = self.experts
        ep_size = getattr(exp, 'ep_size', 1)
        expert_list = getattr(exp, 'expert_list', None)
        if ep_size > 1 and expert_list is not None:
            g2l = {int(g): li for li, g in enumerate(expert_list)}
        else:
            n_local = exp.gate_up.weight.shape[0]
            g2l = {li: li for li in range(n_local)}

        # Fused grouped MoE via dlinfer (decode-scale token counts only).
        # The FusedMoE weights are EP-sharded: this rank holds only its
        # n_local experts (exp.gate_up.weight / exp.down.weight, indexed by
        # LOCAL id 0..n_local-1). The gate is replicated so every rank computes
        # the same GLOBAL topk_ids; we remap them to LOCAL ids (non-local slots
        # -> sentinel 0 with weight 0, which npu_moe_init_routing_v2 /
        # npu_moe_token_unpermute turn into an exactly zero contribution --
        # mathematically identical to the dense {0,1}-mask x weight, since
        # swiglu(0)=0 and 0@w=0). Each rank thus computes only its local
        # experts' partial over ALL tokens; the EP all_reduce below sums the
        # per-token partials across ranks. This collapses the former dense
        # top_k x n_local Python loop (128 dense F.linear + clamp + silu + mul
        # per layer, ~250K tiny elementwise ops/gap in the decode trace) into a
        # single grouped gemm + fused npu_swiglu (swiglu_limit clamp folded in
        # via _grouped_mlp).
        #
        # Gated by token count: _grouped_mlp upcasts to float32 (peak
        # ~[num_tokens*top_k, 2*ffn]) which OOMs at prefill scale but is tiny at
        # decode scale. Above the threshold fall back to the per-expert dense
        # loop (small per-expert peak, OOM-safe). See V4_FUSED_MOE_MAX_TOKENS.
        use_fused = hidden_states.shape[0] <= V4_FUSED_MOE_MAX_TOKENS
        if use_fused:
            device = hidden_states.device
            if (not getattr(self, '_g2l_tbl_built', False)
                    or self._g2l_tbl.device != device
                    or self._is_local_mask.dtype != topk_weights.dtype):
                _tbl = torch.zeros(self.num_experts, dtype=torch.int32,
                                   device=device)
                _ism = torch.zeros(self.num_experts, dtype=topk_weights.dtype,
                                   device=device)
                for g, l in g2l.items():
                    _tbl[g] = l
                    _ism[g] = 1.0
                self._g2l_tbl = _tbl          # [num_experts] int32, address-stable
                self._is_local_mask = _ism   # [num_experts] bf16, address-stable
                self._g2l_tbl_built = True
            topk_ids_local = self._g2l_tbl[topk_ids]                    # [n,k] in [0,n_local)
            topk_w_local = topk_weights * self._is_local_mask[topk_ids]  # non-local -> 0
            from dlinfer.vendor.ascend.moe import fused_moe_naive
            out = fused_moe_naive(
                hidden_states,
                exp.gate_up.weight,     # [n_local, 2*ffn, hidden]
                exp.down.weight,        # [n_local, hidden, ffn]
                topk_w_local,
                topk_ids_local,
                self.top_k,
                renormalize=False,      # gate already normalized + scaled
                chunked_moe_layout=None,
                swiglu_limit=(float(self.swiglu_limit)
                              if self.swiglu_limit is not None else 0.0),
            )
        else:
            # Dense EP-aware expert dispatch (prefill / large token counts).
            # Compute every local expert for ALL tokens at fixed shape,
            # weighted by a {0,1} routing mask x weight -- static-shape (no
            # data-dependent boolean indexing / host-sync). One expert at a
            # time keeps the float peak at [num_tokens, 2*ffn] (vs the fused
            # path's [num_tokens*top_k, 2*ffn] float32), avoiding the prefill
            # OOM.
            out = torch.zeros(hidden_states.shape[0], hidden_states.shape[1],
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)
            for i in range(self.top_k):
                expert_ids = topk_ids[:, i]      # [num_tokens] global expert ids
                weights = topk_weights[:, i]    # [num_tokens]
                for global_eid, local_eid in g2l.items():
                    mask = (expert_ids == global_eid).to(out.dtype)   # [n] {0,1}
                    coef = (mask * weights).unsqueeze(-1).to(out.dtype)
                    # gate_up.weight [num_exp, 2*ffn, hidden]; down [num_exp,hidden,ffn]
                    w_gate_up = exp.gate_up.weight[local_eid]
                    w_down = exp.down.weight[local_eid]
                    gu = F.linear(hidden_states, w_gate_up)          # [n, 2*ffn]
                    if self.swiglu_limit is not None:
                        gate = torch.clamp(gu[:, :self.ffn_dim],
                                           max=self.swiglu_limit)
                        up = torch.clamp(gu[:, self.ffn_dim:],
                                         min=-self.swiglu_limit,
                                         max=self.swiglu_limit)
                        act = F.silu(gate) * up
                    else:
                        act = F.silu(gu[:, :self.ffn_dim]) * gu[:, self.ffn_dim:]
                    expert_out = F.linear(act, w_down)               # [n, hidden]
                    out = out + expert_out * coef

        # Shared expert: TP-sharded partial (is_tp=True, all_reduce=False in
        # DeepseekV2MLP). It must be summed over the ranks that hold
        # complementary shards -- its mlp_tp group. The EP all_reduce sums the
        # ROUTED-expert partials over the EP group. These two groups coincide
        # only when mlp_tp == ep_size (dp1: both 16); then folding the shared
        # partial into the EP all_reduce is valid (one reduce sums both). When
        # they differ (dp2: mlp_tp=8 < ep=16), the shared expert is sharded
        # only 8 ways, so folding it into the 16-way EP all_reduce would sum
        # each shared dim over the wrong rank count and blow up the output --
        # reduce shared over its mlp_tp group and routed over the EP group
        # separately, then add. This matches vllm's DP2xTP8xEP16 structure.
        if ep_size > 1:
            dist_ctx = get_dist_manager().current_context()
            if self.shared_experts is not None:
                shared_out = self.shared_experts(hidden_states)
                mlp_ws, _ = get_tp_world_rank('mlp')
                if mlp_ws == ep_size:
                    # groups coincide (dp1): fold shared into the EP reduce.
                    out = out + shared_out
                    all_reduce(out, group=dist_ctx.ep_gpu_group)
                else:
                    # groups differ (dp2): reduce separately then add.
                    all_reduce(out, group=dist_ctx.ep_gpu_group)
                    all_reduce(shared_out, group=dist_ctx.mlp_tp_group.gpu_group)
                    out = out + shared_out
            else:
                all_reduce(out, group=dist_ctx.ep_gpu_group)
        elif self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        if gather_ep:
            # Slice the EP all_reduce result (the full padded+gathered set) back
            # to this rank's REAL tokens. Rank ep_rank owns the slice
            # [ep_rank*max_n : (ep_rank+1)*max_n]; its real my_n tokens sit at
            # the head of that slice (padding zeros were appended at the tail),
            # so take [ep_rank*max_n : ep_rank*max_n + my_n].
            out = out[ep_rank * max_n:ep_rank * max_n + my_n]
        if os.environ.get('V4_DEBUG_MOE', '0') == '1':
            # UNCONDITIONAL sync (no capturing gate) -- diagnostic: surfaces the
            # faulting op for prefill. In eager this is safe; if prefill were
            # graph-captured this sync raises 107027 (itself diagnostic).
            torch.npu.synchronize()
            _fin = bool(out.isfinite().all().item()) if out.numel() else True
            _mx = float(out.abs().max().item()) if out.numel() else 0.0
            print(f'[V4-MOE-OUT] rank={dist.get_rank()} L{self.layer_idx} '
                  f'my_n={my_n} out={tuple(out.shape)} '
                  f'fin={_fin} absmax={_mx:.4e} ptr={out.data_ptr()}',
                  flush=True)
        return out.view(*orig_shape)

    def _forward_mc2(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                     orig_shape: torch.Size) -> torch.Tensor:
        """Path B: standard dlinfer ``fused_moe_mc2`` EP dispatch.

        Each rank dispatches its own (padded) tokens to the expert-owning EP
        ranks and combines them back -- no spanning replication, no EP
        all_reduce. The pad + x_active_mask keep the fused dispatch/combine
        collective shape-balanced across ranks (an idle DP group pads to the
        active count with an all-False mask), and are no-ops under graph
        capture (the engine already makes per-rank token counts uniform at
        ``padding_batch_size``). Graph-capturable: ``npu_moe_distribute_*_v2``
        are fused NPU ops with no host sync (unlike the alltoall path whose
        ``combined_splits.tolist()`` would break capture -- but A2 never
        selects alltoall anyway).
        """
        ep = self.experts
        ep_size = getattr(ep, 'ep_size', 1)
        my_n = hidden_states.shape[0]
        _dctx = get_dist_manager().current_context()
        ep_rank = _dctx.ep_rank
        _ep_group = _dctx.ep_gpu_group
        # Uniform EP-wide pad (shape-balanced dispatch). Under graph capture the
        # engine already pads every rank to padding_batch_size, so my_n is
        # uniform across ranks -> no pad, no host sync. The all_reduce(MAX).item()
        # below is a host sync that breaks npu graph capture, so skip it while
        # capturing (mirrors the naive spanning path). Capturing is a class flag
        # set only inside the torch.npu.graph capture context, toggled by every
        # rank the same step -> the skipped collective stays balanced.
        from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
            import AscendGraphRunner
        _capturing = getattr(AscendGraphRunner, 'capturing', False)
        if _capturing:
            max_n = my_n
            _pad = 0
        else:
            _cnt = torch.tensor([my_n], device=hidden_states.device,
                                dtype=torch.int64)
            dist.all_reduce(_cnt, op=dist.ReduceOp.MAX, group=_ep_group)
            max_n = int(_cnt.item())
            _pad = max_n - my_n
        if _pad > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, _pad))
            if input_ids is not None:
                input_ids = F.pad(input_ids.view(-1), (0, _pad))
        # Gate on this rank's own (padded) tokens -> GLOBAL topk. The fused
        # dispatch routes by global expert id to the owning rank, so no
        # global->local remap is needed (unlike the local naive path's
        # _g2l_tbl). Padding tokens hash-route to some expert but are masked
        # out below.
        topk_weights, topk_ids = self.gate(hidden_states, input_ids=input_ids)
        # x_active_mask: real tokens True, padding False. Idle DP ranks (my_n
        # ~ 0) pad to max_n with an all-False mask -> dispatch sends nothing,
        # combine receives nothing, collective stays balanced (no HCCL hang).
        # The mask MUST be length max_n (== hidden_states.dim0 after the pad
        # above): first my_n True, last _pad False. Do NOT append _pad more
        # False entries (that made dim0 = max_n+_pad > x.dim0, which the A2
        # aclnnMoeDistributeDispatchV2 tolerated but the A3 V4 op rejects with
        # "xActiveMask dim0 != x dim0" + "params shape is empty" tiling fail).
        x_active_mask = torch.ones(max_n, dtype=torch.bool,
                                   device=hidden_states.device)
        if _pad > 0:
            x_active_mask[my_n:] = False
        _mmd = get_step_ctx_manager().current_context().moe_metadata
        from dlinfer.vendor.ascend.moe import fused_moe_mc2
        out = fused_moe_mc2(
            hidden_states,
            ep.gate_up.weight,
            ep.down.weight,
            topk_weights,
            topk_ids,
            renormalize=False,          # gate already normalized + scaled
            ep_size=ep_size,
            ep_rank=ep_rank,
            moe_group_name=_mmd.moe_group_name,
            x_active_mask=x_active_mask,
            swiglu_limit=(float(self.swiglu_limit)
                          if self.swiglu_limit is not None else 0.0),
        )
        if _pad > 0:
            out = out[:my_n].contiguous()
        # Shared expert: TP-sharded partial (is_tp=True, all_reduce=False),
        # summed over its mlp_tp group (node-local 8-way under dp2), NOT folded
        # into the EP dispatch. Matches vllm's DP2xTP8xEP16: routed EP16 via MC2
        # (cross-node), shared TP8 (node-local). Runs on this rank's own real
        # tokens (shared is per-token; padding positions are sliced off).
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states[:my_n])
            mlp_ws, _ = get_tp_world_rank('mlp')
            if mlp_ws > 1:
                all_reduce(shared_out, group=_dctx.mlp_tp_group.gpu_group)
            out = out + shared_out
        return out.view(*orig_shape)

    def _forward_all2all(self, hidden_states: torch.Tensor,
                         input_ids: torch.Tensor,
                         orig_shape: torch.Size) -> torch.Tensor:
        """Path C: standard dlinfer ``fused_moe_all2all`` EP dispatch for PREFILL.

        Engaged when op_backend selects ALLTOALL (A3 prefill, token count >
        mc2_token_capacity). Unlike MC2 (``npu_moe_distribute_dispatch_v2``,
        which on A3 has a HARD xDim0<=512 tiling limit + an HCCL_BUFFSIZE wall
        + a large HBM activation OOM at prefill scale), the all2all path
        dispatches via ``histc`` + ``all_to_all_single`` -- sparse, no
        xDim0/HCCL walls, ~1/ep peak HBM -- so it handles prefill-scale token
        counts. This is the INTENDED prefill path (``select_moe_comm_type``
        returns ALLTOALL for A3 prefill) but is NOT reachable through the
        standard ``fused_moe`` wrapper, because that wrapper's ``moe_prepare``
        TP-splits the tokens (tp_size=8); our V4 routed experts are EP-only
        (tp_world_size=1, no TP sharding, same as ``_forward_mc2``), so a
        TP-split would mis-shard them. We therefore call ``fused_moe_all2all``
        DIRECTLY with the full (un-split) token set.

        Internal-DP2 idle group: an idle DP rank enters with 0 input tokens;
        its dispatch sends 0 yet it still participates in the EP
        ``all_to_all_single`` (receiving the active ranks' tokens routed to
        the experts it owns, computing them, returning them in combine) -- the
        EP-native behavior, needing NO all_reduce(MAX)/x_active_mask pad
        (unlike MC2). Padding is UNSAFE here: all2all has no mask to exclude
        padding tokens, so they would pollute the output.
        """
        ep = self.experts
        ep_size = getattr(ep, 'ep_size', 1)
        _dctx = get_dist_manager().current_context()
        ep_rank = _dctx.ep_rank
        _ep_group = _dctx.ep_gpu_group
        # Gate (V4 hash routing via input_ids) on this rank's OWN tokens.
        topk_weights, topk_ids = self.gate(hidden_states, input_ids=input_ids)
        topk_ids = topk_ids.to(torch.int32)
        # expert_ids_per_ep_rank: maps each GLOBAL expert id -> its LOCAL index
        # within this ep_rank. Length = num_experts (global). Same formula as
        # DlinferFusedMoEImpl (lmdeploy/backends/dlinfer/moe.py L59), which the
        # standard path sets but our custom MoE bypasses.
        num_local_experts = ep.gate_up.weight.size(0)
        expert_ids_per_ep_rank = torch.tensor(
            [i % num_local_experts for i in range(self.num_experts)],
            dtype=torch.int32, device=hidden_states.device,
        )
        from dlinfer.vendor.ascend.moe import fused_moe_all2all
        out = fused_moe_all2all(
            hidden_states,
            ep.gate_up.weight,
            ep.down.weight,
            topk_weights,
            topk_ids,
            renormalize=False,          # gate already normalized + scaled
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=_ep_group,
            expert_ids_per_ep_rank=expert_ids_per_ep_rank,
            swiglu_limit=(float(self.swiglu_limit)
                          if self.swiglu_limit is not None else 0.0),
        )
        # Shared expert: TP-sharded partial, summed over its mlp_tp group
        # (node-local 8-way under dp2), same as _forward_mc2. Runs on this
        # rank's own tokens (shared is per-token; all2all hidden has no pad).
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)
            mlp_ws, _ = get_tp_world_rank('mlp')
            if mlp_ws > 1:
                all_reduce(shared_out, group=_dctx.mlp_tp_group.gpu_group)
            out = out + shared_out
        return out.view(*orig_shape)


class DeepseekV4DecoderLayer(nn.Module):
    """DeepSeek V4 decoder layer with mHC hooks."""

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps

        self.self_attn = DeepseekV4Attention(config, layer_idx, dtype=dtype, device=device)
        self.mlp = DeepseekV4MoE(config, layer_idx, dtype=dtype, device=device)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # mHC parameters (F32). mix_hc = (2 + hc_mult) * hc_mult.
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device),
                                       requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device),
                                      requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device), requires_grad=False)

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor,
               hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """mHC pre-hook. Mirrors dsa_v1.py npu_hc_pre_v2 call.
        Returns (out, post, comb) for hc_post.
        """
        out, post, comb = dllm.hc_pre(
            x, hc_fn, hc_scale, hc_base,
            self.hc_mult, self.hc_sinkhorn_iters,
            self.norm_eps, self.hc_eps)
        return out, post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor):
        """mHC post-hook. Mirrors dsa_v1.py npu_hc_post (4D unsqueeze)."""
        y = dllm.hc_post(
            x.unsqueeze(0), residual.unsqueeze(0),
            post.unsqueeze(0), comb.unsqueeze(0))
        return y.squeeze(0)

    def forward(self, hidden_states: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                swa_kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
                block_table: torch.Tensor, seq_lens: torch.Tensor,
                query_start_loc: torch.Tensor, sas_metadata,
                residual: torch.Tensor = None,
                dsa_caches=None, dsa_meta=None,
                input_ids: torch.Tensor = None,
                is_decoding: bool = False) -> torch.Tensor:
        """Decoder layer forward (SWA + c4/c128).

        Mirrors vllm DeepseekV2DecoderLayer.forward:
          residual = hs.clone()
          hs = hc_pre(hs, hc_attn_fn, ...)
          hs = input_layernorm(hs)
          hs = self_attn(hs, cos, sin, ...)
          hs = hc_post(hs, residual, post, comb)
          residual = hs.clone()
          hs = hc_pre(hs, hc_ffn_fn, ...)
          hs = post_attention_layernorm(hs)
          hs = mlp(hs)
          hs = hc_post(hs, residual, post, comb)
        """
        # --- attention block ---
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn,
            self.hc_attn_scale, self.hc_attn_base)
        # 0904 chunk2 crash pinpoint: bisect residual.clone()/hc_pre vs
        # input_layernorm vs self_attn. [V4-C2-MH] passed (mHC OK), [V4-C2-L0]
        # (after layer) raised -> fault in layer 0. [V4-C2-ENT] (first sync in
        # _forward_c4c128) raised -> fault is BEFORE the attention, in
        # clone/hc_pre/input_layernorm. Gate: chunk2 prefill (tok<50, not
        # decoding), first 3 layers. fsync survives 507057 -> LAST tag = good.
        if (not is_decoding and self.layer_idx < 3
                and hidden_states.shape[0] < 50
                and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'):
            _pr2 = (dist.get_rank() if (dist.is_available()
                     and dist.is_initialized()) else 0)
            if _pr2 == 0:
                torch.npu.synchronize()
                _hp = (f'[V4-C2-HP] L{self.layer_idx} POST-HCPRE SYNC OK '
                       f'shape={tuple(hidden_states.shape)} '
                       f'fin={bool(hidden_states.isfinite().all().item())} '
                       f'absmax={float(hidden_states.abs().max().item()):.4e}')
                print(_hp, flush=True)
                try:
                    with open('/tmp/v4_dbg.log', 'a') as _f:
                        _f.write(_hp + '\n'); _f.flush(); os.fsync(_f.fileno())
                except Exception:
                    pass
        hidden_states = self.input_layernorm(hidden_states)
        if (not is_decoding and self.layer_idx < 3
                and hidden_states.shape[0] < 50
                and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'):
            _pr3 = (dist.get_rank() if (dist.is_available()
                     and dist.is_initialized()) else 0)
            if _pr3 == 0:
                torch.npu.synchronize()
                _ln = (f'[V4-C2-LN] L{self.layer_idx} POST-INPUTLN SYNC OK '
                       f'shape={tuple(hidden_states.shape)} '
                       f'fin={bool(hidden_states.isfinite().all().item())} '
                       f'absmax={float(hidden_states.abs().max().item()):.4e}')
                print(_ln, flush=True)
                try:
                    with open('/tmp/v4_dbg.log', 'a') as _f:
                        _f.write(_ln + '\n'); _f.flush(); os.fsync(_f.fileno())
                except Exception:
                    pass
        _attn_raw = self.self_attn(
            hidden_states, cos, sin, swa_kv_cache, slot_mapping,
            block_table, seq_lens, query_start_loc, sas_metadata,
            dsa_caches=dsa_caches, dsa_meta=dsa_meta,
            is_decoding=is_decoding)
        if os.environ.get('V4_DEBUG_MOE', '0') == '1':
            # gate host-sync to eager only; during npu.graph capture a host
            # sync waits for ops that are recorded-not-executed -> 107027.
            from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph \
                import AscendGraphRunner
            _cap = getattr(AscendGraphRunner, 'capturing', False)
            if not _cap:
                torch.npu.synchronize()
                _mi = torch.npu.mem_get_info()
                print(f'[V4-LYR] rank={dist.get_rank()} L{self.layer_idx} '
                      f'post-attn free={_mi[0]//1048576}MB '
                      f'hs={tuple(_attn_raw.shape)}', flush=True)
        hidden_states = self.hc_post(_attn_raw, residual, post, comb)

        # --- FFN/MoE block ---
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn,
            self.hc_ffn_scale, self.hc_ffn_base)
        hidden_states = self.post_attention_layernorm(hidden_states)
        _mlp_raw = self.mlp(hidden_states, input_ids=input_ids)
        hidden_states = self.hc_post(_mlp_raw, residual, post, comb)

        return hidden_states


class DeepseekV4Model(nn.Module):
    """DeepSeek V4 model body."""

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # RoPE: YaRN over qk_rope_head_dim. compress_rope_theta (per-layer for
        # compressed layers) is a Task B forward concern.
        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = get_rope_theta(config)
        rope_params = dict(
            emb_type=emb_type,
            dim=rope_dim,
            max_position_embeddings=rope_max_pos_emb,
            base=rope_base,
        )
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_emb = build_rotary_embedding(**rope_params)

        # mHC head parameters (applied before lm_head).
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32, device=device), requires_grad=False)
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32, device=device), requires_grad=False)
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32, device=device), requires_grad=False)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor,
                hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """mHC head: collapse the hc_mult axis back to [tok, hidden].

        Mirrors vllm ``DeepseekV4Model.hc_head`` (pure-python ref, not an op).
        Input ``x`` is 3D ``[tok, hc_mult, hidden]``; output is 2D
        ``[tok, hidden]``. An RMSNorm-style rsqrt is applied over the full
        hc_dim, then a sigmoid mixing weight selects across the hc_mult axis.
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()                       # [tok, hc_mult*hidden]
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt   # [tok, hc_mult]
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)  # [tok, hidden]
        return y.to(dtype)

    def forward(self,
                input_ids: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_values: list | None = None,
                attn_metadata: Any = None,
                inputs_embeds: torch.Tensor | None = None,
                dsa_inputs: list | None = None,
                # The dlinfer ascend graph capture path (ascend_cudagraph.py
                # fill_buffers_cudagraph) injects moe_metadata / state_ids into
                # the model kwargs. The V4 MoE here is a manual F.linear
                # dispatch (not a dlinfer grouped MoE op), so these are not
                # consumed -- accept and ignore so graph capture's warmup call
                # does not raise on unexpected kwargs.
                moe_metadata: Any = None,
                state_ids: torch.Tensor = None,
                **_graph_extra,
                ) -> torch.Tensor:
        """Model body forward (layer loop + hc_head + norm).

        Mirrors vllm ``DeepseekV4Model.forward``:
          embed -> expand [tok, hc_mult, hidden] -> layer loop ->
          hc_head (3D -> 2D) -> norm.

        ``dsa_inputs`` is a per-layer list of dicts carrying the DSA-specific
        tensors (cos, sin, swa_kv_cache, slot_mapping, block_table, seq_lens,
        query_start_loc, sas_metadata, dsa_caches, dsa_meta). In the single-NPU
        test path the caller builds it; in the engine path it is built by the
        ascend backend (``v4_dsa.build_v4_dsa_inputs``) from the step context
        and forwarded here via ``DeepseekV4ForCausalLM.forward``.
        """
        # lmdeploy delivers input_ids as 2D [batch, seq]; the V4 body (ported
        # from vllm) operates on flat tokens [num_tokens, hidden] throughout
        # (hc expand/layer loop/hc_head/lm_head all assume 2D), so flatten
        # before embedding -- matching vllm's token-flat convention.
        if input_ids is not None and input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.dim() > 2:
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        hidden_states = inputs_embeds                       # [tok, hidden]

        if os.environ.get('V4_DEBUG_MOE', '0') == '1':
            _mi = torch.npu.mem_get_info()
            print(f'[V4-FWD] rank={dist.get_rank()} start tok={hidden_states.shape[0]} '
                  f'free={_mi[0]//1048576}MB', flush=True)

        # CHUNK-2+ position_ids probe: if position_ids[0]==0 (not the history
        # start 2304) on a small-q continuation chunk, RoPE is wrong -> query
        # won't match chunk-1 KV -> retrieval fails despite correct bounds.
        # Gated on not-capturing: .tolist() is a host sync that crashes
        # npu.graph capture (107027/ERR99999); this forward runs at capture.
        if (os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and position_ids is not None and position_ids.numel()
                and position_ids.numel() < 50):
            try:
                from dlinfer.framework.lmdeploy_ext.cudagraph.\
                    ascend_cudagraph import AscendGraphRunner as _AGRpos
                _cap_pos = getattr(_AGRpos, 'capturing', False)
            except Exception:
                _cap_pos = False
            if not _cap_pos:
                _pr = (dist.get_rank() if (dist.is_available()
                        and dist.is_initialized()) else 0)
                if _pr == 0:
                    _pid = position_ids.reshape(-1).tolist()
                    if _pid[0] > 1000:
                        print(f'[V4-POS] rank={_pr} tok={hidden_states.shape[0]} '
                              f'position_ids=[{_pid[0]}..{_pid[-1]}] '
                              f'(len={len(_pid)})', flush=True)


        # mHC: expand to [tok, hc_mult, hidden] (vllm: unsqueeze(1).repeat(1,
        # hc_mult, 1)). hc_pre inside each layer collapses this back to 2D for
        # attention, and hc_post expands it back to 3D -- so attention always
        # receives 2D [tok, hidden] (matching the c4/c128 compressor csrc).
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        # 0904 chunk2 crash pinpoint (main forward): the [V4-POS] .tolist() sync
        # above printed => no pending chunk-1 async fault. So the 507057 (which
        # surfaces at the forward's completion event query) is launched AFTER
        # [V4-POS] in chunk-2's forward. This sync catches it right after the
        # mHC expand (the first NPU op after [V4-POS]). If this RAISES -> fault
        # is in the mHC expand or a delayed chunk-1 op not caught by .tolist();
        # if it passes -> fault is in the layer loop (input_layernorm /
        # _forward_c4c128). fsync -> survives 507057 so LAST tag = fault point.
        if (os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                and position_ids is not None and position_ids.numel()
                and position_ids.numel() < 50):
            try:
                from dlinfer.framework.lmdeploy_ext.cudagraph.\
                    ascend_cudagraph import AscendGraphRunner as _AGRmh
                _cap_mh = getattr(_AGRmh, 'capturing', False)
            except Exception:
                _cap_mh = False
            if not _cap_mh:
                _mr = (dist.get_rank() if (dist.is_available()
                        and dist.is_initialized()) else 0)
                if _mr == 0:
                    torch.npu.synchronize()
                    _mh = (f'[V4-C2-MH] rank={_mr} tok={hidden_states.shape[0]} '
                           f'POST-MHC-EXPAND SYNC OK shape={tuple(hidden_states.shape)} '
                           f'fin={bool(hidden_states.isfinite().all().item())} '
                           f'absmax={float(hidden_states.abs().max().item()):.4e}')
                    print(_mh, flush=True)
                    try:
                        with open('/tmp/v4_dbg.log', 'a') as _f:
                            _f.write(_mh + '\n'); _f.flush(); os.fsync(_f.fileno())
                    except Exception:
                        pass

        # FULL-graph (V4_FULL_GRAPH_DECODE=1): compute the SHARED decode metadata
        # IN-GRAPH from the device graph-input buffers (kv_seqlens /
        # block_offsets, refreshed each step by fill_buffers_cudagraph) and the
        # position_ids forward arg, overriding the None sentinels
        # build_v4_dsa_inputs left on the replay path. The in-graph ops land in
        # the captured aclgraph's private pool at stable addresses (read
        # refreshed at replay -- no _pin needed, mirroring vllm-ascend FULL).
        # Gated on kv_seqlens being a DEVICE tensor: True on the graph capture
        # + replay path (fill_buffers repoints attn_metadata.kv_seqlens to the
        # device input buffer); False on pre-capture eager warmup (kv_seqlens
        # is CPU there) -> the eagerly-built dsa_inputs are used as-is.
        if (dsa_inputs is not None
                and os.environ.get('V4_FULL_GRAPH_DECODE', '0') == '1'
                and attn_metadata is not None
                and getattr(attn_metadata, 'is_decoding', False)
                and getattr(attn_metadata, 'kv_seqlens', None) is not None
                and attn_metadata.kv_seqlens.device.type == 'npu'):
            from lmdeploy.pytorch.backends.dlinfer.ascend.v4_dsa import \
                build_v4_decode_meta_in_graph as _build_in_graph
            _bs = int(dsa_inputs[0]['swa_kv_cache'].shape[1])
            _shared = _build_in_graph(
                position_ids, attn_metadata, self.config, _bs,
                hidden_states.device, hidden_states.dtype)
            for _idx in range(len(dsa_inputs)):
                _di = dsa_inputs[_idx]
                # SWA layers (dsa_meta is None) use the rope_theta=10000 RoPE;
                # c4/c128 layers (dsa_meta set) use compress_rope_theta=160000.
                if _di.get('dsa_meta') is None:
                    _di['cos'] = _shared['cos_cur']
                    _di['sin'] = _shared['sin_cur']
                else:
                    _di['cos'] = _shared['cos_cur_cmp']
                    _di['sin'] = _shared['sin_cur_cmp']
                    _dm = _di['dsa_meta']
                    _dm.start_pos = _shared['start_pos']
                    # compress_kv/indexer_k tables are NOT overridden here --
                    # they keep the eager pre-step dsa_kv_bt (a separate
                    # _V4StateAlloc graph-stable backing, refreshed each
                    # pre-step via assign(), same pattern as state_bt which
                    # is also not overridden in-graph). The old code aliased
                    # _shared['swa_block_table'] here -> under sliding_window
                    # swa eviction cross-request-contaminated compress_kv/
                    # indexer_k (0903 structural bug). dsa_kv_bt is full-
                    # history, own pool, slot-stable -> no contamination.
                    # Stage 1b: indexer_kvlens is now FULL (absolute) -- the
                    # in-graph build derives it from position_ids (= full_kv),
                    # NOT the offset seq_lens. Matches the eager path's
                    # full_kv_lens (from kv_lens_cpu). The offset seq_lens
                    # below is swa-only (windowed swa_block_table + mask).
                    _dm.indexer_kvlens = _shared['indexer_kvlens']
                _di['slot_mapping'] = _shared['swa_slot_mapping']
                _di['block_table'] = _shared['swa_block_table']
                _di['seq_lens'] = _shared['seq_lens']
                _di['query_start_loc'] = _shared['query_start_loc']

        for idx, layer in enumerate(self.layers):
            args = dsa_inputs[idx] if dsa_inputs is not None else {}
            if os.environ.get('V4_DEBUG_MOE', '0') == '1':
                _mi = torch.npu.mem_get_info()
                print(f'[V4-FWD] rank={dist.get_rank()} L{idx} pre free={_mi[0]//1048576}MB',
                      flush=True)
            hidden_states = layer(hidden_states, input_ids=input_ids, **args)
            # 0904 chunk2 crash pinpoint: [V4-C2-MH] (after mHC) passed, so the
            # fault is in the layer loop. This sync after each of the FIRST 3
            # layers (idx<3) catches the faulting layer: the LAST idx printed =
            # the last GOOD layer; the next layer faulted. fsync survives 507057.
            if (idx < 3 and os.environ.get('V4_DEBUG_ALLOC', '0') == '1'
                    and position_ids is not None and position_ids.numel()
                    and position_ids.numel() < 50
                    and not _cap_mh):
                _lr = (dist.get_rank() if (dist.is_available()
                        and dist.is_initialized()) else 0)
                if _lr == 0:
                    torch.npu.synchronize()
                    _lt = (f'[V4-C2-L{idx}] rank={_lr} POST-LAYER{idx} SYNC OK '
                           f'fin={bool(hidden_states.isfinite().all().item())} '
                           f'absmax={float(hidden_states.abs().max().item()):.4e}')
                    print(_lt, flush=True)
                    try:
                        with open('/tmp/v4_dbg.log', 'a') as _f:
                            _f.write(_lt + '\n'); _f.flush(); os.fsync(_f.fileno())
                    except Exception:
                        pass

        # hc_head (vllm DeepseekV4Model.hc_head): 3D -> 2D [tok, hidden]
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn,
            self.hc_head_scale, self.hc_head_base)
        # RMSNorm returns a single tensor when no residual is passed (the V4
        # model keeps residual internal to each DecoderLayer).
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class DeepseekV4ForCausalLM(nn.Module, CudaGraphMixin):
    """DeepSeek V4 Flash for CausalLM (construction + weight loading).

    Forward (model body + hc_head + norm) is implemented in
    ``DeepseekV4Model.forward``; this wrapper delegates to it and returns
    hidden_states. Logits come from ``get_logits`` (lm_head).
    """

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = DeepseekV4Model(config, dtype=dtype, device=device)
        # lm_head (not tied)
        self.lm_head = build_rowwise_linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self._load_buffers = dict()

    def support_cuda_graph(self, input_ids: torch.Tensor,
                          position_ids: torch.Tensor,
                          past_key_values: list[list[torch.Tensor]],
                          attn_metadata: Any = None,
                          inputs_embeds: torch.Tensor = None,
                          **kwargs):
        """Capture the graph only on globally-uniform decode steps (vllm-ascend
        UNIFORM_BATCH semantics, extended to internal DP).

        ``context.global_is_decoding()`` is True only when NO DP rank is doing
        a real prefill (see dp_utils.GatheredDPForwardMeta.global_is_decoding):
        for dp=1 it falls back to per-rank ``is_decoding``; for dp>1 it is the
        cross-DP aggregate, so a prefill on one DP group forces eager on ALL
        ranks for that step. This gating is REQUIRED for dp>1 graph mode:
        lmdeploy's internal DP runs a mixed step (group0 prefill eager +
        group1 decode-dummy) when one group prefills and the other is idle.
        If the idle group engaged the captured decode graph (which skips the
        EP all_reduce, see DeepseekV4MoE.forward) while the prefill group ran
        eager (with the EP all_reduce), the EP collective would desync ->
        all_reduce/all_gather shape mismatch (e.g. 6 vs 1) -> deadlock.
        global_is_decoding=False on mixed steps keeps both groups eager (the
        all_reduce(MAX)+pad path equalizes shapes), so only a fully-decoding
        step -- where every rank holds the uniform padding_batch_size -- is
        captured. Pure-decode (all groups decoding) -> graph engages -> all
        ranks replay with uniform shapes -> balanced."""
        context = self.ctx_mgr.current_context()
        return bool(context.global_is_decoding())

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]],
                attn_metadata: Any = None,
                inputs_embeds: torch.Tensor = None,
                **kwargs):
        """Forward. Mirrors lmdeploy DeepseekV2ForCausalLM.forward: delegates
        to ``self.model`` and returns hidden_states; logits are produced by
        ``get_logits`` (called by the engine / test).

        ``dsa_inputs`` (per-layer DSA cache+metadata, see
        ``DeepseekV4Model.forward``) is forwarded via ``kwargs``.  In the engine
        path ``dsa_inputs`` is not passed by the caller; instead it is built by
        the ascend backend (``v4_dsa.build_v4_dsa_inputs``) and stashed on the
        step context as ``v4_dsa_inputs`` -- fetch it here so the model body
        receives the 6-tuple paging state.
        """
        if 'dsa_inputs' not in kwargs or kwargs.get('dsa_inputs') is None:
            ctx = self.ctx_mgr.current_context()
            if ctx is not None and getattr(ctx, 'v4_dsa_inputs', None) is not None:
                kwargs['dsa_inputs'] = ctx.v4_dsa_inputs
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs)
        # lmdeploy's engine post-process expects a batch dim 0: the output is
        # sliced as hidden_states[0] -> [tokens, hidden] then last-token-per-
        # seq for sampling. The V4 body operates token-flat (2D [tokens,
        # hidden], like vllm), so add the batch axis here to match the
        # [batch, tokens, hidden] convention used by DeepseekV2 et al.
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output.

        vllm's LogitsProcessor computes the logits matmul in fp32
        (logits_dtype default = float32); a bf16 lm_head accumulation drifts
        ~0.5-1.0 logit and flips near-ties (e.g. '.' vs '.,' after "Paris").
        Match vllm: upcast hidden + lm_head weight to fp32 before the matmul.
        """
        if not getattr(self, '_lm_fp32_done', False):
            with torch.no_grad():
                self.lm_head.weight.data = self.lm_head.weight.data.float()
            self._lm_fp32_done = True
        logits = self.lm_head(hidden_states.float())
        return logits

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list):
        """Load expert weights into the fused MoE."""
        for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            break
        else:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    def _load_attn_sink(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        """Load attn_sink, narrowing to local heads when TP > 1."""
        param = params_dict[name]
        world_size, rank = get_tp_world_rank('attn')
        if world_size > 1:
            heads_per_rank = self.config.num_attention_heads // world_size
            head_start = rank * heads_per_rank
            loaded_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
        load_weight(param, loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights.

        Checkpoint tensor names use the native V4 layout (``embed.weight``,
        ``head.weight``, ``layers.{i}.attn.*``, ``layers.{i}.ffn.*``,
        ``layers.{i}.hc_*``, top-level ``hc_head_*``). They are remapped to
        the lmdeploy module tree (``model.embed_tokens``, ``lm_head``,
        ``model.layers.{i}.self_attn`` / ``.mlp`` / ``.input_layernorm``
        / ``.post_attention_layernorm`` / ``.hc_*``).
        """
        config = self.config

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        num_experts = config.n_routed_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        num_hidden_layers = config.num_hidden_layers
        num_nextn_predict_layers = getattr(config, 'num_nextn_predict_layers', 0)
        nextn_keys = [f'.layers.{num_hidden_layers + i}' for i in range(num_nextn_predict_layers)]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # skip non-param rotary buffers
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue

            # skip MTP / nextn layers
            if any(nextn_key in name for nextn_key in nextn_keys):
                continue

            # ---- name remapping -------------------------------------------------
            # native V4 names do not carry a "model." prefix; add it so they
            # align with the DeepseekV4ForCausalLM -> DeepseekV4Model tree.
            if not name.startswith('model'):
                name = f'model.{name}'

            # top-level renames
            if 'embed_tokens' not in name and 'embed.' in name:
                name = name.replace('embed.', 'embed_tokens.')
            if 'model.head.' in name and 'model.lm_head.' not in name:
                name = name.replace('model.head.', 'lm_head.')

            # expert w1/w2/w3 -> gate_proj/down_proj/up_proj (handles both
            # routed experts and shared experts)
            if '.w1.' in name:
                name = name.replace('.w1.', '.gate_proj.')
            if '.w2.' in name:
                name = name.replace('.w2.', '.down_proj.')
            if '.w3.' in name:
                name = name.replace('.w3.', '.up_proj.')

            # layer norm renames
            name = name.replace('.ffn_norm.', '.post_attention_layernorm.')
            name = name.replace('.attn_norm.', '.input_layernorm.')

            # submodule renames (attn -> self_attn, ffn -> mlp)
            name = name.replace('.ffn.', '.mlp.')
            if '.self_attn' not in name:
                name = name.replace('.attn.', '.self_attn.')

            # gate bias -> e_score_correction_bias
            if name.endswith('.gate.bias'):
                name = name.replace('.gate.bias', '.gate.e_score_correction_bias')

            # skip FP8 scales (bf16 checkpoint has none, but be safe)
            if name.endswith('.scale'):
                name = name.replace('.scale', '.weight_scale')

            # ---- dispatch ------------------------------------------------------
            if 'attn_sink' in name:
                self._load_attn_sink(name, loaded_weight, params_dict)
                continue

            if '.experts.' in name:
                self._load_weight_experts(name, loaded_weight, params_dict,
                                          expert_params_mapping=expert_params_mapping)
                continue

            # shared_experts gate_up / down stacked mapping
            handled = False
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # do not touch routed experts here (handled above)
                if '.experts.' in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                load_weight(param, loaded_weight, shard_id=shard_id)
                handled = True
                break
            if handled:
                continue

            # generic direct load
            if name in params_dict:
                param = params_dict[name]
                load_weight(param, loaded_weight)
            else:
                # Anything landing here is unmapped. For a bf16 V4 checkpoint
                # this branch should stay empty; collecting these is how the
                # "no unexpected weight" self-check is performed (see report).
                pass
