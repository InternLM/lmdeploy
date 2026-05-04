# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import os.path as osp
import re
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
    quantize_model1_fp8_sparse,
)
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.kernels.cuda.v4_compressor import (
    fill_compressed_kv,
    fill_compress_state,
    score_kv,
)
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, RMSNorm, SiluAndMul
from lmdeploy.pytorch.nn import V4Attention as NativeV4Attention
from lmdeploy.pytorch.nn import V4Indexer as NativeV4Indexer
from lmdeploy.pytorch.nn.rotary_embedding import build_rotary_embedding
from lmdeploy.pytorch.backends.rotary_embedding import RopeType, YarnParameters
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_down_linear, build_gateup_linear, build_o_proj
from lmdeploy.pytorch.nn.moe import FusedMoEV4
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

from .deepseek_v4_utils import build_compress_topk_indices, build_prefix_positions, build_window_topk_indices
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1, build_embedding

logger = get_logger('lmdeploy')

_KERNEL_MODULE_CACHE: dict[str, object] = {}


def _load_v4_kernel_module(model_path: str):
    """Load the official DeepSeek-V4 kernel helpers from the model
    directory."""
    model_path = osp.abspath(model_path)
    if model_path in _KERNEL_MODULE_CACHE:
        return _KERNEL_MODULE_CACHE[model_path]

    kernel_path = osp.join(model_path, 'inference', 'kernel.py')
    if not osp.exists(kernel_path):
        raise FileNotFoundError(f'Can not find DeepSeek-V4 kernel.py at {kernel_path}.')

    mod_name = f'lmdeploy_deepseek_v4_kernel_{abs(hash(kernel_path))}'
    spec = importlib.util.spec_from_file_location(mod_name, kernel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _KERNEL_MODULE_CACHE[model_path] = module
    return module


def _get_world_size_rank():
    return get_tp_world_rank('attn')


@contextmanager
def _set_default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

def _load_vector_shard(param: nn.Parameter, loaded_weight: torch.Tensor, world_size: int, rank: int):
    if world_size > 1:
        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
    param.copy_(loaded_weight)


def _dequantize_wo_a_shard(weight: torch.Tensor, scale: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Convert HF FP8 `wo_a` checkpoint tensors into the BF16 format used by
    inference/model.py.

    The official DeepSeek-V4 `convert.py` first shards `wo_a` along dim 0, then
    applies the block scale to recover a BF16 weight matrix for the final einsum.
    """
    if world_size > 1:
        weight = weight.chunk(world_size, dim=0)[rank].contiguous()
        scale = scale.chunk(world_size, dim=0)[rank].contiguous()
    weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
    weight = weight * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def _map_v4_expert_param_name(name: str, use_fused_experts: bool) -> tuple[str, str] | None:
    expert_match = re.search(r'\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$', name)
    if expert_match is None:
        return None
    proj = expert_match.group(2)
    suffix = expert_match.group(3)
    if use_fused_experts:
        if proj == 'w1':
            return name[:expert_match.start()] + f'.ffn.experts.ckpt_gate_up.{suffix}', 'gate'
        if proj == 'w3':
            return name[:expert_match.start()] + f'.ffn.experts.ckpt_gate_up.{suffix}', 'up'
        return name[:expert_match.start()] + f'.ffn.experts.ckpt_down.{suffix}', 'down'
    if proj == 'w1':
        return name, 'gate'
    if proj == 'w3':
        return name, 'up'
    return name, 'down'

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply the official DeepSeek-V4 Hadamard rotation used by the indexer."""
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1)**-0.5)


def _build_window_positions(total_lens: torch.Tensor, window_size: int):
    """Build chronologically ordered trailing window positions in ring-buffer
    coordinates padded with `-1`."""
    device = total_lens.device
    if window_size == 0:
        empty = torch.empty((total_lens.numel(), 0), dtype=torch.long, device=device)
        return empty, total_lens.new_zeros((total_lens.numel(), )), empty.bool()
    arange = torch.arange(window_size, device=device).unsqueeze(0)
    window_lens = total_lens.clamp(max=window_size)
    starts = total_lens - window_lens
    mask = arange < window_lens.unsqueeze(1)
    positions = torch.remainder(starts.unsqueeze(1) + arange, window_size)
    positions = torch.where(mask, positions, positions.new_full((), -1))
    return positions, window_lens, mask


@dataclass
class SeqInfo:
    """Bundled sequence-length metadata, computed once at the model level."""

    q_seqlens: torch.Tensor          # [bsz]
    start_pos: torch.Tensor          # [bsz], long, = kv_seqlens - q_seqlens
    cu_q_seqlens: torch.Tensor       # [bsz+1], int32, padded cumsum
    kv_seqlens: torch.Tensor         # [bsz], int32
    is_decoding: bool
    total_lens: torch.Tensor         # [bsz], = kv_seqlens

    @classmethod
    def from_metadata(cls, attn_metadata) -> 'SeqInfo':
        q_seqlens = attn_metadata.q_seqlens
        start_pos = (attn_metadata.kv_seqlens.to(torch.long) - q_seqlens.to(torch.long))
        kv_seqlens = attn_metadata.kv_seqlens.to(torch.int32)
        cu_q_seqlens = F.pad(q_seqlens.cumsum(0).to(torch.int32), (1, 0))
        is_decoding = attn_metadata.is_decoding
        total_lens = kv_seqlens
        return cls(q_seqlens=q_seqlens, start_pos=start_pos, cu_q_seqlens=cu_q_seqlens,
                   kv_seqlens=kv_seqlens, is_decoding=is_decoding, total_lens=total_lens)

    def __getitem__(self, idx) -> 'SeqInfo':
        """Slice per-sequence fields for single-seq prefill paths."""
        q_seqlens = self.q_seqlens[idx]
        start_pos = self.start_pos[idx]
        cu_q_seqlens = F.pad(q_seqlens.cumsum(0).to(torch.int32), (1, 0))
        kv_seqlens = (start_pos + q_seqlens).to(torch.int32)
        total_lens = self.total_lens[idx]
        return SeqInfo(q_seqlens=q_seqlens, start_pos=start_pos, cu_q_seqlens=cu_q_seqlens,
                       kv_seqlens=kv_seqlens, is_decoding=self.is_decoding, total_lens=total_lens)


@dataclass
class V4Args:
    dim: int
    n_heads: int
    vocab_size: int
    moe_inter_dim: int
    n_layers: int
    n_hash_layers: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    score_func: str
    route_scale: float
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    @property
    def n_groups(self) -> int:
        """Compatibility alias for code paths that expect official runtime
        naming."""
        return self.o_groups


class Compressor(nn.Module):

    def __init__(self,
                 args: V4Args,
                 layer_id: int,
                 kernel_mod,
                 compress_ratio: int,
                 head_dim: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None,
                 rotate: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.kernel_mod = kernel_mod
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32, device=device),
                                requires_grad=False)
        self.wkv = build_colwise_linear(self.dim, coff * self.head_dim, False, dtype=dtype, device=device)
        self.wgate = build_colwise_linear(self.dim, coff * self.head_dim, False, dtype=dtype, device=device)
        self.norm = RMSNorm(self.head_dim, args.norm_eps, dtype=dtype, device=device)
        self.apply_rotary = ApplyRotaryEmb()
        self.state_cache_name = self._get_state_cache_name()

    def _get_state_cache_name(self):
        if self.rotate:
            assert self.compress_ratio == 4
            return 'v4_compress_state_r4_idx'
        if self.compress_ratio == 4:
            return 'v4_compress_state_r4'
        return 'v4_compress_state_r128'

    def _get_block_cache_name(self):
        if self.rotate:
            return 'v4_index_kv_r4'
        if self.compress_ratio == 4:
            return 'v4_compressed_kv_r4'
        return 'v4_compressed_kv_r128'

    def _get_fp8_cache_name(self):
        if self.rotate:
            return None
        if self.compress_ratio == 4:
            return 'v4_compressed_kv_r4_fp8'
        if self.compress_ratio == 128:
            return 'v4_compressed_kv_r128_fp8'
        return None

    def forward(self,
                x: torch.Tensor,
                start_pos: torch.Tensor,
                slot: torch.Tensor,
                context: StepContext,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor],
                seq_info: SeqInfo):
        """Unified forward for both prefill and decode.

        Delegates ring-buffer management and scoring to the Triton kernels
        (fill_compress_state, score_kv) instead of managing state in Python.

        Args:
            x: [bsz, seqlen, dim] input tensor.
            start_pos: Tensor[bsz] start position for each sequence.
            context: StepContext with block_caches, named_state_caches, etc.
            slot: Tensor[bsz] state cache slot for each sequence.
            compress_pos_emb: (cos, sin) tuple for compressed KV RoPE.
            seq_info: Bundled sequence-length metadata.
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        dtype = x.dtype
        overlap = self.overlap
        coff = 1 + overlap
        rows = coff * ratio

        q_seqlens = seq_info.q_seqlens
        cu_q_seqlens = seq_info.cu_q_seqlens
        kv_seqlens = seq_info.kv_seqlens

        # ---- Phase A: Projections ----
        kv = self.wkv(x)       # [bsz, seqlen, D]
        score = self.wgate(x)   # [bsz, seqlen, D]

        kv_flat = kv.view(-1, kv.size(-1))
        score_flat = score.view(-1, score.size(-1))

        state_ids = slot.long().to(torch.int32)

        # ---- Phase C: Get state views ----
        state_cache = context.named_state_caches[self.state_cache_name][self.layer_id]
        kv_state = state_cache[:, :rows]
        score_state = state_cache[:, rows:2 * rows]

        # ---- Phase D: score_kv (reads state, produces compressed_kv) ----
        compressed_kv = kv_flat.new_zeros(kv_flat.size(0), self.head_dim)
        max_seqlen_q = seqlen
        score_kv(kv_flat, score_flat, self.ape, kv_state, score_state, state_ids,
                 cu_q_seqlens, kv_seqlens, compressed_kv, overlap, max_seqlen_q)

        # ---- Phase E: fill_compress_state (writes new state) ----
        fill_compress_state(kv_flat, score_flat, self.ape, kv_state, score_state, state_ids,
                            cu_q_seqlens, kv_seqlens)

        # ---- Phase F: Post-processing (norm + RoPE + quant on entire compressed_kv) ----
        compressed_kv = self.norm(compressed_kv.to(dtype))
        # Apply RoPE to compressed KV rope dims
        kv_rope = compressed_kv[..., -rd:].unsqueeze(1)  # [total_flat, 1, rd]
        cos_c, sin_c = compress_pos_emb
        self.apply_rotary.forward_single(kv_rope, cos_c, sin_c, inplace=True, complex_mode=True)
        compressed_kv[..., -rd:] = kv_rope.squeeze(1)
        if self.rotate:
            compressed_kv = rotate_activation(compressed_kv)
            self.kernel_mod.fp4_act_quant(compressed_kv, 32, True)
        else:
            self.kernel_mod.act_quant(compressed_kv[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)

        # ---- Phase G: Write to paged block cache via fill_compressed_kv ----
        block_caches = context.block_caches
        block_offsets = context.block_offsets
        block_size = context.cache_config.block_size
        cache_name = self._get_block_cache_name()
        fp8_cache_name = self._get_fp8_cache_name()
        bf16_cache = block_caches[cache_name][self.layer_id] if cache_name in block_caches else None
        fp8_cache = block_caches[fp8_cache_name][self.layer_id] if fp8_cache_name else None

        fill_compressed_kv(compressed_kv, bf16_cache, cu_q_seqlens, kv_seqlens, block_offsets,
                           self.compress_ratio, block_size, max_seqlen_q,
                           fp8_cache=fp8_cache)


class Indexer(nn.Module):

    def __init__(self,
                 config,
                 args: V4Args,
                 layer_id: int,
                 kernel_mod,
                 compress_ratio: int,
                 world_size: int,
                 rank: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.layer_id = layer_id
        self.kernel_mod = kernel_mod
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.world_size = world_size
        self.rank = rank

        quantization_config = getattr(config, 'quantization_config', None)
        self.wq_b = build_colwise_linear(
            args.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config
        )
        self.weights_proj = build_colwise_linear(
            args.dim,
            self.n_heads,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.compressor = Compressor(args, layer_id, kernel_mod, compress_ratio, self.head_dim,
                                     dtype=dtype, device=device, rotate=True)
        self.indexer_fwd = NativeV4Indexer(index_topk=self.index_topk,
                                           compress_ratio=self.compress_ratio,
                                           world_size=world_size)
        self.apply_rotary = ApplyRotaryEmb()

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                start_pos: torch.Tensor,
                offset: int,
                context: StepContext,
                slot: torch.Tensor,
                index_kv_cache: torch.Tensor,
                block_offsets: torch.Tensor,
                block_size: int,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor],
                seq_info: SeqInfo,
                is_decoding: bool = False,
                index_scratch: torch.Tensor | None = None):
        rd = self.rope_head_dim
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        cos, sin, _ = rotary_pos_emb
        self.apply_rotary.forward_single(q[..., -rd:], cos, sin, inplace=True, complex_mode=True)
        q = rotate_activation(q)
        self.kernel_mod.fp4_act_quant(q, 32, True)

        comp_start_pos = seq_info.start_pos if is_decoding else start_pos
        self.compressor(x, comp_start_pos, slot, context, compress_pos_emb, seq_info)
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)

        meta = V4IndexerMetadata(block_offsets=block_offsets,
                                 start_pos=comp_start_pos,
                                 state_ids=comp_start_pos.new_zeros(comp_start_pos.shape),
                                 compress_ratio=self.compress_ratio)
        return self.indexer_fwd(q, weights, index_kv_cache, meta, block_size, self.layer_id,
                                index_scratch, offset, is_decoding)

class DeepseekV4BMM(nn.Module):
    """Wrapped bmm."""

    def __init__(self, batch: int, in_features: int, out_features: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        batch = self._update_batch(batch)

        weight = self.create_weight(batch, in_features, out_features, dtype=dtype, device=device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_parameter('weight', weight)
        weight.weight_loader = self.weight_loader

        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device

    def _update_batch(self, batch: int):
        """Update out features."""
        world_size, _ = get_tp_world_rank('attn')
        batch = batch // world_size
        return batch

    def create_weight(self, batch: int, in_features: int, out_features: int, dtype: torch.dtype, device: torch.device):
        """Create weight."""
        return torch.empty((batch, in_features, out_features), dtype=dtype, device=device)

    def weight_loader(self, param: nn.Parameter, weight: torch.Tensor):
        """Weight loader."""
        world_size, rank = get_tp_world_rank('attn')
        weight = weight.chunk(world_size, 0)[rank]
        param.data.copy_(weight)

    def forward(self, x: torch.Tensor, output: torch.Tensor):
        """forward."""
        torch.bmm(x.transpose(0, 1), self.weight, out=output.transpose(0, 1))


class Attention(nn.Module):

    def __init__(self,
                 config,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        world_size, rank = _get_world_size_rank()
        self.kernel_mod = kernel_mod
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.o_lora_rank = args.o_lora_rank

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32, device=device),
                                      requires_grad=False)
        self.attn_sink.weight_loader = self._attn_sink_loader

        self.wq_a = build_colwise_linear(
            args.dim, args.q_lora_rank,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            )
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps, device=device)
        self.wq_b = build_colwise_linear(
            args.q_lora_rank,
            args.n_heads * args.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
            dp_disable_tp=True,
        )
        self.wkv = build_colwise_linear(
            args.dim, self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )
        self.kv_norm = RMSNorm(args.head_dim, self.eps, device=device)
        self.wo_a = DeepseekV4BMM(
            self.n_groups,
            args.n_heads * args.head_dim // self.n_groups,
            args.o_lora_rank,
            dtype=dtype,
            device=device)
        self.wo_b = build_o_proj(
            args.n_groups * args.o_lora_rank,
            args.dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.attn_fwd = NativeV4Attention(head_size=self.head_dim,
                                          scale=self.softmax_scale,
                                          window_size=self.window_size,
                                          compress_ratio=self.compress_ratio)
        self.compressor = None
        self.indexer = None
        self.apply_rotary = ApplyRotaryEmb()
        if self.compress_ratio:
            self.compressor = Compressor(args, self.layer_id, kernel_mod, self.compress_ratio, self.head_dim,
                                         dtype=dtype, device=device)
            if self.compress_ratio == 4:
                world_size, rank = get_tp_world_rank('attn')
                self.indexer = Indexer(config, args, self.layer_id, kernel_mod, self.compress_ratio,
                                       world_size, rank, dtype=dtype, device=device)

    def _attn_sink_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        world_size, rank = get_tp_world_rank('attn')
        _load_vector_shard(param, loaded_weight, world_size, rank)

    def _resolve_attention_caches(self, context: StepContext) -> dict:
        named_state_caches = context.named_state_caches
        block_caches = context.block_caches

        window_state = named_state_caches['v4_window_kv'][self.layer_id]
        window_state_fp8 = named_state_caches['v4_window_kv_fp8'][self.layer_id]

        compressed_kv = None
        compressed_kv_fp8 = None
        index_kv = None
        if self.compress_ratio:
            if self.compress_ratio == 4:
                compressed_kv_fp8 = block_caches['v4_compressed_kv_r4_fp8'][self.layer_id]
                index_kv = block_caches['v4_index_kv_r4']
            else:
                compressed_kv_fp8 = block_caches['v4_compressed_kv_r128_fp8'][self.layer_id]

        return dict(window_state=window_state,
                    window_state_fp8=window_state_fp8,
                    compressed_kv=compressed_kv,
                    compressed_kv_fp8=compressed_kv_fp8,
                    index_kv=index_kv)

    @staticmethod
    def _write_window_state_prefill_batched(window_state_cache: torch.Tensor,
                                             kv_flat: torch.Tensor,
                                             start_pos: torch.Tensor,
                                             q_seqlens: torch.Tensor,
                                             slot: torch.Tensor,
                                             window_size: int):
        """Batched ring-buffer write for all prefill sequences.

        Uses advanced indexing to scatter kv_flat into window_state_cache
        at the correct (slot, ring_position) indices.
        """
        num_seqs = start_pos.numel()
        cu_q = torch.cat([start_pos.new_zeros(1), q_seqlens.cumsum(0)])
        total_tokens = kv_flat.size(0)

        # Build per-token indices
        # token_slot[i] = which slot token i writes to
        # token_pos_in_seq[i] = position of token i within its sequence
        token_slot = slot.repeat_interleave(q_seqlens.long())  # [total_tokens]
        token_seq = torch.arange(num_seqs, device=slot.device).repeat_interleave(q_seqlens.long())
        token_pos_in_seq = torch.arange(total_tokens, device=slot.device) - cu_q[token_seq]

        # Compute the absolute position: start_pos + pos_in_seq
        token_start = start_pos.repeat_interleave(q_seqlens.long())
        token_abs_pos = token_start + token_pos_in_seq

        # For overflow (seqlen > window_size): only the last window_size tokens are kept
        # Skip tokens where abs_pos < (total_len - window_size)
        total_lens = start_pos + q_seqlens
        token_total = total_lens[token_seq]
        cutoff_pos = (token_total - window_size).clamp(min=0)
        valid = token_abs_pos >= cutoff_pos

        # Ring-buffer position
        ring_pos = torch.remainder(token_abs_pos, window_size)

        # Scatter write
        valid_slot = token_slot[valid].long()
        valid_ring = ring_pos[valid].long()
        valid_kv = kv_flat[valid]
        window_state_cache[valid_slot, valid_ring] = valid_kv

    def _pack_window_state_batched(self, window_state_cache: torch.Tensor,
                                    window_state_fp8_cache: torch.Tensor,
                                    slot: torch.Tensor,
                                    block_size: int):
        """Batched FP8 pack for all prefill sequences' window states."""
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

        # Gather all slots' window state
        selected = window_state_cache[slot.long()]  # [num_seqs, window_size, head_dim]

        # Use the Triton kernel to pack all tokens at once
        # slot values must be the real slot indices into window_state_fp8_cache
        num_seqs = slot.numel()
        slot_expanded = slot.long().repeat_interleave(self.window_size)
        pos_expanded = torch.arange(self.window_size, device=slot.device).repeat(num_seqs).long()
        kv_tokens = selected.reshape(-1, self.head_dim)
        pack_window_tokens_fp8(kv_tokens, window_state_fp8_cache, slot_expanded, pos_expanded)

    def _build_decode_attention_metadata(self,
                                         indices_in_kvcache: torch.Tensor = None,
                                         topk_length: torch.Tensor = None,
                                         extra_indices_in_kvcache: torch.Tensor = None,
                                         extra_topk_length: torch.Tensor = None):
        return V4AttentionMetadata(is_decoding=True,
                                   indices_in_kvcache=indices_in_kvcache,
                                   topk_length=topk_length,
                                   extra_indices_in_kvcache=extra_indices_in_kvcache,
                                   extra_topk_length=extra_topk_length)

    def forward(self,
                x: torch.Tensor,
                seq_info: SeqInfo,
                slot: torch.Tensor,
                context: StepContext,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None):
        is_decoding = seq_info.is_decoding
        rd = self.rope_head_dim
        bsz = seq_info.start_pos.numel()

        if is_decoding:
            x = x.transpose(0, 1)

        # ---- Batched projections (work on full [bsz, seqlen, dim] or [1, total_tokens, dim]) ----
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        cos, sin, neg_sin = rotary_pos_emb
        q_rope = q[..., -rd:]  # [bsz, seq, n_heads, rd]
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = kv[..., -rd:]  # [bsz, seq, rd]
        # Triton kernel expects 3D (seq, heads, dim) — flatten batch+seq
        n_tokens = q_rope.shape[0] * q_rope.shape[1]
        n_heads = q_rope.shape[2]
        q_rope_3d = q_rope.reshape(n_tokens, n_heads, rd)
        kv_rope_3d = kv_rope.reshape(n_tokens, 1, rd)
        q_rope_3d, kv_rope_3d = self.apply_rotary(q_rope_3d, kv_rope_3d, cos, sin, inplace=False,
                                                    complex_mode=True)
        q[..., -rd:] = q_rope_3d.reshape_as(q_rope)
        kv[..., -rd:] = kv_rope_3d.reshape_as(kv_rope)
        self.kernel_mod.act_quant(kv[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)

        # ---- Batched Compressor call ----
        if self.compress_ratio:
            self.compressor(x, seq_info.start_pos.long(), slot.long(), context, compress_pos_emb, seq_info)

        if is_decoding:
            out = self._forward_decode_core(q, kv, qr, x, seq_info, slot, context, rotary_pos_emb, compress_pos_emb, bsz)
        else:
            out = self._forward_prefill_core(q, kv, qr, x, seq_info, slot, context, rotary_pos_emb, compress_pos_emb, bsz)

        # ---- Output projection (inverse RoPE via precomputed neg_sin) ----
        self.apply_rotary.forward_single(out[..., -rd:], cos, neg_sin, inplace=True,
                                         complex_mode=True)
        total_tokens = out.size(0) * out.size(1)
        out = out.view(total_tokens, self.n_local_groups, -1)
        proj_in = out
        out = proj_in.new_empty(*proj_in.shape[:-1], self.o_lora_rank)
        self.wo_a(proj_in, out)
        if is_decoding:
            # bsz is updated, we need to reconstruct the output shape here
            return self.wo_b(out.flatten(-2, -1).view(1, total_tokens, -1))
        else:
            return self.wo_b(out.flatten(-2, -1).view(1, total_tokens, -1))

    def _forward_decode_core(self, q, kv, qr, x, seq_info: SeqInfo, slot, context, rotary_pos_emb, compress_pos_emb, bsz):
        start_pos = seq_info.start_pos
        total_lens = seq_info.total_lens

        caches = self._resolve_attention_caches(context)
        block_offsets = context.block_offsets.long()
        token_block_size = context.cache_config.block_size

        window_pos = torch.remainder(start_pos, self.window_size).long()
        slot_idx = slot.long()
        caches['window_state'][slot_idx, window_pos] = kv[:, 0]
        # Update FP8 window cache for the current decode token
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        pack_window_tokens_fp8(kv[:, 0], caches['window_state_fp8'], slot_idx, window_pos)
        window_state = caches['window_state'].index_select(0, slot_idx)
        window_state_fp8 = caches['window_state_fp8'].index_select(0, slot_idx)

        # Window positions (shared by all ratio paths)
        window_positions, window_lens, _ = _build_window_positions(total_lens.long(), self.window_size)
        extra_indices_in_kvcache = window_positions.unsqueeze(1).to(torch.int32)  # [bsz, 1, window_size]
        extra_topk_length = window_lens.to(torch.int32)                           # [bsz]
        offset = self.window_size

        # Build compressed indices based on compress_ratio
        compressed_cache_fp8 = None
        indices_in_kvcache = None
        topk_length = None

        if self.compress_ratio:
            compressed_cache_fp8 = caches['compressed_kv_fp8']

            if self.indexer is not None:
                # ratio=4: Indexer provides physical indices into fp8 compressed cache
                index_cache = caches['index_kv']
                index_out = self.indexer(x=x,
                                         qr=qr,
                                         start_pos=start_pos.long(),
                                         offset=offset,
                                         context=context,
                                         slot=slot,
                                         index_kv_cache=index_cache,
                                         block_offsets=block_offsets,
                                         block_size=token_block_size,
                                         rotary_pos_emb=rotary_pos_emb,
                                         compress_pos_emb=compress_pos_emb,
                                         seq_info=seq_info,
                                         index_scratch=None,
                                         is_decoding=True)
                indices_in_kvcache = index_out.indices_in_kvcache
                topk_length = index_out.topk_length
            else:
                # ratio=128: logical-to-physical index conversion for FlashMLA sparse path
                num_compressed = torch.div(total_lens, self.compress_ratio, rounding_mode='floor').long()
                max_comp = max(block_offsets.size(1) * token_block_size // self.compress_ratio, 1)
                comp_positions, _ = build_prefix_positions(num_compressed, max_comp)
                entries_per_block = compressed_cache_fp8.size(1)
                valid = comp_positions >= 0
                safe_pos = comp_positions.clamp(min=0)
                token_positions = safe_pos * self.compress_ratio
                block_idx = torch.div(token_positions, token_block_size, rounding_mode='floor').long()
                max_block_idx = block_offsets.size(1)
                safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
                phys_blocks = block_offsets.gather(1, safe_block_idx).long()
                block_off = torch.remainder(safe_pos, entries_per_block).long()
                phys_indices = phys_blocks * entries_per_block + block_off
                indices_in_kvcache = torch.where(valid, phys_indices,
                                                 phys_indices.new_full((), -1)).unsqueeze(1).to(torch.int32)
                topk_length = num_compressed.to(torch.int32)
        else:
            # ratio=0: No compressed KV. Use window as k_cache (topk_length=0 → not read).
            indices_in_kvcache = torch.full((bsz, 1, 1), -1, dtype=torch.int32, device=q.device)
            topk_length = torch.zeros(bsz, dtype=torch.int32, device=q.device)

        attn_meta = self._build_decode_attention_metadata(
            indices_in_kvcache=indices_in_kvcache,
            topk_length=topk_length,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length)
        return self.attn_fwd.forward_decode(q,
                                            window_state_fp8,
                                            self.attn_sink,
                                            attn_meta,
                                            token_block_size,
                                            compressed_kv_fp8_cache=compressed_cache_fp8)

    def _forward_prefill_core(self, q, kv, qr, x, seq_info: SeqInfo, slot, context, rotary_pos_emb, compress_pos_emb, num_seqs):
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        start_pos = seq_info.start_pos
        q_seqlens = seq_info.q_seqlens
        total_lens = seq_info.total_lens

        caches = self._resolve_attention_caches(context)
        block_offsets = context.block_offsets
        window_block_size = context.cache_config.kernel_block_size
        token_block_size = context.cache_config.block_size

        # CPU-side upper bounds for flatten_v4_kv (avoids GPU .item() sync)
        max_kv = context.max_kv_seqlen
        sum_kv = context.sum_kv_seqlen
        cr = self.compress_ratio if self.compress_ratio else 1
        max_flat_kv_len = min(max_kv, self.window_size) + max_kv // cr
        total_flat_kv_tokens = sum_kv + sum_kv // cr
        # CPU-side upper bound for build_compress_topk_indices
        max_compress_width = max_kv // cr

        # Pre-compute window_kv_lens for Indexer offset
        window_kv_lens = total_lens.clamp(max=self.window_size)

        # Prefill: x is [1, total_tokens, ...], flatten to [total_tokens, ...]
        q_flat = q.squeeze(0)    # [total_tokens, n_heads, head_dim]
        kv_flat = kv.squeeze(0)  # [total_tokens, head_dim]
        qr_flat = qr.squeeze(0)  # [total_tokens, q_lora_rank]
        x_flat = x.squeeze(0)    # [total_tokens, dim]

        # ---- Phase 1: Write window state (batched) ----
        self._write_window_state_prefill_batched(
            caches['window_state'], kv_flat, start_pos, q_seqlens, slot, self.window_size)
        self._pack_window_state_batched(
            caches['window_state'], caches['window_state_fp8'], slot, window_block_size)

        # ---- Phase 2: Per-seq Indexer call (writes compressed KV) ----
        # Cannot be batched because: (1) the internal Compressor writes ring-buffer
        # state and paged block cache as side effects — zero-padded input would
        # corrupt both; (2) the Indexer backend derives total_lens from seqlen,
        # so padded seqlen produces wrong compression points for shorter sequences.
        compress_topk = None
        if self.compress_ratio:
            if self.indexer is not None:
                cu_q = torch.cat([q_seqlens.new_zeros(1), q_seqlens.cumsum(0)])
                parts = []
                for s in range(num_seqs):
                    sl = q_seqlens[s].item()
                    off = cu_q[s].item()
                    seq_x = x[:, off:off + sl]
                    seq_qr = qr[:, off:off + sl]
                    cos_r, sin_r, neg_sin_r = rotary_pos_emb
                    seq_rotary_pos_emb = (cos_r[off:off + sl], sin_r[off:off + sl], neg_sin_r[off:off + sl])
                    cos_c, sin_c = compress_pos_emb
                    seq_compress_pos_emb = (cos_c[off:off + sl], sin_c[off:off + sl])
                    seq_info_s = seq_info[s:s + 1]

                    index_out = self.indexer(
                        x=seq_x,
                        qr=seq_qr,
                        start_pos=start_pos[s:s + 1],
                        offset=window_kv_lens[s:s + 1],
                        context=context,
                        slot=slot[s:s + 1],
                        index_kv_cache=caches['index_kv'],
                        block_offsets=block_offsets[s:s + 1],
                        block_size=token_block_size,
                        rotary_pos_emb=seq_rotary_pos_emb,
                        compress_pos_emb=seq_compress_pos_emb,
                        seq_info=seq_info_s,
                        index_scratch=None,
                        is_decoding=False)

                    raw = index_out.indices_in_kvcache  # [1, sl, topk]
                    parts.append(raw.squeeze(0))  # [sl, topk]

                # Pad to   same topk width before concatenating
                max_topk = max(p.size(1) for p in parts)
                parts = [
                    F.pad(p, (0, max_topk - p.size(1)), value=-1) if p.size(1) < max_topk else p
                    for p in parts
                ]
                compress_topk = torch.cat(parts, dim=0)
            else:
                compress_topk = build_compress_topk_indices(
                    total_lens, self.compress_ratio,
                    offset=window_kv_lens,
                    q_seqlens=q_seqlens,
                    start_pos=start_pos,
                    causal=True,
                    max_width=max_compress_width)

        # ---- Phase 3: Flatten window + compressed KV into contiguous tensor ----
        # BF16 compressed KV caches are eliminated; read from FP8 instead
        compressed_kv_cache = None
        fp8_compressed_kv_cache = caches['compressed_kv_fp8'] if self.compress_ratio else None
        # Select only the window states for sequences in the current batch
        batch_window_kv = caches['window_state'].index_select(0, slot.long())
        flat_kv, cu_seqlens_k = flatten_v4_kv(
            batch_window_kv,
            compressed_kv_cache,
            block_offsets.long(),
            total_lens.long(),
            self.window_size,
            self.compress_ratio,
            total_flat_kv_tokens,
            max_flat_kv_len,
            fp8_compressed_kv_cache=fp8_compressed_kv_cache)

        # ---- Phase 4: Build topk indices and convert to global ----
        window_topk = build_window_topk_indices(
            total_lens, self.window_size,
            q_seqlens=q_seqlens,
            start_pos=start_pos,
            causal=True)  # [total_q_tokens, window_size]

        if compress_topk is not None:
            topk_indices = torch.cat([window_topk, compress_topk], dim=-1)  # [total_q_tokens, total_topk]
        else:
            topk_indices = window_topk  # [total_q_tokens, window_size]

        # Convert per-seq-local indices to global flat indices
        # cu_seqlens_k[:-1] gives per-seq KV start offset in the flat tensor
        kv_start_per_token = torch.repeat_interleave(
            cu_seqlens_k[:-1], q_seqlens.to(torch.long))
        neg_mask = topk_indices < 0
        topk_indices = topk_indices + kv_start_per_token.unsqueeze(1)
        topk_indices[neg_mask] = -1
        topk_indices = topk_indices.unsqueeze(1).to(torch.int32)  # [total_q_tokens, 1, total_topk]

        # ---- Phase 5: Call flash_mla_sparse_fwd ----
        out = self.attn_fwd.forward_prefill(q_flat, flat_kv, self.attn_sink, topk_indices)
        # out is [total_q_tokens, n_heads, head_dim] -> reshape to [1, total_q_tokens, n_heads, head_dim]
        return out.unsqueeze(0)


class Gate(nn.Module):

    def __init__(self, layer_id: int, args: V4Args, device: torch.device | str | None):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, device=device), requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size,
                                                    args.n_activated_experts,
                                                    dtype=torch.int32,
                                                    device=device),
                                        requires_grad=False)
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32, device=device),
                                     requires_grad=False)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == 'softmax':
            scores = scores.softmax(dim=-1)
        elif self.score_func == 'sigmoid':
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != 'softmax':
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):

    def __init__(self, config, dim: int, inter_dim: int, kernel_mod, dtype=None, swiglu_limit=0.0, device=None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.w13 = build_gateup_linear(dim, [inter_dim, inter_dim], bias=False, dtype=dtype, device=device,
                                      quant_config=quantization_config,
                                      is_tp=False)
        self.w2 = build_down_linear(inter_dim, dim, bias=False, quant_config=quantization_config,
                                      is_tp=False, dtype=dtype, device=device)
        self.swiglu_limit = swiglu_limit
        self.act_fn = SiluAndMul(inplace=True)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
        dtype = x.dtype
        gate_up = self.w13(x)
        if self.swiglu_limit > 0:
            gate, up = gate_up.chunk(2, dim=-1)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
            x = F.silu(gate) * up
        else:
            x = self.act_fn(gate_up)

        if weights is not None:
            x = weights * x.float()
            x = x.to(dtype)
        return self.w2(x)


class MoE(nn.Module):

    def __init__(self,
                 config,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args, device=device)
        self.experts = FusedMoEV4(args.dim,
                                    args.moe_inter_dim,
                                    args.n_routed_experts,
                                    args.n_activated_experts,
                                    swiglu_limit=args.swiglu_limit,
                                    device=device)
        self.shared_experts = Expert(
            config,
            args.dim,
            args.moe_inter_dim,
            kernel_mod,
            dtype=dtype,
            swiglu_limit=0.0,
            device=device)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = self.experts(x, weights, indices)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):

    def __init__(self,
                 config,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.norm_eps = args.norm_eps
        self.attn = Attention(config, layer_id, args, kernel_mod, dtype=dtype, device=device)
        self.ffn = MoE(config, layer_id, args, kernel_mod, dtype=dtype, device=device)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, dtype=dtype, device=device)
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
        with _set_default_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, device=device), requires_grad=False)
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, device=device), requires_grad=False)
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, device=device), requires_grad=False)
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, device=device), requires_grad=False)
            self.hc_attn_scale = nn.Parameter(torch.empty(3, device=device), requires_grad=False)
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, device=device), requires_grad=False)
        self.kernel_mod = kernel_mod

    def _hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = self.kernel_mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult,
                                                            self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, seq_info: SeqInfo, input_ids: torch.Tensor,
                slot: torch.Tensor, context: StepContext,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, seq_info, slot, context, rotary_pos_emb=rotary_pos_emb,
                       compress_pos_emb=compress_pos_emb)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x


class DeepseekV4ForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """DeepSeek-V4 model.

    Decode uses FlashMLA sparse decode with FP8 window + compressed KV caches.
    """

    def __init__(self,
                 config,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.model_path = getattr(config, '_name_or_path', None) or getattr(config, 'name_or_path', None)
        if self.model_path is None:
            raise RuntimeError('DeepSeek-V4 requires config._name_or_path to load official kernels.')
        self.kernel_mod = _load_v4_kernel_module(self.model_path)
        self.world_size, self.rank = _get_world_size_rank()
        self.dtype = dtype or torch.bfloat16
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.args = V4Args(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_layers=config.num_hidden_layers,
            n_hash_layers=config.num_hash_layers,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            score_func=config.scoring_func,
            route_scale=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            q_lora_rank=config.q_lora_rank,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            norm_eps=config.rms_norm_eps,
            o_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            window_size=config.sliding_window,
            compress_ratios=tuple(config.compress_ratios),
            compress_rope_theta=config.compress_rope_theta,
            original_seq_len=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_factor=config.rope_scaling['factor'],
            beta_fast=config.rope_scaling['beta_fast'],
            beta_slow=config.rope_scaling['beta_slow'],
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
        )
        args = self.args
        # Plain RoPE: Default type (no YaRN correction), base=rope_theta
        self.rotary_emb_plain = build_rotary_embedding(
            dim=args.rope_head_dim,
            max_position_embeddings=args.original_seq_len,
            base=args.rope_theta,
            emb_type=RopeType.Default,
            device=device,
        )
        # Compress RoPE: YaRN type with compress_rope_theta base
        # attention_factor=1.0 disables mscale scaling (official V4 inference
        # uses YaRN frequency interpolation but does not apply mscale)
        yarn_params = YarnParameters(beta_fast=args.beta_fast, beta_slow=args.beta_slow, attention_factor=1.0)
        self.rotary_emb_compress = build_rotary_embedding(
            dim=args.rope_head_dim,
            max_position_embeddings=args.original_seq_len,
            base=args.compress_rope_theta,
            scaling_factor=args.rope_factor,
            emb_type=RopeType.Yarn,
            yarn_params=yarn_params,
            device=device,
        )
        self.embed = build_embedding(config.vocab_size,
                                       config.hidden_size,
                                       None,
                                       device=self.device,
                                       dtype=self.dtype,
                                       is_tp=True,)
        self.layers = nn.ModuleList([
            Block(config, layer_idx, self.args, self.kernel_mod, dtype=self.dtype, device=self.device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=self.dtype, device=self.device)
        self.head = self.build_lm_head(
                            config.hidden_size,
                            config.vocab_size,
                            bias=False,
                            dtype=dtype,
                            device=self.device)
        hc_dim = config.hc_mult * config.hidden_size
        with _set_default_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, device=self.device),
                                           requires_grad=False)
            self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, device=self.device), requires_grad=False)
            self.hc_head_scale = nn.Parameter(torch.empty(1, device=self.device), requires_grad=False)

        # buffer to load weight
        self._load_buffers = dict()

    def _hc_head(self, x: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def get_logits(self, hidden_states: torch.Tensor):
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.head(hidden_states)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list | None = None,
                attn_metadata=None,
                inputs_embeds: torch.Tensor | None = None,
                state_ids: torch.Tensor | None = None,
                **kwargs):
        if state_ids is None:
            raise RuntimeError('DeepSeek-V4 requires state_ids to provide stable cache slots.')

        seq_info = SeqInfo.from_metadata(attn_metadata)
        context = self.ctx_mgr.current_context()

        safe_state_ids = state_ids.to(torch.long)

        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

        # Compute rotary (cos, sin, neg_sin) from position_ids (outside the layer loop)
        # V4 uses complex-number RoPE: cos/sin are (seq_len, rd//2), not duplicated
        # neg_sin is precomputed once to avoid per-layer -sin allocation
        rd = self.args.rope_head_dim
        cos_plain, sin_plain = self.rotary_emb_plain(h, position_ids)
        cos_plain = cos_plain[0, :, :rd // 2]
        sin_plain = sin_plain[0, :, :rd // 2]
        rotary_pos_emb_plain = (cos_plain, sin_plain, -sin_plain)

        cos_compress, sin_compress = self.rotary_emb_compress(h, position_ids)
        cos_compress = cos_compress[0, :, :rd // 2]
        sin_compress = sin_compress[0, :, :rd // 2]
        rotary_pos_emb_compress = (cos_compress, sin_compress, -sin_compress)

        compress_pos_emb = {}
        for ratio in (4, 128):
            cidx = (position_ids + 1 - ratio).clamp(min=0)
            cos_c, sin_c = self.rotary_emb_compress(h, cidx)
            compress_pos_emb[ratio] = (cos_c[0, :, :rd // 2], sin_c[0, :, :rd // 2])

        # Single layer loop — no tensor indexing inside
        for layer in self.layers:
            if layer.attn.compress_ratio:
                rot_emb, comp_emb = rotary_pos_emb_compress, compress_pos_emb[layer.attn.compress_ratio]
            else:
                rot_emb, comp_emb = rotary_pos_emb_plain, None
            h = layer(h, seq_info, input_ids, safe_state_ids, context,
                      rotary_pos_emb=rot_emb, compress_pos_emb=comp_emb)

        return h

    def prepare_inputs_for_generation(self,
                                      past_key_values: list[list[torch.Tensor]],
                                      inputs_embeds: torch.Tensor | None = None,
                                      context: StepContext = None):
        return dict(
            input_ids=context.input_ids,
            position_ids=context.position_ids,
            past_key_values=past_key_values,
            attn_metadata=context.attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
        )

    def _load_weights_attn(self, name: str, weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        def _maybe_load_wo_a(weight_name, scale_name):
            if weight_name not in self._load_buffers or scale_name not in self._load_buffers: # type: ignore
                return
            dequantized = _dequantize_wo_a_shard(self._load_buffers.pop(weight_name),
                                                 self._load_buffers.pop(scale_name), 1, 0)
            o_groups = self.config.o_groups
            dequantized = dequantized.view(o_groups, -1, dequantized.size(-1))
            dequantized = dequantized.transpose(-1, -2)
            param = params_dict[weight_name]
            load_weight(param, dequantized)

        if '.wo_a' in name:
            if name.endswith('.weight'):
                weight_name = name
                scale_name = name.replace('.weight', '.scale')
                self._load_buffers[name] = weight
            elif name.endswith('.scale'):
                scale_name = name
                weight_name = name.replace('.scale', '.weight')
                self._load_buffers[name] = weight
            else:
                raise RuntimeError(f'Unexpected wo_a param name: {name}')
            _maybe_load_wo_a(weight_name, scale_name)
            return
        elif '.scale' in name:
            name = name.replace('.scale', '.weight_scale_inv')
        load_weight(params_dict[name], weight)

    def _load_expert(self, name: str, weight: torch.Tensor, params_dict: dict[str, nn.Parameter]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.w13', '.w1', 0),
            ('.w13', '.w3', 1),
        ]
        if '.shared_experts.' in name:
            if name.endswith('.scale'):
                name = name.replace('.scale', '.weight_scale_inv')
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                load_weight(param, weight, shard_id=shard_id)
                break
            else:
                load_weight(params_dict[name], weight)
            return

        # load other
        loaded_weight = weight
        expert_match = re.search(r'\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$', name)
        if expert_match is not None:
            expert_id = int(expert_match.group(1))
            mapped = _map_v4_expert_param_name(name, True)
            assert mapped is not None
            param_name, shard_id = mapped
            if param_name in params_dict:
                load_weight(params_dict[param_name], loaded_weight, expert_id=expert_id, shard_id=shard_id)
        else:
            load_weight(params_dict[name], weight)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.startswith('mtp.'):
                continue

            if name.endswith('tie2eid'):
                name = name.replace('tie2eid', 'tid2eid')
            if '.ffn.' in name:
                self._load_expert(name, loaded_weight, params_dict)
                continue
            if '.attn.' in name:
                self._load_weights_attn(name, loaded_weight, params_dict)
                continue
            if name not in params_dict:
                logger.warning(f'Skip unknown DeepSeek-V4 weight: {name}')
                continue
            load_weight(params_dict[name], loaded_weight)
