# Copyright (c) OpenMMLab. All rights reserved.
import re
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.backends.compressor import V4CompressorMetadata
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata
from lmdeploy.pytorch.backends.rotary_embedding import RopeType, YarnParameters
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, HcSplitSinkhorn, RMSNorm, SiluAndMul
from lmdeploy.pytorch.nn import V4Attention as NativeV4Attention
from lmdeploy.pytorch.nn import V4Compressor as NativeV4Compressor
from lmdeploy.pytorch.nn import V4Indexer as NativeV4Indexer
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_down_linear, build_gateup_linear, build_o_proj
from lmdeploy.pytorch.nn.moe import FusedMoEV4FP4
from lmdeploy.pytorch.nn.rotary_embedding import build_rotary_embedding
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1, build_embedding

logger = get_logger('lmdeploy')


@contextmanager
def _set_default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

def _load_vector_shard(param: nn.Parameter, loaded_weight: torch.Tensor):
    world_size, rank = get_tp_world_rank('attn')
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


def _map_v4_expert_param_name(name: str) -> tuple[str, str] | None:
    expert_match = re.search(r'\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$', name)
    if expert_match is None:
        return None
    proj = expert_match.group(2)
    suffix = expert_match.group(3)
    # Both V4ExpertTPWeights and V4ExpertWeights use gate_up/down attribute names.
    # The weight_loader on each parameter handles TP/EP sharding internally.
    if proj == 'w1':
        return name[:expert_match.start()] + f'.ffn.experts.gate_up.{suffix}', 'gate'
    if proj == 'w3':
        return name[:expert_match.start()] + f'.ffn.experts.gate_up.{suffix}', 'up'
    return name[:expert_match.start()] + f'.ffn.experts.down.{suffix}', 'down'

@dataclass

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


@dataclass
class V4Caches:
    """Cache dictionaries extracted once from StepContext, passed down to sub-
    modules."""
    named_state_caches: dict   # {name: [per_layer_tensor]}
    block_caches: dict         # {name: [per_layer_tensor]}



class Compressor(nn.Module):

    def __init__(self,
                 args: V4Args,
                 layer_id: int,
                 compress_ratio: int,
                 head_dim: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None,
                 rotate: bool = False):
        super().__init__()
        self.layer_id = layer_id
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
        self.compressor_impl = NativeV4Compressor(
            compress_ratio=compress_ratio,
            overlap=self.overlap,
            head_dim=head_dim)
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
                slot: torch.Tensor,
                caches: V4Caches,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor],
                v4_compressor_meta: V4CompressorMetadata = None):
        """Unified forward for both prefill and decode.

        Delegates ring-buffer management and scoring to the backend-dispatched
        V4Compressor impl instead of calling kernels directly.

        Args:
            x: [bsz, seqlen, dim] for decode, [1, total_tokens, dim] for prefill.
            slot: Tensor[bsz] state cache slot for each sequence.
            caches: Cache dictionaries (named_state_caches + block_caches).
            compress_pos_emb: (cos, sin) tuple for compressed KV RoPE.
            v4_compressor_meta: Pre-built compressor metadata.
        """
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        overlap = self.overlap
        coff = 1 + overlap
        rows = coff * ratio

        # ---- Phase A: Projections ----
        kv = self.wkv(x)       # same layout as x
        score = self.wgate(x)   # same layout as x

        kv_flat = kv.view(-1, kv.size(-1))
        score_flat = score.view(-1, score.size(-1))

        state_ids = slot

        # ---- Phase C: Get state views ----
        state_cache = caches.named_state_caches[self.state_cache_name][self.layer_id]
        kv_state = state_cache[:, :rows]
        score_state = state_cache[:, rows:2 * rows]

        # ---- Phase D+E: score + fill state (via backend dispatch) ----
        compressed_kv = self.compressor_impl.score_and_fill_state(
            kv_flat, score_flat, self.ape, kv_state, score_state, state_ids, v4_compressor_meta)

        # ---- Phase F: Post-processing (norm + RoPE + quant on entire compressed_kv) ----
        compressed_kv = self.norm(compressed_kv)
        # Apply RoPE to compressed KV rope dims
        kv_rope = compressed_kv[..., -rd:].unsqueeze(1)  # [total_flat, 1, rd]
        cos_c, sin_c = compress_pos_emb
        self.apply_rotary.forward_single(kv_rope, cos_c, sin_c, inplace=True, complex_mode=True)
        if self.rotate:
            compressed_kv = self.compressor_impl.rotate_activation(compressed_kv)
        else:
            # TODO: FP8 quantize NoPE dims directly and write to FP8 cache,
            # eliminating the BF16 round-trip that act_quant(inplace=True) did
            pass

        # ---- Phase G: Write to paged block cache (via backend dispatch) ----
        block_caches = caches.block_caches
        cache_name = self._get_block_cache_name()
        fp8_cache_name = self._get_fp8_cache_name()
        kv_cache = block_caches[cache_name][self.layer_id] if cache_name in block_caches else None
        fp8_cache = block_caches[fp8_cache_name][self.layer_id] if fp8_cache_name else None
        scale_cache_name = f'{cache_name}_scale' if self.rotate else None
        if scale_cache_name and scale_cache_name in block_caches:
            kv_scale_cache = block_caches[scale_cache_name][self.layer_id]
        else:
            kv_scale_cache = None

        self.compressor_impl.write_compressed_kv(
            compressed_kv, kv_cache, v4_compressor_meta,
            fp8_cache=fp8_cache,
            kv_scale_cache=kv_scale_cache)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.compressor_impl.rotate_activation(x)


class Indexer(nn.Module):

    def __init__(self,
                 config,
                 args: V4Args,
                 layer_id: int,
                 compress_ratio: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio

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
        self.compressor = Compressor(args, layer_id, compress_ratio, self.head_dim,
                                     dtype=dtype, device=device, rotate=True)
        self.indexer_fwd = NativeV4Indexer(index_topk=self.index_topk,
                                           compress_ratio=self.compress_ratio)
        self.apply_rotary = ApplyRotaryEmb()

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                caches: V4Caches,
                slot: torch.Tensor,
                index_kv_cache: torch.Tensor,
                index_kv_scale_cache: torch.Tensor,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor],
                v4_indexer_meta: V4IndexerMetadata = None,
                v4_compressor_meta: V4CompressorMetadata = None):
        rd = self.rope_head_dim
        q = self.wq_b(qr).unflatten(-1, (-1, self.head_dim))

        cos, sin, _ = rotary_pos_emb
        self.apply_rotary.forward_single(q[..., -rd:], cos, sin, inplace=True, complex_mode=True)
        q = self.compressor.rotate_activation(q)

        self.compressor(x, slot, caches, compress_pos_emb, v4_compressor_meta=v4_compressor_meta)
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)

        return self.indexer_fwd(q, weights, index_kv_cache, index_kv_scale_cache, v4_indexer_meta)

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
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        world_size, _ = get_tp_world_rank('attn')
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
            self.compressor = Compressor(args, self.layer_id, self.compress_ratio, self.head_dim,
                                         dtype=dtype, device=device)
            if self.compress_ratio == 4:
                world_size, _ = get_tp_world_rank('attn')
                self.indexer = Indexer(config, args, self.layer_id, self.compress_ratio,
                                       dtype=dtype, device=device)

    def _attn_sink_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        _load_vector_shard(param, loaded_weight)

    def _resolve_attention_caches(self, caches: V4Caches) -> dict:
        named_state_caches = caches.named_state_caches
        block_caches = caches.block_caches

        window_state_fp8 = named_state_caches['v4_window_kv_fp8'][self.layer_id]

        compressed_kv = None
        compressed_kv_fp8 = None
        index_kv = None
        index_kv_scale = None
        if self.compress_ratio:
            if self.compress_ratio == 4:
                compressed_kv_fp8 = block_caches['v4_compressed_kv_r4_fp8'][self.layer_id]
                index_kv = block_caches['v4_index_kv_r4'][self.layer_id]
                index_kv_scale = block_caches['v4_index_kv_r4_scale'][self.layer_id]
            else:
                compressed_kv_fp8 = block_caches['v4_compressed_kv_r128_fp8'][self.layer_id]

        return dict(window_state_fp8=window_state_fp8,
                    compressed_kv=compressed_kv,
                    compressed_kv_fp8=compressed_kv_fp8,
                    index_kv=index_kv,
                    index_kv_scale=index_kv_scale)

    def forward(self,
                x: torch.Tensor,
                v4_meta: V4AttentionMetadata,
                v4_indexer_meta: V4IndexerMetadata,
                v4_compressor_meta: V4CompressorMetadata,
                slot: torch.Tensor,
                caches: V4Caches,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None):
        rd = self.rope_head_dim

        # ---- Projections + RoPE (no prefill/decode branch) ----
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        cos, sin, neg_sin = rotary_pos_emb
        q_rope = q[..., -rd:]
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = kv[..., -rd:]
        n_tokens = q_rope.shape[0] * q_rope.shape[1]
        n_heads = q_rope.shape[2]
        q_rope_3d = q_rope.reshape(n_tokens, n_heads, rd)
        kv_rope_3d = kv_rope.reshape(n_tokens, 1, rd)
        q_rope_3d, kv_rope_3d = self.apply_rotary(q_rope_3d, kv_rope_3d, cos, sin, inplace=False,
                                                    complex_mode=True)
        q[..., -rd:] = q_rope_3d.reshape_as(q_rope)
        kv[..., -rd:] = kv_rope_3d.reshape_as(kv_rope)

        # ---- Compressor call (shared) ----
        if self.compress_ratio:
            self.compressor(x, slot, caches, compress_pos_emb, v4_compressor_meta=v4_compressor_meta)

        # ---- Indexer call (model-level, result passed to backend) ----
        index_out = None
        if self.compress_ratio and self.indexer is not None:
            attn_caches = self._resolve_attention_caches(caches)
            index_out = self.indexer(x=x, qr=qr, caches=caches, slot=slot,
                                     index_kv_cache=attn_caches['index_kv'],
                                     index_kv_scale_cache=attn_caches['index_kv_scale'],
                                     rotary_pos_emb=rotary_pos_emb,
                                     compress_pos_emb=compress_pos_emb,
                                     v4_indexer_meta=v4_indexer_meta,
                                     v4_compressor_meta=v4_compressor_meta)

        # ---- Unified attention (backend dispatches decode/prefill) ----
        attn_caches = self._resolve_attention_caches(caches)
        out = self.attn_fwd(q, kv, self.attn_sink, v4_meta, attn_caches, slot,
                            index_out=index_out)

        # ---- Output projection (inverse RoPE via precomputed neg_sin) ----
        self.apply_rotary.forward_single(out[..., -rd:], cos, neg_sin, inplace=True,
                                         complex_mode=True)
        total_tokens = out.size(0) * out.size(1)
        out = out.view(total_tokens, self.n_local_groups, -1)
        proj_in = out
        out = proj_in.new_empty(*proj_in.shape[:-1], self.o_lora_rank)
        self.wo_a(proj_in, out)
        return self.wo_b(out.flatten(-2, -1).view(1, total_tokens, -1))


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

    def __init__(self, config, dim: int, inter_dim: int, dtype=None, swiglu_limit=0.0, device=None):
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
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args, device=device)
        self.experts = FusedMoEV4FP4(args.dim,
                                    args.moe_inter_dim,
                                    args.n_routed_experts,
                                    args.n_activated_experts,
                                    swiglu_limit=args.swiglu_limit,
                                    device=device)
        self.shared_experts = Expert(
            config,
            args.dim,
            args.moe_inter_dim,
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
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
        self.norm_eps = args.norm_eps
        self.layer_id = layer_id
        self.attn = Attention(config, layer_id, args, dtype=dtype, device=device)
        self.ffn = MoE(config, layer_id, args, dtype=dtype, device=device)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, dtype=dtype, device=device)
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.hc_split_sinkhorn_impl = HcSplitSinkhorn(self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
        with _set_default_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, device=device), requires_grad=False)
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, device=device), requires_grad=False)
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, device=device), requires_grad=False)
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, device=device), requires_grad=False)
            self.hc_attn_scale = nn.Parameter(torch.empty(3, device=device), requires_grad=False)
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, device=device), requires_grad=False)

    def _hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = self.hc_split_sinkhorn_impl(mixes, hc_scale, hc_base)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, v4_meta: V4AttentionMetadata,
                v4_indexer_meta: V4IndexerMetadata, v4_compressor_meta: V4CompressorMetadata,
                input_ids: torch.Tensor, slot: torch.Tensor, caches: V4Caches,
                rotary_pos_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                compress_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, v4_meta, v4_indexer_meta, v4_compressor_meta, slot, caches,
                       rotary_pos_emb=rotary_pos_emb, compress_pos_emb=compress_pos_emb)
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
            original_seq_len=config.rope_scaling['original_max_position_embeddings'],
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
            Block(config, layer_idx, self.args, dtype=self.dtype, device=self.device)
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

        context = self.ctx_mgr.current_context()

        # Build V4AttentionMetadata once from attn_metadata + step_ctx.
        # Uses backend-dispatched subclass (e.g. CudaV4AttentionMetadata) so
        # backend-specific indices are pre-computed once per step.
        safe_state_ids = state_ids.to(torch.long)
        v4_meta_cls = get_backend().get_v4_attention_metadata_cls()
        v4_meta = v4_meta_cls.from_step_context(
            attn_metadata, context,
            window_size=self.args.window_size, slot=safe_state_ids)

        # Pre-build indexer/compressor metadata once (not per-layer)
        v4_indexer_meta = V4IndexerMetadata(
            block_offsets=v4_meta.block_offsets,
            is_decoding=v4_meta.is_decoding,
            cu_q_seqlens=v4_meta.cu_q_seqlens,
            kv_seqlens=v4_meta.kv_seqlens,
            q_seqlens=v4_meta.q_seqlens,
            max_kv_seqlen=v4_meta.max_kv_seqlen,
            max_q_seqlen=v4_meta.max_q_seqlen,
            block_size=v4_meta.block_size,
        )
        v4_compressor_meta = V4CompressorMetadata(
            cu_q_seqlens=v4_meta.cu_q_seqlens,
            kv_seqlens=v4_meta.kv_seqlens,
            block_offsets=v4_meta.block_offsets,
            block_size=v4_meta.block_size,
            max_kv_seqlen=v4_meta.max_kv_seqlen,
        )

        caches = V4Caches(
            named_state_caches=context.named_state_caches,
            block_caches=context.block_caches,
        )

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
            h = layer(h, v4_meta, v4_indexer_meta, v4_compressor_meta, input_ids, safe_state_ids, caches,
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
            mapped = _map_v4_expert_param_name(name)
            assert mapped is not None
            param_name, shard_id = mapped
            if param_name in params_dict:
                load_weight(params_dict[param_name], loaded_weight, expert_id=expert_id, shard_id=shard_id)
        else:
            load_weight(params_dict[name], weight)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        def __skip_layers():
            """We might change the number of layers so we can debug the model
            with less gpus."""
            import re
            matches = re.findall(r'layers\.(\d+)\.', name)
            if not matches:
                return False
            layer_id = int(matches[0])
            return layer_id >= self.config.num_hidden_layers

        for name, loaded_weight in weights:
            if name.startswith('mtp.'):
                continue

            if __skip_layers():
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
