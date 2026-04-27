# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import math
import os.path as osp
import re
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, SiluAndMul
from lmdeploy.pytorch.nn import V4Attention as NativeV4Attention
from lmdeploy.pytorch.nn import V4Indexer as NativeV4Indexer
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_down_linear, build_gateup_linear, build_o_proj
from lmdeploy.pytorch.nn.moe import FusedMoEV4
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

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

@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_idx, max_idx, dim):
        if min_idx == max_idx:
            max_idx += 0.001
        linear = (torch.arange(dim, dtype=torch.float32) - min_idx) / (max_idx - min_idx)
        return torch.clamp(linear, 0, 1)

    freqs = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        if freqs_cis.ndim == 3:
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), x.size(-1))
        elif freqs_cis.ndim == 2 and freqs_cis.size(0) == x.size(0):
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), x.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        if freqs_cis.ndim == 4:
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
        elif freqs_cis.ndim == 3 and freqs_cis.size(0) == x.size(0):
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply the official DeepSeek-V4 Hadamard rotation used by the indexer."""
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1)**-0.5)


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device: torch.device | str):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([
            torch.arange(start_pos + 1, window_size, device=device),
            torch.arange(0, start_pos + 1, device=device)
        ],
                           dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1, device=device), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int, device: torch.device | str):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _build_prefix_positions(lengths: torch.Tensor, max_len: int):
    """Build `[0, ..., len-1]` positions padded with `-1`."""
    device = lengths.device
    if max_len == 0:
        empty = torch.empty((lengths.numel(), 0), dtype=torch.long, device=device)
        return empty, empty.bool()
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    positions = torch.where(mask, arange, arange.new_full((), -1))
    return positions, mask


def _build_window_positions(total_lens: torch.Tensor, window_size: int):
    """Build chronologically ordered trailing window positions padded with
    `-1`."""
    device = total_lens.device
    if window_size == 0:
        empty = torch.empty((total_lens.numel(), 0), dtype=torch.long, device=device)
        return empty, total_lens.new_zeros((total_lens.numel(), )), empty.bool()
    arange = torch.arange(window_size, device=device).unsqueeze(0)
    window_lens = total_lens.clamp(max=window_size)
    starts = total_lens - window_lens
    mask = arange < window_lens.unsqueeze(1)
    positions = starts.unsqueeze(1) + arange
    positions = torch.where(mask, positions, positions.new_full((), -1))
    return positions, window_lens, mask


def _build_topk_range(lengths: torch.Tensor, width: int, offset: int = 0):
    """Build `[offset, ..., offset+len-1]` padded with `-1`."""
    positions, mask = _build_prefix_positions(lengths, width)
    positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1)


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
                 kernel_mod,
                 compress_ratio: int,
                 head_dim: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None,
                 rotate: bool = False):
        super().__init__()
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
        self.freqs_cis = None

    def _overlap_transform(self, tensor: torch.Tensor, value=0):
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((tensor.size(0), tensor.size(1), 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int, compress_state: torch.Tensor):
        bsz, seqlen, _ = x.size()
        assert bsz == 1
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        dtype = x.dtype
        kv = self.wkv(x)
        score = self.wgate(x)
        rows = (2 if self.overlap else 1) * ratio

        # compress_state shape: (rows, state_dim) where state_dim == coff * head_dim
        kv_state = compress_state[:rows]
        score_state = compress_state[rows:2 * rows]

        if start_pos == 0:
            score_state.fill_(float('-inf'))

        should_compress = False
        compressed = None
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if self.overlap else 0
            if self.overlap and cutoff >= ratio:
                kv_state[:ratio] = kv[0, cutoff - ratio:cutoff]
                score_state[:ratio] = score[0, cutoff - ratio:cutoff] + self.ape
            if cutoff > 0:
                work_kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
                work_score = score[:, :cutoff].unflatten(1, (-1, ratio)) + self.ape
                if self.overlap:
                    work_kv = self._overlap_transform(work_kv, 0)
                    work_score = self._overlap_transform(work_score, float('-inf'))
                compressed = (work_kv * work_score.softmax(dim=2)).sum(dim=2)
            if remainder > 0:
                kv_state[offset:offset + remainder] = kv[0, cutoff:]
                score_state[offset:offset + remainder] = score[0, cutoff:] + self.ape[:remainder]
        else:
            should_compress = (start_pos + 1) % ratio == 0
            pos = start_pos % ratio
            score = score + self.ape[pos]
            if self.overlap:
                kv_state[ratio + pos] = kv[0, 0]
                score_state[ratio + pos] = score[0, 0]
                if should_compress:
                    merged_kv = torch.cat([kv_state[:ratio, :self.head_dim], kv_state[ratio:, self.head_dim:]], dim=0)
                    merged_score = torch.cat(
                        [score_state[:ratio, :self.head_dim], score_state[ratio:, self.head_dim:]], dim=0)
                    compressed = (merged_kv * merged_score.softmax(dim=0)).sum(dim=0, keepdim=True).unsqueeze(0)
                    kv_state[:ratio] = kv_state[ratio:]
                    score_state[:ratio] = score_state[ratio:]
            else:
                kv_state[pos] = kv[0, 0]
                score_state[pos] = score[0, 0]
                if should_compress:
                    compressed = (kv_state * score_state.softmax(dim=0)).sum(dim=0, keepdim=True).unsqueeze(0)
        if compressed is None:
            return None
        compressed = self.norm(compressed.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(compressed[..., -rd:], freqs_cis)
        if self.rotate:
            compressed = rotate_activation(compressed)
            self.kernel_mod.fp4_act_quant(compressed, 32, True)
        else:
            self.kernel_mod.act_quant(compressed[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)
        return compressed

    def forward_decode(self,
                       x: torch.Tensor,
                       start_pos: torch.Tensor,
                       compress_state: torch.Tensor,
                       valid_mask: torch.Tensor):
        """Tensorized decode path for `q_len == 1`.

        Returns:
            compressed: `[bsz, head_dim]` with valid rows filled.
            emit_mask: `[bsz]`, whether the current token emits one compressed entry.
        """
        bsz, seqlen, _ = x.size()
        assert seqlen == 1
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        dtype = x.dtype
        kv = self.wkv(x).squeeze(1)
        score = self.wgate(x).squeeze(1)
        safe_start_pos = torch.where(valid_mask, start_pos, start_pos.new_zeros(()))
        rows = (2 if self.overlap else 1) * ratio
        kv_state = compress_state[:, :rows]
        score_state = compress_state[:, rows:2 * rows]
        score_state.masked_fill_((valid_mask & (safe_start_pos == 0)).view(-1, 1, 1), float('-inf'))

        emit_mask = valid_mask & (torch.remainder(safe_start_pos + 1, ratio) == 0)
        pos = torch.remainder(safe_start_pos, ratio)
        score = score + self.ape[pos]

        batch_idx = torch.arange(bsz, device=x.device)
        compressed = x.new_zeros((bsz, self.head_dim))

        if self.overlap:
            write_pos = ratio + pos
            prev_kv = kv_state[batch_idx, write_pos]
            prev_score = score_state[batch_idx, write_pos]
            kv_state[batch_idx, write_pos] = torch.where(valid_mask.view(-1, 1), kv, prev_kv)
            score_state[batch_idx, write_pos] = torch.where(valid_mask.view(-1, 1), score, prev_score)
            merged_kv = torch.cat([kv_state[:, :ratio, :self.head_dim], kv_state[:, ratio:, self.head_dim:]], dim=1)
            merged_score = torch.cat([score_state[:, :ratio, :self.head_dim], score_state[:, ratio:, self.head_dim:]],
                                     dim=1)
            active_compressed = (merged_kv * merged_score.softmax(dim=1)).sum(dim=1)
            kv_state[:, :ratio] = torch.where(emit_mask.view(-1, 1, 1), kv_state[:, ratio:], kv_state[:, :ratio])
            score_state[:, :ratio] = torch.where(emit_mask.view(-1, 1, 1), score_state[:, ratio:],
                                                 score_state[:, :ratio])
        else:
            prev_kv = kv_state[batch_idx, pos]
            prev_score = score_state[batch_idx, pos]
            kv_state[batch_idx, pos] = torch.where(valid_mask.view(-1, 1), kv, prev_kv)
            score_state[batch_idx, pos] = torch.where(valid_mask.view(-1, 1), score, prev_score)
            active_compressed = (kv_state * score_state.softmax(dim=1)).sum(dim=1)

        active_compressed = self.norm(active_compressed.to(dtype))
        active_freqs = self.freqs_cis[(safe_start_pos + 1 - ratio).clamp(min=0)].unsqueeze(1)
        active_view = active_compressed.unsqueeze(1)
        apply_rotary_emb(active_view[..., -rd:], active_freqs)
        active_compressed = active_view.squeeze(1)

        if self.rotate:
            active_compressed = rotate_activation(active_compressed)
            self.kernel_mod.fp4_act_quant(active_compressed, 32, True)
        else:
            self.kernel_mod.act_quant(active_compressed[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)

        compressed = torch.where(emit_mask.view(-1, 1), active_compressed, compressed)
        return compressed, emit_mask


class Indexer(nn.Module):

    def __init__(self,
                 config,
                 args: V4Args,
                 kernel_mod,
                 compress_ratio: int,
                 world_size: int,
                 rank: int,
                 dtype: torch.dtype,
                 device: torch.device | str | None):
        super().__init__()
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
        self.compressor = Compressor(args, kernel_mod, compress_ratio, self.head_dim,
                                     dtype=dtype, device=device, rotate=True)
        self.freqs_cis = None
        self.indexer_fwd = NativeV4Indexer(index_topk=self.index_topk,
                                           compress_ratio=self.compress_ratio,
                                           world_size=world_size)

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                start_pos: int,
                offset: int,
                compress_state: torch.Tensor,
                index_kv_cache: torch.Tensor,
                block_offsets: torch.Tensor,
                seq_idx: int,
                block_size: int,
                layer_id: int):
        bsz, seqlen, _ = x.size()
        assert bsz == 1
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        self.kernel_mod.fp4_act_quant(q, 32, True)
        new_kv = self.compressor(x, start_pos, compress_state)

        # gather existing index entries from block cache
        total_len = start_pos + seqlen
        num_index = total_len // ratio
        if num_index > 0:
            index_tokens = []
            for pos in range(num_index):
                bidx = pos // block_size
                boff = pos % block_size
                pblock = block_offsets[seq_idx, bidx]
                index_tokens.append(index_kv_cache[layer_id, pblock, boff])
            index_cache = torch.stack(index_tokens)
        else:
            index_cache = None

        if new_kv is not None:
            # write new index entries into block cache
            for i, entry in enumerate(new_kv[0]):
                abs_pos = (start_pos // ratio) + i
                bidx = abs_pos // block_size
                boff = abs_pos % block_size
                pblock = block_offsets[seq_idx, bidx]
                index_kv_cache[layer_id, pblock, boff] = entry
            if index_cache is None:
                index_cache = new_kv[0]
            else:
                index_cache = torch.cat([index_cache, new_kv[0]], dim=0)

        if index_cache is None or index_cache.size(0) == 0:
            return get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, x.device)
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        score = torch.einsum('bshd,td->bsht', q, index_cache)
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if self.world_size > 1:
            dist.all_reduce(score)
        topk = score.topk(min(self.index_topk, index_cache.size(0)), dim=-1)[1]
        return topk + offset

    def forward_decode(self,
                       x: torch.Tensor,
                       qr: torch.Tensor,
                       start_pos: torch.Tensor,
                       offset: int,
                       compress_state: torch.Tensor,
                       index_kv_cache: torch.Tensor,
                       block_offsets: torch.Tensor,
                       block_size: int,
                       layer_id: int,
                       index_scratch: torch.Tensor,
                       valid_mask: torch.Tensor):
        bsz, seqlen, _ = x.size()
        assert seqlen == 1
        rd = self.rope_head_dim
        self.compressor.freqs_cis = self.freqs_cis
        safe_start_pos = torch.where(valid_mask, start_pos, start_pos.new_zeros(()))
        freqs_cis = self.freqs_cis[safe_start_pos].unsqueeze(1)

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        self.kernel_mod.fp4_act_quant(q, 32, True)

        new_kv, emit_mask = self.compressor.forward_decode(x, safe_start_pos, compress_state, valid_mask)
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        meta = V4IndexerMetadata(block_offsets=block_offsets,
                                 start_pos=safe_start_pos,
                                 valid_mask=valid_mask,
                                 state_ids=start_pos.new_zeros(start_pos.shape),
                                 compress_ratio=self.compress_ratio)
        topk = self.indexer_fwd.forward_decode(q,
                                               weights,
                                               new_kv,
                                               emit_mask,
                                               index_kv_cache,
                                               meta,
                                               block_size,
                                               layer_id,
                                               index_scratch)
        valid = topk >= 0
        return torch.where(valid, topk + offset, topk)

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
                                          compress_ratio=self.compress_ratio,
                                          kernel_mod=kernel_mod)
        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(args, kernel_mod, self.compress_ratio, self.head_dim,
                                         dtype=dtype, device=device)
            if self.compress_ratio == 4:
                world_size, rank = get_tp_world_rank('attn')
                self.indexer = Indexer(config, args, kernel_mod, self.compress_ratio,
                                       world_size, rank, dtype=dtype, device=device)

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.original_seq_len, original_seq_len, rope_theta,
                                              args.rope_factor, args.beta_fast, args.beta_slow).to(device)

    def _attn_sink_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        world_size, rank = get_tp_world_rank('attn')
        _load_vector_shard(param, loaded_weight, world_size, rank)

    @staticmethod
    def _gather_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                              block_size: int):
        """Gather entries from a named block cache with `-1` padded
        positions."""
        if positions.numel() == 0:
            return cache.new_empty((*positions.shape, cache.size(-1)))
        safe_positions = positions.clamp(min=0)
        block_idx = torch.div(safe_positions, block_size, rounding_mode='floor').long()
        max_block_idx = block_offsets.size(1)
        valid = (positions >= 0) & (block_idx < max_block_idx)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_off = torch.remainder(safe_positions, block_size).long()
        phys_blocks = block_offsets.gather(1, safe_block_idx).long()
        gathered = cache[phys_blocks, block_off]
        return torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))

    @staticmethod
    def _write_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, batch_idx: torch.Tensor,
                             positions: torch.Tensor, values: torch.Tensor, block_size: int,
                             write_mask: torch.Tensor | None = None):
        """Write one entry per batch item into a named block cache."""
        if positions.numel() == 0:
            return
        block_idx = torch.div(positions, block_size, rounding_mode='floor').long()
        valid = (positions >= 0) & (block_idx < block_offsets.size(1))
        safe_block_idx = block_idx.clamp(max=block_offsets.size(1) - 1)
        block_off = torch.remainder(positions, block_size).long()
        phys_blocks = block_offsets[batch_idx, safe_block_idx].long()
        if write_mask is None:
            write_mask = valid
        else:
            write_mask = write_mask & valid
        if write_mask is None:
            cache[phys_blocks, block_off] = values.to(cache.dtype)
            return
        target = cache[phys_blocks, block_off]
        values = values.to(target.dtype)
        blend_mask = write_mask.view(-1, *([1] * (values.dim() - 1)))
        cache[phys_blocks, block_off] = torch.where(blend_mask, values, target)

    def _build_decode_attention_metadata(self,
                                         block_offsets: torch.Tensor,
                                         total_lens: torch.Tensor,
                                         slot: torch.Tensor,
                                         valid_mask: torch.Tensor,
                                         topk_indices: torch.Tensor,
                                         compressed_positions: torch.Tensor | None = None,
                                         decode_scratch: dict[str, torch.Tensor] | None = None):
        window_positions, window_lens, _ = _build_window_positions(total_lens.long(), self.window_size)

        compressed_valid_mask = None
        if compressed_positions is not None:
            compressed_valid_mask = compressed_positions >= 0
        elif self.compress_ratio:
            num_compressed = torch.div(total_lens, self.compress_ratio, rounding_mode='floor').long()
            if decode_scratch is not None:
                if self.compress_ratio == 4:
                    max_width = decode_scratch['selected_compressed_kv_r4'].size(1)
                else:
                    max_width = decode_scratch['selected_compressed_kv_r128'].size(1)
            else:
                max_width = int(num_compressed.max().item()) if num_compressed.numel() > 0 else 0
            compressed_positions, compressed_valid_mask = _build_prefix_positions(num_compressed, max_width)

        return V4AttentionMetadata(is_decoding=True,
                                   block_offsets=block_offsets,
                                   q_seqlens=torch.ones_like(total_lens),
                                   kv_seqlens=total_lens,
                                   state_ids=slot.long(),
                                   topk_indices=topk_indices,
                                   window_positions=window_positions,
                                   window_lens=window_lens,
                                   valid_mask=valid_mask,
                                   compress_ratio=self.compress_ratio,
                                   compressed_positions=compressed_positions,
                                   compressed_valid_mask=compressed_valid_mask)

    def forward(self, x: torch.Tensor, start_pos: int, slot: int, context: StepContext, seq_idx: int):
        bsz, seqlen, _ = x.size()
        assert bsz == 1
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        rd = self.rope_head_dim
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        self.kernel_mod.act_quant(kv[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)

        block_caches = context.block_caches
        named_state_caches = context.named_state_caches
        block_offsets = context.block_offsets
        block_size = context.cache_config.kernel_block_size
        raw_kv_cache = block_caches['v4_raw_kv'][self.layer_id]

        # ---- write raw kv ----
        raw_positions = torch.arange(start_pos, start_pos + seqlen, device=x.device, dtype=torch.long)
        raw_batch_idx = torch.full_like(raw_positions, seq_idx)
        self._write_cache_entries(raw_kv_cache, block_offsets, raw_batch_idx, raw_positions, kv[0], block_size)

        # ---- gather window ----
        total_len = start_pos + seqlen
        if start_pos == 0:
            # prefill: use the full kv directly (topk indices are absolute positions)
            window_kv = kv[0]
        else:
            window_start = max(0, total_len - self.window_size)
            window_positions = torch.arange(window_start, total_len, device=x.device, dtype=torch.long).unsqueeze(0)
            window_kv = self._gather_cache_entries(raw_kv_cache, block_offsets[seq_idx:seq_idx + 1], window_positions,
                                                   block_size)[0]

        topk_idxs = get_window_topk_idxs(self.window_size, bsz, seqlen, start_pos, x.device)
        compressed_kv = None
        offset = window_kv.size(0) + (0 if start_pos > 0 else kv.size(1))

        if self.compress_ratio:
            # ---- compressed / index ----
            if self.compress_ratio == 4:
                compressed_kv_cache = block_caches['v4_compressed_kv_r4'][self.layer_id]
                compress_state = named_state_caches['v4_compress_state_r4'][slot]
            else:
                compressed_kv_cache = block_caches['v4_compressed_kv_r128'][self.layer_id]
                compress_state = named_state_caches['v4_compress_state_r128'][slot]

            # call compressor
            self.compressor.freqs_cis = self.freqs_cis
            compressed = self.compressor(x, start_pos, compress_state)
            if compressed is not None:
                compressed_positions = torch.arange(start_pos // self.compress_ratio,
                                                    start_pos // self.compress_ratio + compressed.size(1),
                                                    device=x.device,
                                                    dtype=torch.long)
                compressed_batch_idx = torch.full_like(compressed_positions, seq_idx)
                self._write_cache_entries(compressed_kv_cache,
                                          block_offsets,
                                          compressed_batch_idx,
                                          compressed_positions,
                                          compressed[0],
                                          block_size)

            # gather compressed
            num_compressed = total_len // self.compress_ratio
            compressed_positions = torch.arange(num_compressed, device=x.device, dtype=torch.long).unsqueeze(0)
            compressed_kv = self._gather_cache_entries(compressed_kv_cache,
                                                       block_offsets[seq_idx:seq_idx + 1],
                                                       compressed_positions,
                                                       block_size)[0]

            # indexer
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
                index_kv_cache = block_caches['v4_index_kv_r4']
                compress_state_idx = named_state_caches['v4_compress_state_r4_idx'][slot]
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset, compress_state_idx, index_kv_cache,
                                                  block_offsets, seq_idx, block_size, self.layer_id)
            else:
                compress_topk_idxs = get_compress_topk_idxs(self.compress_ratio, bsz, seqlen, start_pos, offset,
                                                            x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

        # ---- build full_kv ----
        pieces = [window_kv]
        if compressed_kv is not None and compressed_kv.numel() > 0:
            pieces.append(compressed_kv)
        full_kv = torch.cat(pieces, dim=0)

        out = self.kernel_mod.sparse_attn(q, full_kv.unsqueeze(0), self.attn_sink, topk_idxs.int(), self.softmax_scale)
        apply_rotary_emb(out[..., -rd:], freqs_cis, True)
        out = out.view(bsz, seqlen, self.n_local_groups, -1)
        old_out = out.flatten(0, 1)
        out = old_out.new_empty(*old_out.shape[:-1], self.o_lora_rank)
        self.wo_a(old_out, out)
        return self.wo_b(out.flatten(-2, -1).unflatten(0, (bsz, seqlen)))

    def forward_decode(self,
                       x: torch.Tensor,
                       start_pos: torch.Tensor,
                       slot: torch.Tensor,
                       context: StepContext,
                       decode_scratch: dict[str, torch.Tensor] | None = None,
                       valid_mask: torch.Tensor | None = None):
        bsz, seqlen, _ = x.size()
        assert seqlen == 1
        if valid_mask is None:
            valid_mask = torch.ones((bsz, ), dtype=torch.bool, device=x.device)
        rd = self.rope_head_dim
        safe_start_pos = torch.where(valid_mask, start_pos, start_pos.new_zeros(()))
        freqs_cis = self.freqs_cis[safe_start_pos].unsqueeze(1)
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        self.kernel_mod.act_quant(kv[..., :-rd], 64, 'ue8m0', torch.float8_e8m0fnu, True)

        block_caches = context.block_caches
        named_state_caches = context.named_state_caches
        block_offsets = context.block_offsets.long()
        block_size = context.cache_config.kernel_block_size
        raw_kv_cache = block_caches['v4_raw_kv']
        cache_layer = raw_kv_cache[self.layer_id]

        batch_idx = torch.arange(bsz, device=x.device)
        total_lens = torch.where(valid_mask, safe_start_pos + 1, start_pos.new_zeros(()))
        self._write_cache_entries(cache_layer,
                                  block_offsets,
                                  batch_idx,
                                  safe_start_pos.long(),
                                  kv[:, 0],
                                  block_size,
                                  write_mask=valid_mask)

        if decode_scratch is None:
            max_total_len = int(total_lens.max().item())
            decode_scratch = self._alloc_decode_scratch(bsz, max_total_len, x.device)

        window_positions, window_lens, _ = _build_window_positions(total_lens.long(), self.window_size)
        window_topk = _build_topk_range(window_lens.long(), self.window_size, offset=0)
        offset = self.window_size

        compressed_cache = None
        topk_parts = [window_topk]
        compressed_positions = None
        if self.compress_ratio:
            if self.compress_ratio == 4:
                compressed_cache = block_caches['v4_compressed_kv_r4'][self.layer_id]
                compress_state = named_state_caches['v4_compress_state_r4'][slot.long()]
            else:
                compressed_cache = block_caches['v4_compressed_kv_r128'][self.layer_id]
                compress_state = named_state_caches['v4_compress_state_r128'][slot.long()]

            self.compressor.freqs_cis = self.freqs_cis
            new_compressed, emit_mask = self.compressor.forward_decode(x,
                                                                       safe_start_pos.long(),
                                                                       compress_state,
                                                                       valid_mask)
            self._write_cache_entries(compressed_cache,
                                      block_offsets,
                                      batch_idx,
                                      torch.div(safe_start_pos, self.compress_ratio, rounding_mode='floor').long(),
                                      new_compressed,
                                      block_size,
                                      write_mask=emit_mask)

            if self.indexer is not None:
                index_cache = block_caches['v4_index_kv_r4']
                index_state = named_state_caches['v4_compress_state_r4_idx'][slot.long()]
                index_scratch = decode_scratch['selected_index_kv_r4'][:bsz]
                compress_topk = self.indexer.forward_decode(x,
                                                            qr,
                                                            safe_start_pos.long(),
                                                            offset,
                                                            index_state,
                                                            index_cache,
                                                            block_offsets,
                                                            block_size,
                                                            self.layer_id,
                                                            index_scratch,
                                                            valid_mask)
                compressed_positions = torch.where(compress_topk >= 0, compress_topk - offset, compress_topk)
                compressed_positions = compressed_positions.squeeze(1)
                compact_topk = torch.arange(compressed_positions.size(-1), device=x.device, dtype=torch.long)
                compact_topk = compact_topk.view(1, 1, -1).expand_as(compress_topk)
                compress_topk = torch.where(compress_topk >= 0, compact_topk + offset,
                                            compact_topk.new_full((), -1))
            else:
                num_compressed = torch.div(total_lens, self.compress_ratio, rounding_mode='floor').long()
                if self.compress_ratio == 4:
                    comp_width = decode_scratch['selected_compressed_kv_r4'].size(1)
                else:
                    comp_width = decode_scratch['selected_compressed_kv_r128'].size(1)
                compress_topk = _build_topk_range(num_compressed, comp_width, offset=offset)
            topk_parts.append(compress_topk)

        topk_idxs = torch.cat(topk_parts, dim=-1)
        attn_meta = self._build_decode_attention_metadata(block_offsets,
                                                          total_lens.long(),
                                                          slot,
                                                          valid_mask,
                                                          topk_idxs,
                                                          compressed_positions,
                                                          decode_scratch)
        out = self.attn_fwd.forward_decode(q,
                                           cache_layer,
                                           self.attn_sink,
                                           attn_meta,
                                           block_size,
                                           compressed_kv_cache=compressed_cache,
                                           decode_scratch=decode_scratch)
        apply_rotary_emb(out[..., -rd:], freqs_cis, True)
        out = out.view(bsz, seqlen, self.n_local_groups, -1)
        old_out = out.flatten(0, 1)
        out = old_out.new_empty(*old_out.shape[:-1], self.o_lora_rank)
        self.wo_a(old_out, out)
        return self.wo_b(out.flatten(-2, -1).unflatten(0, (bsz, seqlen)))

    def _alloc_decode_scratch(self, batch_size: int, max_total_len: int, device: torch.device):
        max_comp_r4 = max_total_len // 4
        max_comp_r128 = max_total_len // 128
        selected_comp_r4 = self.indexer.index_topk if self.indexer is not None else max_comp_r4
        return {
            'selected_window_kv': torch.empty((batch_size, self.window_size, self.head_dim),
                                              dtype=torch.bfloat16,
                                              device=device),
            'selected_compressed_kv_r4': torch.empty((batch_size, selected_comp_r4, self.head_dim),
                                                     dtype=torch.bfloat16,
                                                     device=device),
            'selected_compressed_kv_r128': torch.empty((batch_size, max_comp_r128, self.head_dim),
                                                       dtype=torch.bfloat16,
                                                       device=device),
            'selected_index_kv_r4': torch.empty((batch_size, max_comp_r4, self.indexer.head_dim),
                                                dtype=torch.bfloat16,
                                                device=device) if self.indexer is not None else torch.empty(
                                                    (batch_size, 0, 0), dtype=torch.bfloat16, device=device),
            'selected_full_kv_r4': torch.empty((batch_size, self.window_size + selected_comp_r4, self.head_dim),
                                               dtype=torch.bfloat16,
                                               device=device),
            'selected_full_kv_r128': torch.empty((batch_size, self.window_size + max_comp_r128, self.head_dim),
                                                 dtype=torch.bfloat16,
                                                 device=device),
        }


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

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor, slot: int, context: StepContext,
                seq_idx: int):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, slot, context, seq_idx)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x

    def forward_decode(self,
                       x: torch.Tensor,
                       start_pos: torch.Tensor,
                       input_ids: torch.Tensor,
                       slot: torch.Tensor,
                       context: StepContext,
                       decode_scratch: dict[str, torch.Tensor] | None = None,
                       valid_mask: torch.Tensor | None = None):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.forward_decode(x, start_pos, slot, context, decode_scratch, valid_mask=valid_mask)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x


class DeepseekV4ForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """DeepSeek-V4 bring-up model.

    Decode uses lmdeploy named block/state caches as the primary history storage. The older per-sequence compressed-
    cache path is kept only as an eager fallback/debug path while tensorized decode is being stabilized.
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

        # TODO: remove this shit!
        self._decode_graph_scratch: dict[str, torch.Tensor] | None = None

    def _hc_head(self, x: torch.Tensor):
        # TODO: try fuse this head
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

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        # The V4 decode path has been refactored to use a backend wrapper,
        # but the cache-aware graph-safe contract is not complete yet.
        # Keep cudagraph disabled until raw/compressed/index retrieval is
        # fully handled inside the backend without dense fallback scratch.
        return False

    def make_buffers_cudagraph(self, graph_meta, history_lengths: torch.Tensor | None = None, **kwargs):
        input_buffers = super().make_buffers_cudagraph(graph_meta, history_lengths=history_lengths, **kwargs)
        max_total_len = graph_meta.num_blocks * graph_meta.block_size
        self._decode_graph_scratch = self._alloc_decode_scratch(graph_meta.max_batchs, max_total_len, graph_meta.device)
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta, input_ids: torch.Tensor, **kwargs):
        new_inputs = super().fill_buffers_cudagraph(graph_meta, input_ids=input_ids, **kwargs)
        new_inputs['decode_scratch'] = self._decode_graph_scratch
        return new_inputs

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list | None = None,
                attn_metadata=None,
                inputs_embeds: torch.Tensor | None = None,
                q_seqlens: torch.Tensor | None = None,
                history_lengths: torch.Tensor | None = None,
                state_ids: torch.Tensor | None = None,
                decode_scratch: dict[str, torch.Tensor] | None = None,
                **kwargs):
        if q_seqlens is None:
            q_seqlens = attn_metadata.q_seqlens
        if history_lengths is None:
            if attn_metadata is not None:
                history_lengths = attn_metadata.kv_seqlens.to(torch.long) - q_seqlens.to(torch.long)
            else:
                history_lengths = position_ids[0].to(torch.long)
        if state_ids is None:
            raise RuntimeError('DeepSeek-V4 requires state_ids to provide stable cache slots.')

        if attn_metadata is not None and attn_metadata.is_decoding and input_ids.size(-1) == q_seqlens.size(0):
            return self._forward_decode_batch(input_ids,
                                              history_lengths.to(torch.long),
                                              state_ids.to(torch.long),
                                              decode_scratch)

        return self._forward_eager_loop(input_ids, q_seqlens, history_lengths, state_ids)

    def _forward_eager_loop(self,
                            input_ids: torch.Tensor,
                            q_seqlens: torch.Tensor,
                            history_lengths: torch.Tensor,
                            state_ids: torch.Tensor):
        flat_input_ids = input_ids.flatten()
        outputs = []
        offset = 0
        context = self.ctx_mgr.current_context()
        for seq_idx, (seqlen, history_len, slot) in enumerate(
                zip(q_seqlens.tolist(), history_lengths.tolist(), state_ids.tolist())):
            seq_input_ids = flat_input_ids[offset:offset + seqlen].unsqueeze(0)
            offset += seqlen
            h = self.embed(seq_input_ids)
            h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
            for layer in self.layers:
                h = layer(h, history_len, seq_input_ids, int(slot), context, seq_idx)
            outputs.append(h)
        return torch.cat(outputs, dim=1) if outputs else self.embed(input_ids).unsqueeze(2)

    def _alloc_decode_scratch(self, batch_size: int, max_total_len: int, device: torch.device):
        scratch: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            attn_scratch = layer.attn._alloc_decode_scratch(batch_size, max_total_len, device)
            for name, tensor in attn_scratch.items():
                if name not in scratch or any(a < b for a, b in zip(scratch[name].shape, tensor.shape)):
                    scratch[name] = tensor
        scratch['selected_valid_lens'] = torch.empty((batch_size, 3), dtype=torch.int32, device=device)
        return scratch

    def _forward_decode_batch(self,
                              input_ids: torch.Tensor,
                              history_lengths: torch.Tensor,
                              state_ids: torch.Tensor,
                              decode_scratch: dict[str, torch.Tensor] | None = None):
        context = self.ctx_mgr.current_context()
        bsz = history_lengths.numel()
        valid_mask = state_ids >= 0
        safe_state_ids = torch.where(valid_mask, state_ids, state_ids.new_zeros(()))
        if decode_scratch is None:
            max_total_len = context.block_offsets.size(1) * context.cache_config.kernel_block_size
            decode_scratch = self._alloc_decode_scratch(bsz, max_total_len, input_ids.device)
        flat_input_ids = input_ids.flatten()[:bsz]
        seq_input_ids = flat_input_ids.unsqueeze(1)
        h = self.embed(seq_input_ids)
        h = h.masked_fill(~valid_mask.view(-1, 1, 1), 0)
        h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        for layer in self.layers:
            h = layer.forward_decode(h,
                                     history_lengths,
                                     seq_input_ids,
                                     safe_state_ids,
                                     context,
                                     decode_scratch,
                                     valid_mask=valid_mask)
            h = h.masked_fill(~valid_mask.view(-1, 1, 1, 1), 0)
        # Align with lmdeploy agent expectations: hidden states are laid out as
        # [1, total_seq, hc_mult, hidden] before `_slice_outs()` runs.
        return h.transpose(0, 1)

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
            q_seqlens=context.q_seqlens,
            history_lengths=context.kv_seqlens - context.q_seqlens,
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
