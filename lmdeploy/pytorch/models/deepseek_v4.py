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

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.utils import get_logger

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin

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
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


@contextmanager
def _set_default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _maybe_to_dtype(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if weight.dtype == dtype:
        return weight
    if dtype == torch.float4_e2m1fn_x2 and weight.dtype == torch.int8:
        return weight.view(dtype)
    return weight.to(dtype)


def _load_vector_shard(param: nn.Parameter, loaded_weight: torch.Tensor, world_size: int, rank: int):
    if world_size > 1:
        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank].contiguous()
    loaded_weight = loaded_weight.to(param.dtype)
    param.data.copy_(loaded_weight)


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


class QuantLinear(nn.Module):
    """FP8 / FP4 / BF16 linear used by the official DeepSeek-V4 checkpoint."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_mod,
                 bias: bool = False,
                 dtype: torch.dtype | None = None,
                 device: torch.device | str | None = None,
                 shard_dim: int | None = None,
                 world_size: int = 1,
                 rank: int = 0):
        super().__init__()
        device = _resolve_device(device)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_mod = kernel_mod
        self.shard_dim = shard_dim
        self.world_size = world_size
        self.rank = rank
        dtype = dtype or torch.bfloat16
        self.scale = None

        local_in = in_features
        local_out = out_features
        if shard_dim == 0:
            assert out_features % world_size == 0
            local_out = out_features // world_size
        elif shard_dim == 1:
            assert in_features % world_size == 0
            local_in = in_features // world_size

        if dtype == torch.float4_e2m1fn_x2:
            self.weight = nn.Parameter(torch.empty(local_out, local_in // 2, dtype=dtype, device=device),
                                       requires_grad=False)
            self.scale = nn.Parameter(torch.empty(local_out, local_in // 32,
                                                  dtype=torch.float8_e8m0fnu,
                                                  device=device),
                                      requires_grad=False)
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                torch.empty(local_out, local_in, dtype=dtype, device=device), requires_grad=False
                )
            self.scale = nn.Parameter(torch.empty((local_out + 127) // 128, (local_in + 127) // 128,
                                                  dtype=torch.float8_e8m0fnu,
                                                  device=device),
                                      requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.empty(local_out, local_in, dtype=dtype, device=device), requires_grad=False
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(local_out, dtype=torch.float32, device=device), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.weight.weight_loader = self._weight_loader
        if self.scale is not None:
            self.scale.weight_loader = self._scale_loader
        if self.bias is not None:
            self.bias.weight_loader = self._bias_loader

    def _chunk_weight(self, loaded_weight: torch.Tensor, shard_dim: int | None):
        if shard_dim is None or self.world_size == 1:
            return loaded_weight
        return loaded_weight.chunk(self.world_size, dim=shard_dim)[self.rank].contiguous()

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        loaded_weight = self._chunk_weight(loaded_weight, self.shard_dim)
        loaded_weight = _maybe_to_dtype(loaded_weight, param.dtype)
        if loaded_weight.numel() == param.numel() and loaded_weight.shape != param.shape:
            loaded_weight = loaded_weight.view_as(param)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        scale_shard_dim = self.shard_dim
        loaded_weight = self._chunk_weight(loaded_weight, scale_shard_dim)
        loaded_weight = _maybe_to_dtype(loaded_weight, param.dtype)
        if loaded_weight.numel() == param.numel() and loaded_weight.shape != param.shape:
            loaded_weight = loaded_weight.view_as(param)
        param.data.copy_(loaded_weight)

    def _bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self.world_size > 1 and self.shard_dim == 0:
            loaded_weight = loaded_weight.chunk(self.world_size, dim=0)[self.rank].contiguous()
        elif self.world_size > 1 and self.shard_dim == 1 and self.rank != 0:
            loaded_weight = torch.zeros_like(param)
        loaded_weight = loaded_weight.to(param.dtype)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        assert self.bias is None, 'DeepSeek-V4 linear path does not expect bias.'
        input_dtype = x.dtype
        act_quant = self.kernel_mod.act_quant
        fp8_gemm = self.kernel_mod.fp8_gemm
        fp4_gemm = self.kernel_mod.fp4_gemm
        if self.weight.dtype == torch.float4_e2m1fn_x2:
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            qx, scale = act_quant(x, 128, 'ue8m0', torch.float8_e8m0fnu)
            with _set_default_dtype(torch.bfloat16):
                y = fp4_gemm(qx, scale, self.weight, self.scale, torch.float8_e8m0fnu)
        elif self.weight.dtype == torch.float8_e4m3fn:
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            qx, scale = act_quant(x, 128, 'ue8m0', torch.float8_e8m0fnu)
            with _set_default_dtype(torch.bfloat16):
                y = fp8_gemm(qx, scale, self.weight, self.scale, torch.float8_e8m0fnu)
        else:
            y = F.linear(x, self.weight)
        if self.shard_dim == 1 and self.world_size > 1:
            y = y.float()
            dist.all_reduce(y)
            y = y.to(input_dtype)
        return y


class ParallelEmbedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 dim: int,
                 world_size: int,
                 rank: int,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        device = _resolve_device(device)
        dtype = dtype or torch.bfloat16
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim, dtype=dtype, device=device),
                                   requires_grad=False)
        self.world_size = world_size
        self.rank = rank
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self.world_size > 1:
            loaded_weight = loaded_weight.chunk(self.world_size, dim=0)[self.rank].contiguous()
        param.data.copy_(loaded_weight.to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x = x.masked_fill(mask, 0)
            y = F.embedding(x, self.weight)
            y = y.masked_fill(mask.unsqueeze(-1), 0)
            dist.all_reduce(y)
            return y
        return F.embedding(x, self.weight)


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, world_size: int, rank: int, device: torch.device | str | None = None):
        super().__init__()
        device = _resolve_device(device)
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim, dtype=torch.float32, device=device),
                                   requires_grad=False)
        self.world_size = world_size
        self.rank = rank
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self.world_size > 1:
            loaded_weight = loaded_weight.chunk(self.world_size, dim=0)[self.rank].contiguous()
        param.data.copy_(loaded_weight.to(param.dtype))

    def get_logits(self, x: torch.Tensor):
        logits = F.linear(x.float(), self.weight)
        if self.world_size > 1:
            gathered = [torch.empty_like(logits) for _ in range(self.world_size)]
            dist.all_gather(gathered, logits)
            logits = torch.cat(gathered, dim=-1)
        return logits


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | str | None = None):
        super().__init__()
        device = _resolve_device(device)
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32, device=device), requires_grad=False)

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


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
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
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
    device = _resolve_device(device)
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
    device = _resolve_device(device)
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


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
                 device: torch.device | str | None,
                 rotate: bool = False):
        super().__init__()
        device = _resolve_device(device)
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
        self.wkv = QuantLinear(self.dim, coff * self.head_dim, kernel_mod, dtype=torch.float32, device=device)
        self.wgate = QuantLinear(self.dim, coff * self.head_dim, kernel_mod, dtype=torch.float32, device=device)
        self.norm = RMSNorm(self.head_dim, args.norm_eps, device=device)
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
        x = x.float()
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


class Indexer(nn.Module):

    def __init__(self,
                 args: V4Args,
                 kernel_mod,
                 compress_ratio: int,
                 world_size: int,
                 rank: int,
                 device: torch.device | str | None):
        super().__init__()
        device = _resolve_device(device)
        self.kernel_mod = kernel_mod
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.world_size = world_size
        self.rank = rank
        self.wq_b = QuantLinear(args.q_lora_rank,
                                self.n_heads * self.head_dim,
                                kernel_mod,
                                dtype=torch.float8_e4m3fn,
                                device=device,
                                shard_dim=0,
                                world_size=world_size,
                                rank=rank)
        self.weights_proj = QuantLinear(args.dim,
                                        self.n_heads,
                                        kernel_mod,
                                        dtype=torch.bfloat16,
                                        device=device,
                                        shard_dim=0,
                                        world_size=world_size,
                                        rank=rank)
        self.compressor = Compressor(args, kernel_mod, compress_ratio, self.head_dim, device=device, rotate=True)
        self.freqs_cis = None

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


class Attention(nn.Module):

    def __init__(self,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 world_size: int,
                 rank: int,
                 device: torch.device | str | None):
        super().__init__()
        device = _resolve_device(device)
        self.kernel_mod = kernel_mod
        self.layer_id = layer_id
        self.world_size = world_size
        self.rank = rank
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32, device=device),
                                      requires_grad=False)
        self.attn_sink.weight_loader = self._attn_sink_loader
        self.wq_a = QuantLinear(args.dim, args.q_lora_rank, kernel_mod, dtype=torch.float8_e4m3fn, device=device)
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps, device=device)
        self.wq_b = QuantLinear(args.q_lora_rank,
                                args.n_heads * args.head_dim,
                                kernel_mod,
                                dtype=torch.float8_e4m3fn,
                                device=device,
                                shard_dim=0,
                                world_size=world_size,
                                rank=rank)
        self.wkv = QuantLinear(args.dim, args.head_dim, kernel_mod, dtype=torch.float8_e4m3fn, device=device)
        self.kv_norm = RMSNorm(args.head_dim, self.eps, device=device)
        self.wo_a = QuantLinear(args.n_heads * args.head_dim // self.n_groups,
                                self.n_groups * args.o_lora_rank,
                                kernel_mod,
                                dtype=torch.bfloat16,
                                device=device,
                                shard_dim=0,
                                world_size=world_size,
                                rank=rank)
        self.wo_b = QuantLinear(args.n_groups * args.o_lora_rank,
                                args.dim,
                                kernel_mod,
                                dtype=torch.float8_e4m3fn,
                                device=device,
                                shard_dim=1,
                                world_size=world_size,
                                rank=rank)
        self.softmax_scale = self.head_dim**-0.5
        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(args, kernel_mod, self.compress_ratio, self.head_dim, device=device)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, kernel_mod, self.compress_ratio, world_size, rank, device=device)

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.original_seq_len, original_seq_len, rope_theta,
                                              args.rope_factor, args.beta_fast, args.beta_slow).to(device)

    def _attn_sink_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        _load_vector_shard(param, loaded_weight, self.world_size, self.rank)

    @staticmethod
    def _gather_window(raw_kv_cache: torch.Tensor, block_offsets: torch.Tensor, seq_idx: int, block_size: int,
                       window_start: int, total_len: int, layer_id: int):
        """Gather window tokens from block cache in ring layout.

        The returned tensor is arranged in the same ring order that get_window_topk_idxs() expects for decode.
        """
        window_size = total_len - window_start
        tokens = []
        for pos in range(window_start, total_len):
            bidx = pos // block_size
            boff = pos % block_size
            pblock = block_offsets[seq_idx, bidx]
            tokens.append(raw_kv_cache[layer_id, pblock, boff])
        window = torch.stack(tokens)
        if window_size < total_len:
            # Re-arrange into ring layout to match get_window_topk_idxs
            cutoff = total_len % window_size
            if cutoff == 0:
                return window
            ring = window.new_empty(window_size, window.size(-1))
            ring[cutoff:window_size] = window[:window_size - cutoff]
            ring[:cutoff] = window[window_size - cutoff:]
            return ring
        return window

    @staticmethod
    def _gather_compressed(compressed_kv_cache: torch.Tensor, block_offsets: torch.Tensor, seq_idx: int,
                           block_size: int, num_compressed: int, layer_id: int):
        """Gather compressed entries from block cache."""
        if num_compressed == 0:
            return None
        tokens = []
        for pos in range(num_compressed):
            bidx = pos // block_size
            boff = pos % block_size
            pblock = block_offsets[seq_idx, bidx]
            tokens.append(compressed_kv_cache[layer_id, pblock, boff])
        return torch.stack(tokens)

    @staticmethod
    def _write_to_block_cache(cache: torch.Tensor, block_offsets: torch.Tensor, seq_idx: int, block_size: int,
                              data: torch.Tensor, start_pos: int, layer_id: int):
        """Write a sequence of entries into a block cache."""
        for i, entry in enumerate(data):
            abs_pos = start_pos + i
            bidx = abs_pos // block_size
            boff = abs_pos % block_size
            pblock = block_offsets[seq_idx, bidx]
            cache[layer_id, pblock, boff] = entry

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
        raw_kv_cache = block_caches['v4_raw_kv']

        # ---- write raw kv ----
        if start_pos == 0:
            self._write_to_block_cache(raw_kv_cache, block_offsets, seq_idx, block_size, kv[0], 0, self.layer_id)
        else:
            abs_pos = start_pos
            bidx = abs_pos // block_size
            boff = abs_pos % block_size
            pblock = block_offsets[seq_idx, bidx]
            raw_kv_cache[self.layer_id, pblock, boff] = kv[0, 0]

        # ---- gather window ----
        total_len = start_pos + seqlen
        if start_pos == 0:
            # prefill: use the full kv directly (topk indices are absolute positions)
            window_kv = kv[0]
        else:
            window_start = max(0, total_len - self.window_size)
            window_kv = self._gather_window(raw_kv_cache, block_offsets, seq_idx, block_size, window_start, total_len,
                                            self.layer_id)

        topk_idxs = get_window_topk_idxs(self.window_size, bsz, seqlen, start_pos, x.device)
        compressed_kv = None
        offset = window_kv.size(0) + (0 if start_pos > 0 else kv.size(1))

        if self.compress_ratio:
            # ---- compressed / index ----
            if self.compress_ratio == 4:
                compressed_kv_cache = block_caches['v4_compressed_kv_r4']
                compress_state = named_state_caches['v4_compress_state_r4'][slot]
            else:
                compressed_kv_cache = block_caches['v4_compressed_kv_r128']
                compress_state = named_state_caches['v4_compress_state_r128'][slot]

            # call compressor
            self.compressor.freqs_cis = self.freqs_cis
            compressed = self.compressor(x, start_pos, compress_state)
            if compressed is not None:
                self._write_to_block_cache(compressed_kv_cache, block_offsets, seq_idx, block_size, compressed[0],
                                           start_pos // self.compress_ratio, self.layer_id)

            # gather compressed
            num_compressed = total_len // self.compress_ratio
            compressed_kv = self._gather_compressed(compressed_kv_cache, block_offsets, seq_idx, block_size,
                                                    num_compressed, self.layer_id)

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
        wo_a = self.wo_a.weight.view(self.n_local_groups, -1, out.size(-1)).to(out.dtype)
        out = torch.einsum('bsgd,grd->bsgr', out, wo_a)
        return self.wo_b(out.flatten(2))


class Gate(nn.Module):

    def __init__(self, layer_id: int, args: V4Args, device: torch.device | str | None):
        super().__init__()
        device = _resolve_device(device)
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

    def __init__(self, dim: int, inter_dim: int, kernel_mod, dtype=None, swiglu_limit=0.0, device=None):
        super().__init__()
        self.w1 = QuantLinear(dim, inter_dim, kernel_mod, dtype=dtype, device=device)
        self.w2 = QuantLinear(inter_dim, dim, kernel_mod, dtype=dtype, device=device)
        self.w3 = QuantLinear(dim, inter_dim, kernel_mod, dtype=dtype, device=device)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):

    def __init__(self,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 world_size: int,
                 rank: int,
                 device: torch.device | str | None):
        super().__init__()
        self.dim = args.dim
        self.world_size = world_size
        self.rank = rank
        self.experts_per_rank = args.n_routed_experts // world_size
        self.start = rank * self.experts_per_rank
        self.end = self.start + self.experts_per_rank
        self.gate = Gate(layer_id, args, device=device)
        expert_dtype = torch.float4_e2m1fn_x2
        self.experts = nn.ModuleDict({
            str(i): Expert(args.dim,
                           args.moe_inter_dim,
                           kernel_mod,
                           dtype=expert_dtype,
                           swiglu_limit=args.swiglu_limit,
                           device=device)
            for i in range(self.start, self.end)
        })
        self.shared_experts = Expert(args.dim,
                                     args.moe_inter_dim,
                                     kernel_mod,
                                     dtype=torch.float8_e4m3fn,
                                     swiglu_limit=0.0,
                                     device=device)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.end).tolist()
        for i in range(self.start, self.end):
            if i >= len(counts) or counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[str(i)](x[idx], weights[idx, top, None])
        if self.world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):

    def __init__(self,
                 layer_id: int,
                 args: V4Args,
                 kernel_mod,
                 world_size: int,
                 rank: int,
                 device: torch.device | str | None):
        super().__init__()
        device = _resolve_device(device)
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args, kernel_mod, world_size, rank, device=device)
        self.ffn = MoE(layer_id, args, kernel_mod, world_size, rank, device=device)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, device=device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, device=device)
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


class DeepseekV4ForCausalLM(nn.Module, DeployModelMixin, CudaGraphMixin):
    """DeepSeek-V4 bring-up model.

    The current integration uses dedicated per-sequence compressed caches
    indexed by lmdeploy `state_offsets`, instead of the generic paged KV cache.
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
        self.device = _resolve_device(device)
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

        self.embed = ParallelEmbedding(config.vocab_size,
                                       config.hidden_size,
                                       self.world_size,
                                       self.rank,
                                       device=self.device,
                                       dtype=self.dtype)
        self.layers = nn.ModuleList([
            Block(layer_idx, self.args, self.kernel_mod, self.world_size, self.rank, device=self.device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=self.device)
        self.head = ParallelHead(config.vocab_size,
                                 config.hidden_size,
                                 self.world_size,
                                 self.rank,
                                 device=self.device)
        hc_dim = config.hc_mult * config.hidden_size
        with _set_default_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, device=self.device),
                                           requires_grad=False)
            self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, device=self.device), requires_grad=False)
            self.hc_head_scale = nn.Parameter(torch.empty(1, device=self.device), requires_grad=False)

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
        return self.head.get_logits(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """DeepSeek-V4 is not graph-safe yet.

        The current bring-up path performs host-side tensor value materialization
        (`tolist`) and mutates Python-managed compressed caches during decode.
        Those operations are incompatible with CUDA graph capture.
        """
        return False

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list | None = None,
                attn_metadata=None,
                inputs_embeds: torch.Tensor | None = None,
                q_seqlens: torch.Tensor | None = None,
                history_lengths: torch.Tensor | None = None,
                state_ids: torch.Tensor | None = None,
                **kwargs):
        if q_seqlens is None:
            q_seqlens = attn_metadata.q_seqlens
        if history_lengths is None:
            history_lengths = position_ids[0].to(torch.long)
        if state_ids is None:
            raise RuntimeError('DeepSeek-V4 requires state_ids to provide stable cache slots.')

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

    def update_model_metas(self,
                           past_key_values: list[list[torch.Tensor]],
                           inputs_embeds: torch.Tensor | None = None,
                           context: StepContext = None):
        return context.model_metas

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        pending_wo_a_weight: dict[str, torch.Tensor] = {}
        pending_wo_a_scale: dict[str, torch.Tensor] = {}

        def _maybe_load_wo_a(base_name: str):
            weight_name = f'{base_name}.weight'
            scale_name = f'{base_name}.scale'
            if weight_name not in pending_wo_a_weight or scale_name not in pending_wo_a_scale:
                return
            if weight_name not in params_dict:
                pending_wo_a_weight.pop(weight_name, None)
                pending_wo_a_scale.pop(scale_name, None)
                return
            dequantized = _dequantize_wo_a_shard(pending_wo_a_weight.pop(weight_name),
                                                 pending_wo_a_scale.pop(scale_name),
                                                 self.world_size,
                                                 self.rank)
            params_dict[weight_name].data.copy_(dequantized.to(params_dict[weight_name].dtype))

        for name, loaded_weight in weights:
            if name.startswith('mtp.'):
                continue
            if name.endswith('tie2eid'):
                name = name.replace('tie2eid', 'tid2eid')
            if '.experts.' in name and '.shared_experts.' not in name:
                match = re.search(r'\.experts\.(\d+)\.', name)
                if match is not None:
                    expert_id = int(match.group(1))
                    start = self.rank * (self.config.n_routed_experts // self.world_size)
                    end = start + (self.config.n_routed_experts // self.world_size)
                    if expert_id < start or expert_id >= end:
                        continue
            if name.endswith('wo_a.weight'):
                pending_wo_a_weight[name] = loaded_weight
                _maybe_load_wo_a(name[:-len('.weight')])
                continue
            if name.endswith('wo_a.scale'):
                pending_wo_a_scale[name] = loaded_weight
                _maybe_load_wo_a(name[:-len('.scale')])
                continue
            if name not in params_dict:
                logger.debug(f'Skip unknown DeepSeek-V4 weight: {name}')
                continue
            load_weight(params_dict[name], loaded_weight)

        if pending_wo_a_weight or pending_wo_a_scale:
            unresolved = sorted(set(pending_wo_a_weight) | set(pending_wo_a_scale))
            raise RuntimeError(f'Unresolved DeepSeek-V4 wo_a weight/scale pairs: {unresolved[:4]}')
