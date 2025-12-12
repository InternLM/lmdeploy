# Copyright (c) OpenMMLab. All rights reserved.

import math
from copy import deepcopy
from enum import Enum, auto
from os import getenv
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, ParallelEmbedding, RMSNorm, RopeType, SiluAndMul,
                                 build_rotary_embedding, build_rotary_params)
from lmdeploy.pytorch.nn.eplb import EPLBDispatchInfo, EPLBManager
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_down_linear, build_gateup_linear, build_o_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.moe import MoeType, SoftmaxTopK, build_fused_moe
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


# microbatch
class ExecType(Enum):
    """Batch exec type."""
    One = auto()
    Two0101 = auto()
    Two0110 = auto()
    TwoLikeOne = auto()
    TwoPrefill = auto()
    TwoDecode = auto()


class BatchWorker:

    def __init__(self, tag: str, generator):
        self._tag = tag
        self._generator = generator
        self._count = 0
        self.output = None

    def next(self):
        assert not self.done

        try:
            next(self._generator)
        except StopIteration as e:
            assert e.value is not None
            self.output = e.value

        self._count += 1

    @property
    def done(self):
        return self.output is not None


def execute_batch(inputs: list, fn, delta_stages: int = 0, exec_type: ExecType = ExecType.One, extern_tag: str = ''):
    worker_list = [BatchWorker(str(idx), fn(**input, tag=str(idx) + extern_tag)) for idx, input in enumerate(inputs)]

    if exec_type == ExecType.One:
        assert len(inputs) == 1
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            i += 1

    if exec_type == ExecType.TwoLikeOne:
        assert len(inputs) == 2
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            i += 1
        i = 0
        while not worker_list[1].done:
            worker_list[1].next()
            i += 1

    if exec_type == ExecType.Two0101:
        assert len(inputs) == 2

        for _ in range(delta_stages):
            worker_list[0].next()
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            worker_list[1].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.Two0110:
        assert len(inputs) == 2

        for _ in range(delta_stages):
            worker_list[0].next()
        i = 0
        while not worker_list[0].done:
            if i % 2 == 0:
                worker_list[0].next()
                worker_list[1].next()
            else:
                worker_list[1].next()
                worker_list[0].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.TwoPrefill:
        """
        before:
        A-attn0->A-attn1
        roll:
        B-attn0->B-attn1->A-dis->A-dis_wait->A-moe->B-dis->B-dis_wait->A-comb->
        B-moe->(A-share->A-comb_wait)->B-comb->A-attn0->A-attn1->(B-share->B-comb_wait)
        after:
        B-dis_wait->B-moe->B-comb->B-comb_wait and end
        """
        assert len(inputs) == 2 and delta_stages in [0, 2]

        for _ in range(2):
            worker_list[0].next()

        pipeline = [
            '1-attn0', '1-attn1', '0-dis', '0-dis_wait', '0-moe', '1-dis', '1-dis_wait', '0-comb', '1-moe',
            '0-share+0-comb_wait', '1-comb', '0-attn0', '0-attn1', '1-share+1-comb_wait'
        ]
        pipline_length = len(pipeline)
        i = 0
        while not worker_list[0].done:
            worker_list[int(pipeline[i % pipline_length][0])].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.TwoDecode:
        """
        before:
        A-attn0->A-attn1->(A-dis->A-share)
        roll:
        B-attn0->A-dis_wait->A-moe->A-comb->B-attn1->A-comb_wait->(B-dis->B-share)->
        A-attn0->B-dis_wait->B-moe->B-comb->A-attn1->B-comb_wait->(A-dis->A-share)
        after:
        B-dis_wait->B-moe->B-comb->B-comb_wait and end
        """
        assert len(inputs) == 2 and delta_stages in [0, 3]

        for _ in range(3):
            worker_list[0].next()

        pipeline = [
            '1-attn0', '0-dis_wait', '0-moe', '0-comb', '1-attn1', '0-comb_wait', '1-dis+1-share', '0-attn0',
            '1-dis_wait', '1-moe', '1-comb', '0-attn1', '1-comb_wait', '0-dis+0-share'
        ]
        pipline_length = len(pipeline)
        i = 0
        while not worker_list[0].done:
            worker_list[int(pipeline[i % pipline_length][0])].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    for worker in worker_list:
        assert worker.done
    return [worker.output for worker in worker_list]


def get_new_meta(attn_metadata, start_idx: int, end_idx: int):
    new_attn_metadata = deepcopy(attn_metadata)
    new_attn_metadata.block_offsets = attn_metadata.block_offsets[start_idx:end_idx, ...]
    new_attn_metadata.q_start_loc = attn_metadata.q_start_loc[start_idx:end_idx] - attn_metadata.q_start_loc[start_idx]
    new_attn_metadata.kv_start_loc = attn_metadata.kv_start_loc[start_idx:end_idx] - \
        attn_metadata.kv_start_loc[start_idx] if attn_metadata.kv_start_loc is not None else None
    new_attn_metadata.q_seqlens = attn_metadata.q_seqlens[start_idx:end_idx]
    new_attn_metadata.kv_seqlens = attn_metadata.kv_seqlens[start_idx:end_idx] \
        if attn_metadata.kv_seqlens is not None else None
    new_attn_metadata.kv_flatten_size = sum(new_attn_metadata.kv_seqlens.tolist()) \
        if attn_metadata.kv_flatten_size is not None else None
    # create buffers for flash mla
    if attn_metadata.num_splits is not None:
        Attention.update_meta_flashmla(new_attn_metadata,
                                       get_step_ctx_manager().current_context().model_config.num_attention_heads)
    return new_attn_metadata


def get_new_rotary_pos_emb(rotary_pos_emb, start_loc, end_loc):
    new_rotary_pos_emb = (rotary_pos_emb[0][start_loc:end_loc, ...].contiguous(), rotary_pos_emb[1][start_loc:end_loc,
                                                                                                    ...].contiguous())
    return new_rotary_pos_emb


def get_new_input(hidden_states, rotary_pos_emb, past_key_values, residual, attn_metadata, start_idx, end_idx,
                  start_loc, end_loc):
    new_hidden_states = hidden_states[:, start_loc:end_loc, :].contiguous()
    new_rotary_pos_emb = get_new_rotary_pos_emb(rotary_pos_emb, start_loc, end_loc)
    new_past_key_values = past_key_values
    new_residual = residual[:, start_loc:end_loc, :].contiguous() if residual is not None else None
    new_attn_metadata = get_new_meta(attn_metadata, start_idx, end_idx)
    return new_hidden_states, new_rotary_pos_emb, new_past_key_values, new_residual, new_attn_metadata


def get_split_flags(attn_metadata, num=2):
    """Split flags for seqlens and startloc, support 2 only."""
    assert num == 2
    if attn_metadata.is_decoding:
        batch_size = attn_metadata.q_start_loc.numel()
        flag_a = {
            'start_idx': 0,
            'end_idx': batch_size // 2,
            'start_loc': 0,
            'end_loc': batch_size // 2,
        }
        flag_b = {
            'start_idx': batch_size // 2,
            'end_idx': batch_size,
            'start_loc': batch_size // 2,
            'end_loc': batch_size,
        }
    else:
        q_start_loc = attn_metadata.q_start_loc.tolist()
        q_seqlens = attn_metadata.q_seqlens.tolist()
        total_len = sum(q_seqlens)
        min_diff = total_len
        split_flag = 1
        for idx in range(1, len(q_seqlens)):
            diff = abs(sum(q_seqlens[:idx]) - sum(q_seqlens[idx:]))
            if diff < min_diff:
                min_diff = diff
                split_flag = idx
        flag_a = {
            'start_idx': 0,
            'end_idx': split_flag,
            'start_loc': q_start_loc[0],
            'end_loc': q_start_loc[split_flag],
        }
        flag_b = {
            'start_idx': split_flag,
            'end_idx': len(q_seqlens),
            'start_loc': q_start_loc[split_flag],
            'end_loc': q_start_loc[-1] + q_seqlens[-1],
        }
    return [flag_a, flag_b]


def split_input(hidden_states,
                rotary_pos_emb,
                past_key_values,
                residual,
                attn_metadata,
                moe_start_idx,
                moe_end_idx,
                num=2):
    """Split input, support 1 or 2 only."""
    # one batch
    if num == 1:
        input = {
            'hidden_states': hidden_states,
            'rotary_pos_emb': rotary_pos_emb,
            'past_key_values': past_key_values,
            'residual': residual,
            'attn_metadata': attn_metadata,
            'start_idx': moe_start_idx,
            'end_idx': moe_end_idx
        }
        extern_tag = 'D' if attn_metadata.is_decoding else 'P'
        return [input], ExecType.One, 0, extern_tag
    else:
        # two batch or more
        flag_list = get_split_flags(attn_metadata, num=num)

        inputs = []
        for flag in flag_list:
            (hidden_states_splited, rotary_pos_emb_splited, past_key_values_splited, residual_splited,
             attn_metadata_splited) = get_new_input(hidden_states, rotary_pos_emb, past_key_values, residual,
                                                    attn_metadata, flag['start_idx'], flag['end_idx'],
                                                    flag['start_loc'], flag['end_loc'])
            input = {
                'hidden_states': hidden_states_splited,
                'rotary_pos_emb': rotary_pos_emb_splited,
                'past_key_values': past_key_values,
                'residual': residual_splited,
                'attn_metadata': attn_metadata_splited,
                'start_idx': moe_start_idx,
                'end_idx': moe_end_idx
            }
            inputs.append(input)

        if attn_metadata.is_decoding:
            exec_type = ExecType.TwoDecode
            delta_stages = 0
            extern_tag = 'D'
        else:
            exec_type = ExecType.TwoPrefill
            delta_stages = 0
            extern_tag = 'P'

        return inputs, exec_type, delta_stages, extern_tag


def merge_output(output_list):
    # one batch
    if len(output_list) == 1:
        return output_list[0]
    # two batch or more
    hidden_states = torch.concat([output[0] for output in output_list], dim=1)
    residual = None
    if output_list[0][1] is not None:
        residual = torch.concat([output[1] for output in output_list], dim=1)
    return hidden_states, residual


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2BMM(nn.Module):
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


class DeepseekV2Attention(nn.Module):
    """Deepseekv2 attention."""

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        num_key_value_heads = getattr(config, 'num_key_value_heads', 1)
        use_flash_mla = getattr(config, 'use_flash_mla', False)

        if self.q_lora_rank is None:
            self.q_proj = build_colwise_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
                quant_config=quantization_config,
                dp_disable_tp=True,
            )
        else:
            self.q_a_proj = build_colwise_linear(
                self.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                dtype=dtype,
                device=device,
                is_tp=False,
                quant_config=quantization_config,
            )
            self.q_a_layernorm = RMSNorm(config.q_lora_rank,
                                         1e-6,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device)
            self.q_b_proj = build_colwise_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
                quant_config=quantization_config,
                dp_disable_tp=True,
            )

        self.kv_a_proj_with_mqa = build_colwise_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank,
                                      1e-6,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device)
        self.kc = DeepseekV2BMM(self.num_heads,
                                config.qk_nope_head_dim,
                                config.kv_lora_rank,
                                dtype=dtype,
                                device=device)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.softmax_scale = self.q_head_dim**(-0.5)

        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get('mscale_all_dim', 0)
            scaling_factor = config.rope_scaling['factor']
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_fwd = Attention(self.num_heads,
                                  config.kv_lora_rank + self.qk_rope_head_dim,
                                  scale=self.softmax_scale,
                                  num_kv_heads=num_key_value_heads,
                                  v_head_size=config.kv_lora_rank,
                                  num_replicate_kv_heads=num_replicate_kv_heads,
                                  use_flash_mla=use_flash_mla)

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim, dtype=dtype, device=device)
        self.o_proj = build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
            quant_config=quantization_config,
        )

    def _q_proj(self, hidden_states, num_heads: int, nope_size: int, pe_size: int):
        """Q proj."""
        q_len = hidden_states.size(1)

        query_states = hidden_states.new_empty(q_len, num_heads, nope_size + pe_size)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: (q_len, num_heads, kv_lora_rank)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe

    def _kv_proj(self, hidden_states, nope_size: int):
        """Kv proj."""
        # (q_len, 1, nope_size + pe_size)
        key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        # (q_len, 1, pe_size)
        k_pe = key_states[..., nope_size:]
        # kv_a_layernorm
        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        return key_states, value_states, k_pe

    def _qkv_proj(self, hidden_states: torch.Tensor, num_heads: int):
        """Qkv proj."""
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        query_states, q_pe = self._q_proj(hidden_states, num_heads, nope_size, pe_size)
        key_states, value_states, k_pe = self._kv_proj(hidden_states, nope_size)

        return query_states, key_states, value_states, q_pe, k_pe

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        dist_config = get_dist_manager().current_config()
        if dist_config.dp > 1:
            num_heads = self.num_heads
        else:
            world_size = dist_config.world_size
            num_heads = self.num_heads // world_size
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        # qkv_proj
        query_states, key_states, value_states, q_pe, k_pe = self._qkv_proj(hidden_states, num_heads=num_heads)

        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            inplace=False,
        )
        query_states[..., nope_size:] = q_pe
        key_states[..., nope_size:] = k_pe

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

        self.vc(attn_output, attn_bmm_out)
        attn_output = attn_bmm_out.flatten(-2, -1)[None]
        attn_output = self.o_proj(attn_output)

        return attn_output


class MoEGate(nn.Module):
    """Deepseek Gate."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 info: EPLBDispatchInfo = None):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.renormalize = self.top_k > 1 and self.norm_topk_prob
        self.router_n_groups = getattr(config, 'router_n_groups', -1)
        assert self.top_k % self.router_n_groups == 0, f'{self.top_k} cannot be divided by {self.router_n_groups}'
        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))
        if self.topk_method == 'noaux_tc':
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts, ), dtype=dtype, device=device))
        self.softmax_topk = SoftmaxTopK(self.top_k, n_groups=self.router_n_groups)
        self.fake_eplb = getenv('LMDEPLOY_FAKE_EPLB', 'False').lower() == 'true'
        self.eplb_dispatch_info = info

    def _compute_scores(self, logits: torch.Tensor):
        """Compute scores."""
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == 'sigmoid':
            scores = logits.sigmoid()
        else:
            raise NotImplementedError('insupportable scoring function '
                                      f'for MoE gating: {self.scoring_func}')
        return scores

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        sequence_length, hidden_dim = hidden_states.shape
        router_logits = F.linear(hidden_states, self.weight)
        if self.fake_eplb:
            # Forcefully manipulate router_logits to simulate expert load balancing (EPLB).
            # This is a benchmark-only hack to achieve optimal performance metrics.
            router_logits = torch.randn_like(router_logits)

        if self.topk_method == 'greedy':
            topk_weight, topk_idx = self.softmax_topk(router_logits)
        elif self.topk_method == 'group_limited_greedy':
            scores = router_logits
            grouped_logits = scores.unflatten(-1, (self.n_group, -1))
            group_scores = (grouped_logits.max(-1).values)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            group_mask = ~group_mask.bool()[..., None]
            grouped_logits = grouped_logits.masked_fill(group_mask, 0.0)
            scores = grouped_logits.flatten(1, 2)
            topk_weight, topk_idx = self.softmax_topk(scores)
        elif self.topk_method == 'noaux_tc':
            scores = self._compute_scores(router_logits)
            scores_for_choice = scores.view(sequence_length, -1) + self.e_score_correction_bias[None]
            if self.router_n_groups > 0:
                assert scores_for_choice.shape[-1] % self.router_n_groups == 0, \
                    f'{scores_for_choice.shape[-1]} cannot be divided by {self.router_n_groups}'
                per_group_top_k = self.top_k // self.router_n_groups
                group_size = scores_for_choice.shape[-1] // self.router_n_groups
                group_offsets = self.softmax_topk.impl.get_group_offsets(self.router_n_groups,
                                                                         group_size,
                                                                         device=scores_for_choice.device)
                scores_for_choice = scores_for_choice.unflatten(-1, (self.router_n_groups, group_size))
                topk_weight, topk_idx = torch.topk(scores_for_choice, per_group_top_k, dim=-1)
                topk_idx = (topk_idx + group_offsets).flatten(-2, -1)
                topk_weight = topk_weight.flatten(-2, -1)
            else:
                group_scores = (scores_for_choice.view(sequence_length, self.n_group,
                                                       -1).topk(2, dim=-1)[0].sum(dim=-1))  # [n, n_group]
                group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
                group_mask = torch.zeros_like(group_scores)  # [n, n_group]
                group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
                score_mask = (group_mask.unsqueeze(-1).expand(sequence_length, self.n_group,
                                                              self.n_routed_experts // self.n_group).reshape(
                                                                  sequence_length, -1))  # [n, e]
                tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
                _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
                topk_weight = scores.gather(1, topk_idx)
        else:
            raise RuntimeError(f'Unsupported topk_method: {self.topk_method}')

        if self.renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            if not topk_weight.is_contiguous():
                topk_weight = topk_weight.contiguous()

        if not self.renormalize or self.topk_method == 'noaux_tc':
            topk_weight = topk_weight * self.routed_scaling_factor

        if self.eplb_dispatch_info is not None:
            topk_idx = EPLBManager.topk_ids_logical_to_physical(topk_idx, self.eplb_dispatch_info)

        return topk_weight, topk_idx


class DeepseekV2MoE(nn.Module):
    """Deepseek v2 MoE."""

    def __init__(self, config: Any, layer_idx, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.renormalize = self.top_k > 1 and self.norm_topk_prob
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        dp = dist_config.dp
        world_size = dist_config.world_size
        moe_all_reduce = dp > 1 and dist_config.tp > 1
        if get_dist_manager().current_context().dist_config.enable_eplb:
            eplb_dispatch_info = EPLBManager.get_dispatch_info(
                ep_rank=dist_ctx.ep_rank,
                layer_idx=layer_idx,
            )
            self.num_experts = EPLBManager.num_physical_experts()
            self.gate = MoEGate(config, dtype=dtype, device=device, info=eplb_dispatch_info)
        else:
            self.gate = MoEGate(config, dtype=dtype, device=device, info=None)
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=moe_all_reduce,
            quant_config=quantization_config,
            layer_idx=layer_idx,
        )
        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size * config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                config=config,
                intermediate_size=intermediate_size,
                dtype=dtype,
                device=device,
                is_shared_expert=True,
            )

        if dp == 1 and world_size > 1:
            self._all_reduce = True
        else:
            self._all_reduce = False

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self.gate(hidden_states)

        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        if self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
            out_states += shared_states
        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self._all_reduce:
            dist.all_reduce(out_states)

        return out_states


class DeepseekV2MLP(nn.Module):
    """Deepseek v2 mlp."""

    def __init__(self,
                 config: Any,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_shared_expert: bool = False):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        if is_shared_expert:
            dist_config = get_dist_manager().current_config()
            dp = dist_config.dp
            if dp == 1:
                # split weight, do all reduce in moe
                is_tp = True
                all_reduce = False
            else:
                # do not split weight on dp
                # TODO: support dp+tp?
                is_tp = False
                all_reduce = False
        else:
            all_reduce = True
            is_tp = True

        # gate up
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=is_tp,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_down_linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            all_reduce=all_reduce,
        )

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class DeepseekV2DecoderLayer(nn.Module):
    """Deepseekv2 decoder layer."""

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = None

        # build attention layer
        if getattr(config, 'use_mla', True):
            self.self_attn = DeepseekV2Attention(config, dtype=dtype, device=device)
        else:
            # deepseek-vl2-tiny uses MHA LlamaAttention structure
            from lmdeploy.pytorch.models.llama import LlamaAttention
            self.self_attn = LlamaAttention(config, dtype=dtype, device=device)

        # mlp
        self.mlp = (DeepseekV2MoE(config, layer_idx, dtype=dtype, device=device) if
                    (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
                     and layer_idx % config.moe_layer_freq == 0) else DeepseekV2MLP(config, dtype=dtype, device=device))

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs

    def forward_yield(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        tag: Any = None,
    ):
        """forward_yield."""
        is_decoding = attn_metadata.is_decoding
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # yield for attn0 and attn1
        yield
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MOE
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_idx = self.mlp.gate(hidden_states)

        topk_weights = self.mlp.experts.renormalize(topk_weights)
        topk_weights = topk_weights.to(torch.float32)
        topk_idx = topk_idx.to(torch.int64)
        hidden_shape = hidden_states.shape
        shared_states = None

        state = {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'raw_hidden_shape': hidden_shape,
            'moe_type': MoeType.DSAsyncDecode if is_decoding else MoeType.DSAsyncPrefill,
        }

        self.mlp.experts.before_dispatch(state)

        # yield for attn1, dis (+share)
        yield
        recv_state = self.mlp.experts.dispatch(state)
        if self.mlp.shared_experts is not None and is_decoding:
            shared_states = self.mlp.shared_experts(hidden_states)
        # yield for dis, dis_wait
        yield
        self.mlp.experts.wait(recv_state)
        # yield for dis_wait, moe
        yield
        gemm_state = self.mlp.experts.gemm(recv_state)
        # yield for moe, comb
        yield
        out_state = self.mlp.experts.combine(gemm_state)
        # yield for comb, (+share) comb_wait
        yield
        if self.mlp.shared_experts is not None and not is_decoding:
            shared_states = self.mlp.shared_experts(hidden_states)
        self.mlp.experts.wait(out_state)
        # yield for (+share) comb_wait, (+share) attn0
        yield
        out_hidden_states = out_state['hidden_states'].view(hidden_shape)
        if shared_states is not None:
            out_hidden_states += shared_states
        elif self.mlp.shared_experts is not None:
            shared_states = self.mlp.shared_experts(hidden_states)
            out_hidden_states += shared_states
        else:
            pass
        out_hidden_states = out_hidden_states.reshape(batch_size, sequence_length, -1)
        outputs = (out_hidden_states, residual)
        return outputs


class DeepseekV2Model(nn.Module):
    """Mixtral model."""

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(config.vocab_size,
                                              config.hidden_size,
                                              self.padding_idx,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

        if get_dist_manager().current_context().dist_config.enable_eplb:
            ep_size_, _ = get_ep_world_rank()
            EPLBManager.init_global_eplb_metadata(ep_size_, config.n_routed_experts, config.num_hidden_layers)
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, quant_config=None, dtype=dtype, device=device)

        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim if getattr(config, 'use_mla', True) else (config.hidden_size //
                                                                                     config.num_attention_heads)
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta

        rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_emb = build_rotary_embedding(**rope_params)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """forward."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def forward_microbatch(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """forward_microbatch."""
        assert self.config.moe_layer_freq == 1
        moe_start_idx = min(self.config.first_k_dense_replace, len(self.layers))

        # embed and mlplayers
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers[:moe_start_idx]):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if moe_start_idx < len(self.layers):
            # run two micro batch
            num = 2
            input_list, exec_type, delta_stages, extern_tag = split_input(hidden_states,
                                                                          rotary_pos_emb,
                                                                          past_key_values,
                                                                          residual,
                                                                          attn_metadata,
                                                                          moe_start_idx,
                                                                          len(self.layers),
                                                                          num=num)

            output_list = execute_batch(inputs=input_list,
                                        fn=self.forward_yieldlayers,
                                        delta_stages=delta_stages,
                                        exec_type=exec_type,
                                        extern_tag=extern_tag)
            hidden_states, residual = merge_output(output_list)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def forward_yieldlayers(self,
                            hidden_states: torch.Tensor,
                            rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                            past_key_values: Optional[List[torch.FloatTensor]] = None,
                            residual: Optional[torch.Tensor] = None,
                            attn_metadata: Any = None,
                            start_idx: int = -1,
                            end_idx: int = -1,
                            tag: Any = None):
        """forward_yieldlayers."""
        for idx in range(start_idx, end_idx):
            past_key_value = past_key_values[idx]
            hidden_states, residual = yield from self.layers[idx].forward_yield(hidden_states,
                                                                                rotary_pos_emb=rotary_pos_emb,
                                                                                past_key_value=past_key_value,
                                                                                residual=residual,
                                                                                attn_metadata=attn_metadata,
                                                                                tag=tag)
        return hidden_states, residual

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class DeepseekV2ForCausalLM(nn.Module, CudaGraphMixin):
    """Mixture model for causalLM."""

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
        self.model = DeepseekV2Model(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)
        self._load_buffers = dict()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if get_step_ctx_manager().current_context().enable_microbatch:
            hidden_states = self.model.forward_microbatch(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds,
            )
        else:
            hidden_states = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds,
            )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
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

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
        """Load weight experts."""
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

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                               update_pe_mapping: List):
        """Load weight attention."""
        device = next(iter(params_dict.values())).device

        def __update_pe(weight, head_dim: int, pe_dim_offset: int):
            # (num_heads, q_head_dim, input_dim)
            weight = weight.unflatten(0, (-1, head_dim))
            # (num_heads, nope_head_dim, input_dim)
            w_pe = weight[:, pe_dim_offset:]
            # (num_heads, nope_head_dim//2, 2, input_dim)
            new_w_pe = w_pe.unflatten(1, (-1, 2))
            # (num_heads, nope_head_dim, input_dim)
            new_w_pe = new_w_pe.transpose(1, 2).flatten(1, 2)
            weight[:, pe_dim_offset:] = new_w_pe
            weight = weight.flatten(0, 1)
            return weight

        def __load_kcvc(name: str, weight: torch.Tensor):
            """Load kc and vc from weight."""
            config = self.config
            v_head_dim = config.v_head_dim
            qk_nope_head_dim = config.qk_nope_head_dim
            w_kc, w_vc = weight.unflatten(0, (-1, qk_nope_head_dim + v_head_dim)).split([qk_nope_head_dim, v_head_dim],
                                                                                        dim=1)
            w_vc = w_vc.transpose(1, 2).contiguous()
            kc_param_name = name.replace('.kv_b_proj', '.kc')
            param_kc = params_dict[kc_param_name]
            load_weight(param_kc, w_kc)
            vc_param_name = name.replace('.kv_b_proj', '.vc')
            param_vc = params_dict[vc_param_name]
            load_weight(param_vc, w_vc)

        def __dequant_weight(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
            """Dequant weight."""
            dim_w0, dim_w1 = weight.shape
            dim_s0, dim_s1 = scale.shape
            assert dim_w0 % dim_s0 == 0
            assert dim_w1 % dim_s1 == 0
            group0 = dim_w0 // dim_s0
            group1 = dim_w1 // dim_s1
            weight = weight.reshape(dim_s0, group0, dim_s1, group1)
            scale = scale.reshape(dim_s0, 1, dim_s1, 1)
            weight = weight.to(scale.dtype) * scale
            weight = weight.to(dtype)
            weight = weight.reshape(dim_w0, dim_w1)
            return weight

        def __load_kcvc_blocked_fp8(name: str, loaded_weight: torch.Tensor):
            """Dequant weight."""
            if name.endswith('.weight'):
                weight_name = name
                scale_name = name.replace('.weight', '.scale')
            elif name.endswith('.weight_scale_inv'):
                weight_name = name.replace('.weight_scale_inv', '.weight')
                scale_name = name
            self._load_buffers[name] = loaded_weight
            if (weight_name in self._load_buffers and scale_name in self._load_buffers):
                weight = self._load_buffers.pop(weight_name)
                scale = self._load_buffers.pop(scale_name)
                kc_param_name = weight_name.replace('.kv_b_proj', '.kc')
                dtype = params_dict[kc_param_name].dtype
                weight = __dequant_weight(weight, scale, dtype)
                __load_kcvc(weight_name, weight)

        for (mod_name, head_dim, pe_dim_offset) in update_pe_mapping:
            if mod_name not in name:
                continue
            if name.endswith('.weight_scale_inv'):
                weight = loaded_weight
            else:
                loaded_weight = loaded_weight.to(device)
                weight = __update_pe(loaded_weight, head_dim, pe_dim_offset)
            param = params_dict[name]
            load_weight(param, weight)
            break
        else:
            if '.kv_b_proj' in name:
                quantization_config = self.quantization_config
                quant_method = None
                if quantization_config is not None:
                    quant_method = quantization_config.get('quant_method')

                loaded_weight = loaded_weight.to(device)
                if quant_method == 'fp8':
                    # update blocked fp8 weight
                    __load_kcvc_blocked_fp8(name, loaded_weight)
                else:
                    __load_kcvc(name, loaded_weight)
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        def __skip_nextn(name, nextn_keys):
            for nextn_key in nextn_keys:
                if nextn_key in name:
                    return True
            return False

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        config = self.config

        update_pe_mapping = []
        if getattr(config, 'use_mla', True):
            qk_rope_head_dim = config.qk_rope_head_dim
            kv_lora_rank = config.kv_lora_rank
            qk_nope_head_dim = config.qk_nope_head_dim
            q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            kv_dim = kv_lora_rank + qk_rope_head_dim
            update_pe_mapping = [('q_proj', q_head_dim, qk_nope_head_dim), ('q_b_proj', q_head_dim, qk_nope_head_dim),
                                 ('kv_a_proj_with_mqa', kv_dim, kv_lora_rank)]
        else:
            # deepseek-vl2-tiny uses MHA LlamaAttention, weight loading differs from MLA
            stacked_params_mapping.extend([
                # (param_name, shard_name, shard_id)
                ('.qkv_proj', '.q_proj', 'q'),
                ('.qkv_proj', '.k_proj', 'k'),
                ('.qkv_proj', '.v_proj', 'v'),
            ])

        num_experts = self.config.n_routed_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        num_hidden_layers = self.config.num_hidden_layers

        num_nextn_predict_layers = getattr(self.config, 'num_nextn_predict_layers', 1)
        nextn_keys = [f'.layers.{num_hidden_layers+i}' for i in range(num_nextn_predict_layers)]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if '.layers' in name:
                # skip nextn
                if __skip_nextn(name, nextn_keys):
                    continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name:
                self._load_weight_experts(name, loaded_weight, params_dict, expert_params_mapping=expert_params_mapping)
            elif '.self_attn' in name and getattr(config, 'use_mla', True):
                # attention
                self._load_weight_attention(name, loaded_weight, params_dict, update_pe_mapping)
            else:
                # other
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
