# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import torch
from torch.profiler import record_function

# from torch import distributed as dist
import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import CacheConfig, DLLMConfig, ModelConfig
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.utils import CtxMgrBase, singleton

if TYPE_CHECKING:
    from lmdeploy.pytorch.strategies.base import StrategyFactoryBase


@dataclass
class DPMeta:
    tp_sizes: List[int] = None
    moe_tp_sizes: List[int] = None

    @staticmethod
    def _gather_tp_sizes(tp: int, seqlen: int, dist_ctx: dist.DistContext, layer_type: str):
        """Gather tp size."""
        attn_tp = dist_ctx.dist_config.attn_tp
        if tp > 1 and tp != attn_tp:
            dist_group = dist.get_dist_group(layer_type=layer_type)
            gather_group = dist_group.gpu_gather_group
            rank = gather_group.rank()
            tp_size_tensor = torch.zeros(gather_group.size(), dtype=torch.int32, device='cuda')
            tp_size_tensor[rank].fill_(seqlen)
            dist.all_gather_into_tensor(tp_size_tensor, tp_size_tensor[rank], group=gather_group)
            tp_sizes = tp_size_tensor.tolist()
            assert all(size >= 0 for size in tp_sizes), (f'seqlen: {seqlen}, Invalid tp sizes: {tp_sizes}')
        else:
            tp_sizes = [seqlen]
        return tp_sizes

    @classmethod
    def build(cls, seqlen: int):
        """Get dp meta."""
        dist_ctx = dist.get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config

        mlp_tp = dist_config.mlp_tp
        tp_sizes = cls._gather_tp_sizes(mlp_tp, seqlen, dist_ctx, layer_type='mlp')

        moe_tp = dist_config.moe_tp
        if moe_tp == mlp_tp:
            moe_tp_sizes = tp_sizes
        else:
            moe_tp_sizes = cls._gather_tp_sizes(moe_tp, seqlen, dist_ctx, layer_type='moe')

        return DPMeta(tp_sizes=tp_sizes, moe_tp_sizes=moe_tp_sizes)

    def sync_tp_size(self, tp_size: int):
        self.tp_sizes = [tp_size] * len(self.tp_sizes)
        self.moe_tp_sizes = [tp_size] * len(self.moe_tp_sizes)


@dataclass
class VisionModelInputs:
    """Vision model inputs."""
    history_lengths: torch.LongTensor = None
    history_image_nums: torch.LongTensor = None
    history_image_token_lengths: torch.LongTensor = None
    input_embeddings: List[List[torch.Tensor]] = None
    input_embedding_ranges: List[torch.LongTensor] = None
    input_embedding_indexing: torch.BoolTensor = None
    input_multimodals: List[MultiModalTensor] = None

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=non_blocking)
            elif k == 'input_embedding_ranges':
                v = [e.to(device, non_blocking=non_blocking) for e in v]
            elif k == 'input_embeddings':
                v = [[e.to(device, non_blocking=non_blocking) for e in li] for li in v]
            elif k == 'input_multimodals':
                new_v = []
                for mm_datas in v:
                    new_mm_datas = dict()
                    for modal_type, data in mm_datas.items():
                        data = [d.to_device(device, non_blocking=non_blocking) for d in data]
                        new_mm_datas[modal_type] = data
                    new_v.append(new_mm_datas)
                v = new_v
            out_dict[k] = v

        return VisionModelInputs(**out_dict)

    def get_inputs(self, history_lengths: torch.Tensor, seq_lengths: torch.Tensor):
        """Get vision embedding inputs."""
        input_embeddings = None
        input_embedding_indexing = None
        if self.input_embeddings is not None and len(self.input_embeddings) > 0:
            input_embedding_li = []
            for (his_len, seq_len, embeddings, emb_ranges) in zip(history_lengths, seq_lengths, self.input_embeddings,
                                                                  self.input_embedding_ranges):
                for emb, (emb_start, emb_end) in zip(embeddings, emb_ranges):
                    start = max(emb_start, his_len) - emb_start
                    end = min(emb_end, his_len + seq_len) - emb_start
                    if 0 <= start < end:
                        input_embedding_li.append(emb[start:end])
            # has embeddings
            if len(input_embedding_li) > 0:
                input_embeddings = torch.cat(input_embedding_li, dim=0)
                device = input_embeddings.device
                starts = history_lengths - self.history_lengths
                ends = starts + seq_lengths
                input_embedding_indexing = torch.cat(
                    [indexing[s:e] for indexing, s, e in zip(self.input_embedding_indexing, starts, ends)], dim=0)
                index_ranges = torch.arange(input_embedding_indexing.numel(), device=device)
                input_embedding_indexing = index_ranges[input_embedding_indexing]
        return input_embeddings, input_embedding_indexing


def get_flatten_multimodals(vision_inputs: VisionModelInputs):
    """Get flatten multimodals."""
    # ignore if vision inputs is None
    if vision_inputs is None:
        return []

    # ignore if input_multimodals is not valid
    input_multimodals = vision_inputs.input_multimodals
    if input_multimodals is None or len(input_multimodals) == 0:
        return []

    # inputs_mms is a dict with type/data_list
    # flatten it to a list of (type, data)
    input_mms = vision_inputs.input_multimodals[0]
    flatten_mms = []
    for k, mms in input_mms.items():
        mms = [(k, mm) for mm in mms]
        flatten_mms += mms

    # sort by start time
    flatten_mms = sorted(flatten_mms, key=lambda mm: mm[1].start)
    return flatten_mms


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    history_lengths: torch.LongTensor
    block_offsets: torch.LongTensor
    is_decoding: bool
    num_ignored_history: torch.LongTensor
    max_q_seqlen: int
    max_kv_seqlen: int
    sum_kv_seqlen: int
    local_adapter_ids: torch.LongTensor = None
    vision_inputs: VisionModelInputs = None
    cross_length: torch.LongTensor = None
    history_cross_length: torch.LongTensor = None
    model_metas: List[Dict[str, Any]] = None
    dp_meta: 'DPMeta' = None
    enable_microbatch: bool = False
    is_dummy: bool = False
    state_offsets: torch.LongTensor = None
    target_hidden_states: torch.Tensor = None
    target_position_ids: torch.Tensor = None

    def step(self, input_ids: torch.LongTensor, step_seqlens: torch.Tensor = None):
        """Update input ids."""
        assert self.is_decoding
        if step_seqlens is None:
            step_seqlens = self.seq_length
        self.history_lengths += step_seqlens
        self.max_kv_seqlen += self.max_q_seqlen
        self.sum_kv_seqlen += self.max_q_seqlen * self.seq_length.numel()
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        self.input_ids = input_ids
        return self

    def split(self, split_size: int):
        """Split inputs."""

        def __add_overlapped_multimodal(flatten_mms: List, input_mms: Dict, end: int, mm_end: int):
            """Add overlapped multimodal data."""
            nonlocal cross_length
            while len(flatten_mms) > 0:
                next_mm = flatten_mms[0]
                next_start = next_mm[1].start
                next_end = next_mm[1].end

                # if next multimodal data is not in the current split, break
                if next_start >= mm_end:
                    break

                key = next_mm[0]
                input_mms.setdefault(key, [])
                input_mms[key].append(next_mm[1])
                end += max(0, next_end - mm_end)
                flatten_mms.pop(0)

                # for mllama
                if cross_length is not None:
                    encoder_len = next_mm[1].encoder_len
                    if encoder_len is not None:
                        cross_length += encoder_len
            return input_mms, end

        def __make_next_vision_inputs(flatten_mms: List, start: int):
            """Make vision inputs."""
            assert len(flatten_mms) > 0

            # start/end of first multimodal data
            mm_start = flatten_mms[0][1].start
            mm_end = flatten_mms[0][1].end

            # when split vision inputs, we require multimodal data should be
            # the start of the split
            # tttvvv... would be split to ttt|vvv...
            if mm_start > self.history_lengths + start:
                end = min(mm_start - self.history_lengths, start + split_size)
                return None, end

            # split by first multimodal data
            key, mm = flatten_mms.pop(0)
            input_mms = {key: [mm]}
            end = start + mm.end - mm.start

            # try add multimodal data between mm_start and mm_end
            # we have not found any model with this pattern yet
            # so basically, nothing would changed
            input_mms, end = __add_overlapped_multimodal(flatten_mms, input_mms, end, mm_end)
            vision_inputs = VisionModelInputs(input_multimodals=[input_mms], )
            return vision_inputs, end

        assert len(self.seq_length) == 1, ('Can not perform split on batched input.')

        input_ids = self.input_ids
        if input_ids.numel() < split_size:
            return self

        flatten_mms = get_flatten_multimodals(self.vision_inputs)

        max_seq_len = self.seq_length[0].item()
        ret = []
        start = 0
        max_kv_seqlen = self.max_kv_seqlen - self.max_q_seqlen

        # for mllama
        history_cross_length = self.history_cross_length
        cross_length = None
        if history_cross_length is not None:
            cross_length = self.history_cross_length.clone()
        while start < max_seq_len:
            if len(flatten_mms) > 0:
                vision_inputs, end = __make_next_vision_inputs(flatten_mms, start)
            else:
                vision_inputs = None
                end = min(max_seq_len, start + split_size)

            max_q_seqlen = end - start
            if isinstance(max_q_seqlen, torch.Tensor):
                max_q_seqlen = max_q_seqlen.item()
            max_kv_seqlen += max_q_seqlen
            target_hidden_states = self.target_hidden_states[:, start:
                                                             end] if self.target_hidden_states is not None else None
            target_position_ids = self.target_position_ids[:,
                                                           start:end] if self.target_position_ids is not None else None
            inp = ModelInputs(
                input_ids=self.input_ids[:, start:end],
                seq_length=input_ids.new_tensor([end - start]),
                block_offsets=self.block_offsets,
                history_lengths=self.history_lengths + start,
                is_decoding=self.is_decoding,
                num_ignored_history=self.num_ignored_history,
                max_q_seqlen=max_q_seqlen,
                max_kv_seqlen=max_kv_seqlen,
                sum_kv_seqlen=max_kv_seqlen,
                local_adapter_ids=self.local_adapter_ids,
                vision_inputs=vision_inputs,
                model_metas=self.model_metas,
                cross_length=cross_length,
                history_cross_length=history_cross_length,
                state_offsets=self.state_offsets,
                target_hidden_states=target_hidden_states,
                target_position_ids=target_position_ids,
            )
            ret.append(inp)
            history_cross_length = cross_length

            start = end

        return ret

    @torch.inference_mode()
    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=non_blocking)
            elif isinstance(v, VisionModelInputs):
                v = v.to_device(device, non_blocking=non_blocking)
            out_dict[k] = v

        return ModelInputs(**out_dict)

    def build_dp_meta(self):
        """Build dp meta."""
        self.dp_meta = DPMeta.build(self.input_ids.numel())

    def log_info(self):
        """Get log info."""
        ret = (f'num_tokens={self.input_ids.numel()}, batch_size={self.seq_length.numel()}'
               f', is_decoding={self.is_decoding}, has_vision={self.vision_inputs is not None}')
        return ret


@dataclass
class StepContext:
    """Context of Model.

    patched model might need extra information to perform inference. This dataclass provide these infos and tools.
    """
    input_ids: torch.LongTensor
    model_config: ModelConfig
    cache_config: CacheConfig
    block_offsets: torch.IntTensor
    position_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    q_seqlens: torch.LongTensor
    kv_seqlens: torch.IntTensor
    q_start_loc: torch.LongTensor
    kv_caches: List
    is_decoding: bool
    sum_kv_seqlen: int
    max_kv_seqlen: int = None
    local_adapter_ids: torch.LongTensor = None
    input_embeddings: torch.Tensor = None
    input_embedding_indexing: torch.Tensor = None
    input_multimodals: List[MultiModalTensor] = None
    vision_inputs: VisionModelInputs = None
    attn_metadata: Any = None
    cross_seqlens: torch.LongTensor = None
    cross_kv_seqlens: torch.LongTensor = None
    cross_attn_metadata: Any = None
    kv_quant_policy: Literal[0, 4, 8] = 0
    model_metas: List[Dict[str, Any]] = None
    dp_meta: DPMeta = None
    enable_microbatch: bool = False
    # for draft model
    target_hidden_states: torch.Tensor = None

    # states for ssm
    state_caches: List = None
    state_offsets: torch.LongTensor = None

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        kv_caches: List = None,
        state_caches: List = None,
        kv_quant_policy: Literal[0, 4, 8] = 0,
    ):
        """Build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            device (str): The device of the tensors.
        """
        q_seqlens = inputs.seq_length
        history_seqlens = inputs.history_lengths

        input_multimodals = None
        if inputs.vision_inputs is not None:
            input_multimodals = inputs.vision_inputs.input_multimodals

        # for vlm
        input_embeddings, input_embedding_indexing = None, None
        if (inputs.vision_inputs is not None and inputs.vision_inputs.input_embeddings is not None):
            input_embeddings, input_embedding_indexing = \
                inputs.vision_inputs.get_inputs(history_seqlens, q_seqlens)

        # position ids
        attention_mask, position_ids = cls.get_mask_and_position_ids(inputs)
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens

        # cross
        cross_seqlens = inputs.cross_length
        cross_kv_seqlens = None
        if inputs.cross_length is not None:
            cross_kv_seqlens = (inputs.cross_length + inputs.history_cross_length)

        # seq_len + history_length
        kv_seqlens = q_seqlens + history_seqlens
        kv_seqlens -= inputs.num_ignored_history

        ret = StepContext(
            input_ids=inputs.input_ids,
            model_config=model_config,
            cache_config=cache_config,
            block_offsets=inputs.block_offsets,
            position_ids=position_ids,
            input_embeddings=input_embeddings,
            input_embedding_indexing=input_embedding_indexing,
            input_multimodals=input_multimodals,
            attention_mask=attention_mask,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            q_start_loc=q_start_loc,
            kv_caches=kv_caches,
            is_decoding=inputs.is_decoding,
            sum_kv_seqlen=inputs.sum_kv_seqlen,
            max_kv_seqlen=inputs.max_kv_seqlen,
            local_adapter_ids=inputs.local_adapter_ids,
            vision_inputs=inputs.vision_inputs,
            kv_quant_policy=kv_quant_policy,
            model_metas=inputs.model_metas,
            cross_seqlens=cross_seqlens,
            cross_kv_seqlens=cross_kv_seqlens,
            dp_meta=inputs.dp_meta,
            enable_microbatch=inputs.enable_microbatch,
            state_caches=state_caches,
            state_offsets=inputs.state_offsets,
            target_hidden_states=inputs.target_hidden_states,
        )

        ret = get_backend().update_step_context(ret)
        return ret

    @classmethod
    def get_mask_and_position_ids(cls, inputs: ModelInputs):
        """Get position ids."""
        q_seqlens = inputs.seq_length
        history_seqlens = inputs.history_lengths
        max_q_seqlen = inputs.max_q_seqlen
        target_position_ids = inputs.target_position_ids
        # decoding
        if max_q_seqlen == 1:
            attention_mask = torch.ones_like(q_seqlens)[:, None]
            if target_position_ids is not None:
                position_ids = target_position_ids
            else:
                position_ids = history_seqlens.unsqueeze(0).clone()
            return attention_mask, position_ids

        num_tokens = inputs.input_ids.numel()
        batch_size = inputs.seq_length.numel()
        device = q_seqlens.device

        # batch with same seqlens
        if max_q_seqlen * batch_size == num_tokens:
            attention_mask = None
            ranges = torch.arange(0, max_q_seqlen, device=device)
            position_ids = history_seqlens[:, None] + ranges[None, :]
            position_ids = position_ids.flatten()
            return attention_mask, position_ids[None]

        # get mask
        mask_range = torch.arange(max_q_seqlen, device=device)[None, :]
        attention_mask = (mask_range < q_seqlens[:, None]).long()
        if target_position_ids is not None:
            return attention_mask, target_position_ids

        # position_ids
        indices = attention_mask.long().cumsum(-1) - 1
        position_ids = indices + history_seqlens.unsqueeze(-1)
        indices[1:] += q_seqlens.cumsum(0)[:-1, None]
        position_ids_1d = position_ids.new_empty(num_tokens)
        position_ids_1d[indices.flatten()] = position_ids.flatten()
        position_ids = position_ids_1d[None]
        return attention_mask, position_ids


@dataclass
class BuildModelContext:
    """Context for building model."""
    disable_vision_encoder: bool = False
    dllm_config: DLLMConfig = None
    strategy_factory: 'StrategyFactoryBase' = None
    enable_return_routed_experts: bool = False


class StepContextManager(CtxMgrBase[StepContext]):

    def __init__(self, build_ctx: BuildModelContext = None):
        super().__init__(None)
        build_ctx = build_ctx or BuildModelContext()
        self.build_ctx = build_ctx

    @record_function('build_step_context')
    def build_context(
        self,
        inputs: ModelInputs,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        kv_caches: List = None,
        state_caches: List = None,
        kv_quant_policy: Literal[0, 4, 8] = 0,
    ):
        """Build context."""
        return StepContext.new(
            inputs,
            model_config,
            cache_config,
            kv_caches,
            state_caches,
            kv_quant_policy,
        )


@singleton
class StepCtxMgrApi(CtxMgrBase[StepContextManager]):
    """Context manager for StepContextManager."""

    def __init__(self):
        super().__init__(None)


set_step_ctx_manager = StepCtxMgrApi().set_context
get_step_ctx_manager = StepCtxMgrApi().current_context
step_ctx_manager = StepCtxMgrApi().context
