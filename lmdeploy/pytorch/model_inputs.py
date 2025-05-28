# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal

import torch
from torch.profiler import record_function

# from torch import distributed as dist
import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor


def _broadcast_tensor(value: torch.Tensor, src: int = 0, device: str = 'cuda'):
    """broadcast tensor."""
    if value.device.type == 'meta':
        value = torch.empty_like(value, device=device)
    dist.broadcast(value, src, group=dist.get_tp_group('gpu'))
    return value


@dataclass
class DPMeta:
    tp_sizes: List[int] = None
    ep_sizes: List[int] = None

    @classmethod
    def build(cls, seqlen: int):
        """get dp meta."""
        dist_ctx = dist.get_dist_manager().current_context()

        tp = dist_ctx.tp
        if tp > 1:
            tp_sizes = [None for _ in range(tp)]
            tp_group = dist.get_tp_group('gpu')
            dist.all_gather_object(tp_sizes, seqlen, group=tp_group)
        else:
            tp_sizes = [seqlen]

        return DPMeta(tp_sizes=tp_sizes, )


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
        """to device."""
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

    def broadcast(self):
        """broadcast inputs.

        Do `dist.broadcast_object_list(inputs.to_device('meta'))`
        before broadcast tensors.
        """
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = _broadcast_tensor(v)
            elif k == 'input_embedding_ranges':
                v = [_broadcast_tensor(e) for e in v]
            elif k == 'input_embeddings':
                v = [[_broadcast_tensor(e) for e in li] for li in v]
            elif k == 'input_multimodals':
                new_v = []
                for mm_datas in v:
                    new_mm_datas = dict()
                    for modal_type, data in mm_datas.items():
                        data = [d.broadcast() for d in data]
                        new_mm_datas[modal_type] = data
                    new_v.append(new_mm_datas)
                v = new_v
            out_dict[k] = v

        return VisionModelInputs(**out_dict)

    def get_inputs(self, history_lengths: torch.Tensor, seq_lengths: torch.Tensor):
        """get vision embedding inputs."""
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


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    history_lengths: torch.LongTensor
    block_offsets: torch.LongTensor
    is_decoding: bool
    num_ignored_history: torch.LongTensor
    local_adapter_ids: torch.LongTensor = None
    vision_inputs: VisionModelInputs = None
    cross_length: torch.LongTensor = None
    history_cross_length: torch.LongTensor = None
    model_metas: List[Dict[str, Any]] = None
    dp_meta: 'DPMeta' = None

    def update(self, input_ids: torch.LongTensor):
        """update input ids."""
        assert self.is_decoding
        self.history_lengths = self.history_lengths + 1
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        self.input_ids = input_ids
        return self

    def split(self, split_size: int):
        """split inputs."""
        assert len(self.seq_length) == 1, ('Can not perform split on batched input.')

        input_ids = self.input_ids
        if input_ids.numel() < split_size:
            return self

        flatten_mms = []
        vision_inputs = self.vision_inputs
        if vision_inputs is not None:
            if vision_inputs.input_multimodals is not None:
                input_mms = vision_inputs.input_multimodals[0]

                flatten_mms = []
                for k, mms in input_mms.items():
                    mms = [(k, mm) for mm in mms]
                    flatten_mms += mms

                flatten_mms = sorted(flatten_mms, key=lambda mm: mm[1].start)

        max_seq_len = self.seq_length[0].item()
        ret = []
        start = 0
        history_cross_length = self.history_cross_length
        cross_length = None
        if history_cross_length is not None:
            cross_length = self.history_cross_length.clone()
        while start < max_seq_len:
            vision_inputs = None
            if len(flatten_mms) > 0:
                mm_start = flatten_mms[0][1].start
                mm_end = flatten_mms[0][1].end
                if mm_start > self.history_lengths + start:
                    end = min(mm_start - self.history_lengths, start + split_size)
                else:
                    input_mms = dict()
                    key, mm = flatten_mms.pop(0)
                    input_mms.setdefault(key, [])
                    input_mms[key].append(mm)
                    end = start + mm.end - mm.start
                    while len(flatten_mms) > 0:
                        next_mm = flatten_mms[0]
                        next_start = next_mm[1].start
                        next_end = next_mm[1].end
                        if next_start < mm_end:
                            key = next_mm[0]
                            input_mms.setdefault(key, [])
                            input_mms[key].append(next_mm[1])
                            end += max(0, next_end - mm_end)
                            flatten_mms.pop(0)

                            if cross_length is not None:
                                encoder_len = next_mm[1].encoder_len
                                if encoder_len is not None:
                                    cross_length += encoder_len
                        else:
                            break
                    vision_inputs = VisionModelInputs(input_multimodals=[input_mms], )
            else:
                end = min(max_seq_len, start + split_size)

            inp = ModelInputs(
                input_ids=self.input_ids[:, start:end],
                seq_length=input_ids.new_tensor([end - start]),
                block_offsets=self.block_offsets,
                history_lengths=self.history_lengths + start,
                is_decoding=self.is_decoding,
                num_ignored_history=self.num_ignored_history,
                local_adapter_ids=self.local_adapter_ids,
                vision_inputs=vision_inputs,
                model_metas=self.model_metas,
                cross_length=cross_length,
                history_cross_length=history_cross_length,
            )
            ret.append(inp)
            history_cross_length = cross_length

            start = end

        return ret

    @torch.inference_mode()
    def to_device(self, device: str, non_blocking: bool = False):
        """to device."""
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

    def broadcast(self):
        """broadcast inputs.

        Do `dist.broadcast_object_list(inputs.to_device('meta'))`
        before broadcast tensors.
        """
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = _broadcast_tensor(v)
            elif isinstance(v, VisionModelInputs):
                v = v.broadcast()
            out_dict[k] = v

        return ModelInputs(**out_dict)

    def build_dp_meta(self):
        """build dp meta."""
        self.dp_meta = DPMeta.build(self.input_ids.numel())

    @classmethod
    def make_dummy(cls,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1):
        """make dummy inputs."""
        input_ids = torch.randint(0, vocab_size, (
            1,
            batch_size,
        ), dtype=torch.long, device=device)
        seq_length = torch.ones((batch_size, ), dtype=torch.long, device=device)
        history_lengths = torch.zeros((batch_size, ), dtype=torch.long, device=device)
        block_offsets = torch.full((batch_size, 1), dummy_block_id, dtype=torch.long, device=device)
        num_ignored_history = torch.zeros((batch_size, ), dtype=torch.long, device=device)

        return cls(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
        )

    def log_info(self):
        """get log info."""
        ret = (f'num_tokens={self.input_ids.numel()}, batch_size={self.seq_length.numel()}'
               f', is_decoding={self.is_decoding}, has_vision={self.vision_inputs is not None}')
        return ret


@dataclass
class StepContext:
    """context of Model.

    patched model might need extra information to perform inference. This dataclass provide these infos and tools.
    """
    input_ids: torch.LongTensor
    model_config: ModelConfig
    block_offsets: torch.IntTensor
    position_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    q_seqlens: torch.LongTensor
    kv_seqlens: torch.IntTensor
    q_start_loc: torch.LongTensor
    kv_caches: List
    is_decoding: bool
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

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        model_config: ModelConfig,
        kv_caches: List = None,
        kv_quant_policy: Literal[0, 4, 8] = 0,
    ):
        """build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            device (str): The device of the tensors.
        """
        q_seqlens = inputs.seq_length
        history_seqlens = inputs.history_lengths
        device = q_seqlens.device

        input_multimodals = None
        if inputs.vision_inputs is not None:
            input_multimodals = inputs.vision_inputs.input_multimodals

        # for vlm
        input_embeddings, input_embedding_indexing = None, None
        if (inputs.vision_inputs is not None and inputs.vision_inputs.input_embeddings is not None):
            input_embeddings, input_embedding_indexing = \
                inputs.vision_inputs.get_inputs(history_seqlens, q_seqlens)

        # kv_seqlens
        if inputs.is_decoding:
            attention_mask = torch.ones_like(q_seqlens)[:, None]
            position_ids = history_seqlens.unsqueeze(-1).clone()
        else:
            max_q_seqlen = q_seqlens.max().item()
            mask_range = torch.arange(max_q_seqlen, device=device)[None, :]
            attention_mask = (mask_range < q_seqlens[:, None]).long()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += history_seqlens.unsqueeze(-1)
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens

        # cross
        cross_seqlens = inputs.cross_length
        cross_kv_seqlens = None
        if inputs.cross_length is not None:
            cross_kv_seqlens = (inputs.cross_length + inputs.history_cross_length)

        # position ids 1d
        position_ids = cls.get_position_ids_1d(position_ids, q_seqlens)[None]
        # seq_len + history_length
        kv_seqlens = q_seqlens + history_seqlens
        kv_seqlens -= inputs.num_ignored_history

        ret = StepContext(
            input_ids=inputs.input_ids,
            model_config=model_config,
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
            local_adapter_ids=inputs.local_adapter_ids,
            vision_inputs=inputs.vision_inputs,
            kv_quant_policy=kv_quant_policy,
            model_metas=inputs.model_metas,
            cross_seqlens=cross_seqlens,
            cross_kv_seqlens=cross_kv_seqlens,
            dp_meta=inputs.dp_meta,
        )

        ret = get_backend().update_step_context(ret)
        return ret

    @classmethod
    def get_position_ids_1d(cls, position_ids: torch.LongTensor, seq_length: torch.LongTensor):
        """get 1d position_ids."""
        if position_ids.size(0) == 1 or position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            device = position_ids.device
            position_ids_1d = [ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        return position_ids_1d


class StepContextManager:

    def __init__(self):
        self._current_ctx = None

    @staticmethod
    @record_function('build_step_context')
    def build_context(
        inputs: ModelInputs,
        model_config: ModelConfig,
        kv_caches: List = None,
        kv_quant_policy: Literal[0, 4, 8] = 0,
    ):
        """build context."""
        return StepContext.new(
            inputs,
            model_config,
            kv_caches,
            kv_quant_policy,
        )

    def set_context(self, ctx: StepContext):
        """set context."""
        self._current_ctx = ctx

    @contextmanager
    def context(self, ctx: StepContext):
        """context context."""
        old_ctx = self.current_context()
        self.set_context(ctx)
        yield ctx
        self.set_context(old_ctx)

    def current_context(self):
        """get current_context."""
        return self._current_ctx


_CTX_MANAGER: StepContextManager = None


def set_step_ctx_manager(mgr: StepContextManager):
    global _CTX_MANAGER
    _CTX_MANAGER = mgr
    return mgr


def get_step_ctx_manager():
    """get device manager."""
    return _CTX_MANAGER


@contextmanager
def step_ctx_manager(mgr: StepContextManager):
    old_mgr = _CTX_MANAGER
    set_step_ctx_manager(mgr)
    yield mgr
    set_step_ctx_manager(old_mgr)
