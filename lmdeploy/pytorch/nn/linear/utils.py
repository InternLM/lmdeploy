# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.utils import get_logger

from ..utils import get_distribute_size

logger = get_logger('lmdeploy')

QKV_SPLIT_LAYOUTS = ['default', 'hgd']


def check_qkv_split_layout(layout: str):
    if layout not in QKV_SPLIT_LAYOUTS:
        raise RuntimeError(f'Expect qkv split layout in {QKV_SPLIT_LAYOUTS}, '
                           f'but get: {layout}')


def update_tp_args(is_tp: bool, all_reduce: bool, colwise: bool, layer_type: str = 'attn'):
    """Update tp args according to the environment."""
    if is_tp:
        world, _ = get_tp_world_rank(layer_type)
        is_tp = world > 1

    if not is_tp or colwise:
        all_reduce = False

    return is_tp, all_reduce


class QKVMixin:
    """Qkv mixin."""

    def __init__(self,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 num_replicate_kv_heads: int = 1,
                 is_tp: bool = False,
                 tp: int = 1,
                 tp_rank: int = 0):
        qkv_split_section = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v,
                                                       num_replicate_kv_heads)
        num_q_heads, num_kv_heads = self._update_num_heads(is_tp, tp, tp_rank, num_q_heads, num_kv_heads)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.num_replicate_kv_heads = num_replicate_kv_heads
        self.qkv_split_section = qkv_split_section

    def get_qkv_out_feautures(self):
        """Get qkv out features."""
        return self._get_qkv_out_features(self.num_q_heads, self.num_kv_heads, self.head_size, self.head_size_v)

    def _get_qkv_out_features(self,
                              num_q_heads: int,
                              num_kv_heads: int,
                              head_size: int,
                              head_size_v: int,
                              num_replicate_kv_heads: int = 1):
        """Get io features."""
        num_kv_heads_real = num_kv_heads // num_replicate_kv_heads
        all_out_features = (num_q_heads * head_size, num_kv_heads_real * head_size, num_kv_heads_real * head_size_v)
        return all_out_features

    def _update_num_heads(self, is_tp: bool, tp: int, tp_rank: int, num_q_heads: int, num_kv_heads: int):
        """Update num heads."""
        if not is_tp:
            return num_q_heads, num_kv_heads
        world_size, rank = tp, tp_rank
        num_q_heads = get_distribute_size(num_q_heads, world_size, rank)
        num_kv_heads = get_distribute_size(num_kv_heads, world_size, rank)

        return num_q_heads, num_kv_heads

    def split_qkv(self, x: torch.Tensor):
        """Split query, key and value."""
        num_q_heads = self.num_q_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        head_size_v = self.head_size_v

        sections = self.all_out_features
        q, k, v = x.split(sections, dim=-1)
        q = q.unflatten(-1, (num_q_heads, head_size))
        k = k.unflatten(-1, (num_kv_heads, head_size))
        v = v.unflatten(-1, (num_kv_heads, head_size_v))
        return q, k, v
