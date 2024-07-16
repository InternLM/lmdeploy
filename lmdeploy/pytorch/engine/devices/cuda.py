# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.attention import get_attn_backend

from .base_device_utils import BaseDeviceUtils


class CUDADeviceUtils(BaseDeviceUtils):

    device = 'cuda'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        attn_backend = get_attn_backend()
        attn_meta_cls = attn_backend.get_metadata_cls()
        attn_meta = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=step_context.q_start_loc,
            q_seqlens=step_context.q_seq_length,
            kv_seqlens=step_context.kv_seq_length,
            max_q_seqlen=step_context.max_q_seq_length,
            max_kv_seqlen=step_context.max_kv_seq_length,
        )

        step_context.attn_meta = attn_meta
        return step_context
