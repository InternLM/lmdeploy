# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Iterable, List, Optional, Tuple

import torch

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.models.patch import get_build_model_context
from lmdeploy.pytorch.nn.embedding import ParallelEmbedding
from lmdeploy.pytorch.nn.linear import build_rowwise_linear


class DeployModelMixin:

    def forward(self, *args, **kwargs):
        """Forward of model."""
        raise NotImplementedError('Not Implemented')

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        raise NotImplementedError('Not Implemented')

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        raise NotImplementedError('Not Implemented')

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return hidden_states

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        return name

    def update_weights(self):
        """Update weights."""
        pass

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """Update model meta."""
        return None

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return None


class DeployModelMixinV1(DeployModelMixin):

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        head_dtype = self.get_lm_head().weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(dtype=head_dtype)
        hidden_states = self.get_lm_head()(hidden_states)
        return hidden_states

    def get_lm_head(self):
        """Get lm_head."""
        return self.lm_head

    def get_input_embeddings(self):
        """Get embeds."""
        raise NotImplementedError('Not Implemented')

    def update_weights(self):
        """Update weights."""
        if getattr(self.config, 'tie_word_embeddings', False):
            self.get_lm_head().weight = self.get_input_embeddings().weight

    def build_lm_head(self,
                      hidden_size: int,
                      vocab_size: int,
                      bias: bool = False,
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None,
                      **kwargs):
        """Build LM Head."""
        bm_ctx = get_build_model_context()
        head_dtype = torch.float32 if bm_ctx.enforce_fp32_head else dtype
        lm_head = build_rowwise_linear(
            hidden_size,
            vocab_size,
            bias,
            dtype=head_dtype,
            device=device,
            **kwargs,
        )
        return lm_head


def vlm_model(vlm_cls):
    if not issubclass(vlm_cls, torch.nn.Module):
        raise ValueError('Only subclasses of nn.Module can be decorated with @vlm_model.')

    @functools.wraps(vlm_cls)
    def wrapper(*args, **kwargs):
        bm_ctx = get_build_model_context()
        disable_vision_encoder = bm_ctx.disable_vision_encoder
        if disable_vision_encoder:
            mod = torch.nn.Identity()
            mod._is_dummy_mod = True
            return mod
        else:
            return vlm_cls(*args, **kwargs)

    return wrapper


def build_embedding(vocab_size: int,
                    hidden_size: int,
                    padding_idx: int,
                    dtype: torch.dtype = None,
                    device: torch.device = None,
                    is_tp: bool = False,
                    **kwargs):
    """Build embedding."""
    bm_ctx = get_build_model_context()

    force_dtype = torch.float32 if bm_ctx.enforce_fp32_head else None
    return ParallelEmbedding(
        vocab_size,
        hidden_size,
        padding_idx,
        dtype=dtype,
        device=device,
        is_tp=is_tp,
        force_dtype=force_dtype,
        **kwargs,
    )
