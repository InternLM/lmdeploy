# Copyright (c) OpenMMLab. All rights reserved.
import functools
from collections.abc import Iterable

import torch

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta, StepContext
from lmdeploy.pytorch.models.patch import get_build_model_context
from lmdeploy.pytorch.nn.embedding import ParallelEmbedding
from lmdeploy.pytorch.nn.linear import build_rowwise_linear


class BaseModelMetaProcessor:
    """Model meta processor base class."""

    def update_inputs(self, inputs: ModelInputs, device: torch.device) -> ModelInputs:
        """Update model inputs."""
        return inputs

    def update_delta(self, inputs: ModelInputs, delta: ModelInputsDelta) -> ModelInputs:
        """Update model inputs for delta."""
        return inputs

    def merge(self, inputs: ModelInputs, other: ModelInputs) -> ModelInputs:
        """Merge model inputs with deltas."""
        return inputs


class DeployModelMixin:

    def forward(self, *args, **kwargs):
        """Forward of model."""
        raise NotImplementedError('Not Implemented')

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        raise NotImplementedError('Not Implemented')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""
        raise NotImplementedError('Not Implemented')

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return hidden_states

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Rename weight."""
        return name

    def update_weights(self):
        """Update weights."""
        pass

    def update_model_metas(self,
                           past_key_values: list[list[torch.Tensor]],
                           inputs_embeds: torch.Tensor | None = None,
                           context: StepContext = None):
        """Update model meta."""
        return None

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return None

    def get_modelmeta_processor(self) -> BaseModelMetaProcessor:
        """Get model meta preprocessor."""
        return BaseModelMetaProcessor()

    @classmethod
    def update_quant_config(cls, quant_config: QuantizationConfig):
        """Update quant config."""
        if quant_config is None:
            return
        if getattr(quant_config, 'ignored_layers', None) is None:
            return quant_config
        ignored_layers = [cls.rename_weight(name) for name in quant_config.ignored_layers]

        added_ignore_layers = set()

        for layer_name in ignored_layers:
            if '.q_proj' in layer_name:
                added_ignore_layers.add(layer_name.replace(
                    '.q_proj',
                    '.qkv_proj',
                ))
            elif '.gate_proj' in layer_name:
                if '.experts' in layer_name:
                    added_ignore_layers.add(layer_name.split('.experts', 1)[0] + '.experts')
                else:
                    added_ignore_layers.add(layer_name.replace('.gate_proj', '.gate_up_proj'))
            elif '.down_proj' in layer_name:
                if '.experts' in layer_name:
                    added_ignore_layers.add(layer_name.split('.experts', 1)[0] + '.experts')
                else:
                    added_ignore_layers.add(layer_name)

        added_ignore_layers = list(added_ignore_layers)

        ignored_layers.extend(added_ignore_layers)
        quant_config.ignored_layers = ignored_layers

        return quant_config


class DeployModelMixinV1(DeployModelMixin):

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        head_dtype = self.lm_head.weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(dtype=head_dtype)
        hidden_states = self.lm_head(hidden_states)
        return hidden_states

    def get_input_embeddings(self):
        """Get embeds."""
        raise NotImplementedError('Not Implemented')

    def update_weights(self):
        """Update weights."""
        if getattr(self.config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.get_input_embeddings().weight

    def build_lm_head(self,
                      hidden_size: int,
                      vocab_size: int,
                      bias: bool = False,
                      dtype: torch.dtype | None = None,
                      device: torch.device | None = None,
                      **kwargs):
        """Build LM Head."""
        bm_ctx = get_build_model_context()
        head_dtype = torch.float32 if bm_ctx.fp32_lm_head else dtype
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

    # run with fp32 only when share weights with lm_head
    force_dtype = None
    if bm_ctx.fp32_lm_head and bm_ctx.tie_word_embeddings:
        force_dtype = torch.float32

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
