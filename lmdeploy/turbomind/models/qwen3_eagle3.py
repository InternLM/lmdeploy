# Copyright (c) OpenMMLab. All rights reserved.
"""EAGLE3 draft weight model for Qwen3 targets."""

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    Eagle3WeightBuilder,
    Eagle3WeightConfig,
    ModuleListBuilder,
    ModuleListConfig,
    TextModelBuilder,
)
from .base import INPUT_MODELS
from .qwen3 import Qwen3TextModel
from .utils import make_model_weight_config


@INPUT_MODELS.register_module(name='qwen3-eagle3')
class Qwen3Eagle3TextModel(Qwen3TextModel):

    def __init__(self, cfg, *, resolver, prefix=''):
        super().__init__(cfg, resolver=resolver)
        self.prefix = prefix
        self.tap_layer_ids = list(cfg.target_layer_ids)

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg,
            self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size,
            root_child='draft_model')
        builder.add_token_embeds(
            pfx.get('embed_tokens.weight'))
        builder.norm = self.norm(pfx + 'norm')
        builder.add_lm_head(
            self._linear(pfx + 'lm_head'))
        builder.layers = self.layers(pfx + 'layers')
        builder.spec = self.spec(pfx)
        builder.build()

    def spec(self, pfx):
        spec = Eagle3WeightBuilder(
            Eagle3WeightConfig(), self._ctx)
        spec.add_target_hidden_proj(
            self._linear(pfx + 'fc'))
        spec.hidden_norms = self.hidden_norms(
            pfx + 'layers')
        return spec.build()

    def hidden_norms(self, pfx):
        norms = ModuleListBuilder(
            ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(
                0, self.cfg.num_hidden_layers):
            norms[i] = self.norm(p + 'hidden_norm')
        return norms.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(
            ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(
                0, self.cfg.num_hidden_layers):
            layer = DecoderLayerBuilder(
                DecoderLayerConfig(), self._ctx)
            layer.attention_norm = self.norm(
                p + 'input_layernorm')
            layer.attention = self.attn(
                p + 'self_attn')
            layer.ffn_norm = self.norm(
                p + 'post_attention_layernorm')
            layer.feed_forward = self.ffn(p + 'mlp', tp=self._mlp_tp)
            layers[i] = layer.build()
        return layers.build()
