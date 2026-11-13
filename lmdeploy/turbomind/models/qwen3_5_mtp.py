# Copyright (c) OpenMMLab. All rights reserved.
"""Checkpoint-native Qwen3.5 MTP draft weight tree."""

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    ModuleListBuilder,
    ModuleListConfig,
    Qwen35MtpWeightBuilder,
    Qwen35MtpWeightConfig,
    TextModelBuilder,
)
from .qwen3_5 import Qwen3_5TextModel
from .utils import make_model_weight_config


class Qwen3_5MtpTextModel(Qwen3_5TextModel):

    def __init__(self, cfg, *, resolver):
        super().__init__(cfg, resolver=resolver)
        self.tap_layer_ids = [cfg.num_hidden_layers]

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        root_cfg.decoder_only = True

        root = TextModelBuilder(
            root_cfg,
            self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size,
            root_child='draft_model')

        root.norm = self.norm(pfx + 'mtp.norm', zero_centered=True)
        root.layers = self.mtp_layers(pfx + 'mtp.layers')
        root.spec = self.mtp_spec(pfx + 'mtp')
        root.build()

    def mtp_layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, layer_pfx in pfx.slices(0, 1):
            layer = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            layer.attention = self.attn(layer_pfx + 'self_attn')
            if self._n_experts > 0:
                layer.moe_ffn = self.moe(layer_pfx + 'mlp')
            else:
                layer.feed_forward = self.ffn(layer_pfx + 'mlp',
                                              self.cfg.intermediate_size,
                                              tp=self._mlp_tp)
            layer.attention_norm = self.norm(
                layer_pfx + 'input_layernorm', zero_centered=True)
            layer.ffn_norm = self.norm(
                layer_pfx + 'post_attention_layernorm', zero_centered=True)
            layers[i] = layer.build()
        return layers.build()

    def mtp_spec(self, pfx):
        spec = Qwen35MtpWeightBuilder(Qwen35MtpWeightConfig(), self._ctx)
        spec.tp = self._model_tp
        spec.add_fc(self._linear(pfx + 'fc'))
        spec.pre_fc_norm_embedding = self.norm(
            pfx + 'pre_fc_norm_embedding', zero_centered=True)
        spec.pre_fc_norm_hidden = self.norm(
            pfx + 'pre_fc_norm_hidden', zero_centered=True)
        return spec.build()
