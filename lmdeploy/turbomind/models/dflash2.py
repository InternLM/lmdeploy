# Copyright (c) OpenMMLab. All rights reserved.
"""DFlash2 draft weight model for TurboMind.

The draft is a 5-layer Qwen3-style decoder (non-causal sliding-window attention)
that shares the target's input embeddings and scores candidates through a
low-rank candidate selector. The checkpoint carries no ``embed_tokens``/``lm_head``.

``ModelWeight::spec`` holds the DFlash2 extras (``Dflash2Weight``): the context
projection ``fc``, ``hidden_norm``, per-layer grouped convs and the selector.
The final ``norm`` lives on the draft root like any decoder.
"""

from ..builders import (
    AttentionBuilder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    ModuleListBuilder,
    ModuleListConfig,
    TextModelBuilder,
)
from ..builders.candidate_selector_weight import (
    CandidateSelectorWeightBuilder,
    CandidateSelectorWeightConfig,
)
from ..builders.dflash2_weight import Dflash2WeightBuilder, Dflash2WeightConfig
from ..builders.grouped_conv_weight import (
    GroupedConvWeightBuilder,
    GroupedConvWeightConfig,
)
from .base import INPUT_MODELS
from .qwen3 import Qwen3TextModel
from .utils import make_model_weight_config


@INPUT_MODELS.register_module(name='qwen3-dflash2')
class Qwen3Dflash2TextModel(Qwen3TextModel):

    def __init__(self, cfg, *, resolver, prefix=''):
        super().__init__(cfg, resolver=resolver)
        self.prefix = prefix
        self.tap_layer_ids = list(cfg.dflash_config['target_layer_ids'])

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        # Decoder-only draft: no own tok_embeddings/output (shares the target's).
        root_cfg.decoder_only = True
        builder = TextModelBuilder(
            root_cfg,
            self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size,
            root_child='draft_model')
        builder.norm = self.norm(pfx + 'norm')
        builder.layers = self.layers(pfx + 'layers')
        builder.spec = self.spec(pfx)
        builder.build()

    def layers(self, pfx):
        # Qwen3-style decoder layers plus the per-layer DFlash2 grouped convs.
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.attention = self.attn(p + 'self_attn')
            d.attention_conv = self.conv(p + 'attention_conv')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            d.feed_forward = self.ffn(p + 'mlp', tp=self._mlp_tp)
            d.mlp_conv = self.conv(p + 'mlp_conv')
            layers[i] = d.build()
        return layers.build()

    def attn(self, pfx):
        # The DFlash2 block forward applies a natural-order RoPE, so the draft
        # q/k are kept un-permuted (skip reorder_rotary_emb).
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']
        m = AttentionBuilder(self._attn_cfg.clone(), self._ctx, tp=self._attn_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)
        m.q_norm = self.norm(pfx + 'q_norm')
        m.k_norm = self.norm(pfx + 'k_norm')
        return m.build()

    def conv(self, pfx):
        c = GroupedConvWeightBuilder(GroupedConvWeightConfig(), self._ctx)
        c.add_kernel_projection(self._linear(pfx + 'kernel_projection'))
        c.add_base_kernel(pfx.pop('base_kernel'))
        return c.build()

    def spec(self, pfx):
        spec = Dflash2WeightBuilder(Dflash2WeightConfig(), self._ctx)
        spec.tp = self._model_tp
        spec.add_fc(self._linear(pfx + 'fc'))
        spec.hidden_norm = self.norm(pfx + 'hidden_norm')
        spec.selector = self.selector(pfx + 'candidate_selector')
        return spec.build()

    def selector(self, pfx):
        s = CandidateSelectorWeightBuilder(CandidateSelectorWeightConfig(), self._ctx)
        s.add_hidden_projection(self._linear(pfx + 'hidden_projection'))
        s.add_predecessor_codebook(pfx.pop('predecessor_codebook'))
        s.add_successor_codebook(pfx.pop('successor_codebook'))
        return s.build()
