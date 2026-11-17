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

    def spec(self, pfx):
        spec = Dflash2WeightBuilder(Dflash2WeightConfig(), self._ctx)
        spec.tp = self._model_tp
        spec.add_fc(self._linear(pfx + 'fc'))
        spec.hidden_norm = self.norm(pfx + 'hidden_norm')
        spec.attention_convs = self.convs(pfx + 'layers', 'attention_conv')
        spec.mlp_convs = self.convs(pfx + 'layers', 'mlp_conv')
        spec.selector = self.selector(pfx + 'candidate_selector')
        return spec.build()

    def convs(self, pfx, name):
        convs = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            c = GroupedConvWeightBuilder(GroupedConvWeightConfig(), self._ctx)
            c.add_kernel_projection(self._linear((p + name) + 'kernel_projection'))
            c.add_base_kernel((p + name).pop('base_kernel'))
            convs[i] = c.build()
        return convs.build()

    def selector(self, pfx):
        s = CandidateSelectorWeightBuilder(CandidateSelectorWeightConfig(), self._ctx)
        s.add_hidden_projection(self._linear(pfx + 'hidden_projection'))
        s.add_predecessor_codebook(pfx.pop('predecessor_codebook'))
        s.add_successor_codebook(pfx.pop('successor_codebook'))
        return s.build()
