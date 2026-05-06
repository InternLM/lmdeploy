# Copyright (c) OpenMMLab. All rights reserved.

from ..linear import round_up_output_groups
from ._base import Builder, BuiltModule, ParallelGroup, SplitSide


class TextModelBuilder(Builder):
    """Builder for the root ModelWeight.

    Constructs a ModelWeight via ``_tm.create_module(ModelWeightConfig)``
    on each context (inherited Builder machinery), then attaches it to
    externally-owned ``ModelRoot`` sentinel handles as their
    ``text_model`` child during ``build()``.

    Owns ``tok_embeddings`` (Tensor param) and ``output`` (LinearWeight
    child) commits on the ModelWeight via ``add_token_embeds`` /
    ``add_lm_head``.
    """

    def __init__(self, config, ctx, *, root_handles,
                 tp: ParallelGroup, vocab_size):
        super().__init__(config, ctx)
        self.tp = tp
        self.config.tp_size = tp.size
        self._root_handles = root_handles
        self._vocab_size = vocab_size

    def build(self) -> BuiltModule:
        """Create ModelWeight via _tm.create_module (via super), then attach
        each per-GPU ModelWeight handle to its sentinel root via
        add_child_raw."""
        built = super().build()
        for i, (root, text_model) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._ctx.devices[i]:
                root.add_child_raw('text_model', text_model)
        return built

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the ``tok_embeddings`` root
        param.

        Shards along hidden (output) dim by ``self.tp.size``. No vocab padding —
        embedding lookup never indexes past ``vocab - 1``.
        """
        self._add_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT)

    def add_lm_head(self, linear):
        """Pad output dim to ``round_up(vocab_size, tp)`` and commit to the
        ``output`` LinearWeight root child."""
        linear = round_up_output_groups(linear, self._vocab_size,
                                        self.tp.size)
        self._add_linear('output', linear, split_side=SplitSide.OUTPUT)
