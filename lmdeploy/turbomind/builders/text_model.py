# Copyright (c) OpenMMLab. All rights reserved.

from ._base import Builder, BuiltModule, SplitSide
from ..linear import Linear, pad_out_dim


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

    def __init__(self, config, contexts, *, root_handles,
                 tp, ranks, vocab_size):
        super().__init__(config=config, contexts=contexts, tp=tp, ranks=ranks)
        self._root_handles = root_handles
        self._vocab_size = vocab_size

    def build(self) -> BuiltModule:
        """Create ModelWeight via _tm.create_module (via super), then attach
        each per-GPU ModelWeight handle to its sentinel root via
        add_child_raw."""
        built = super().build()
        for i, (root, text_model) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._contexts[i]:
                root.add_child_raw('text_model', text_model)
        return built

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the ``tok_embeddings`` root
        param.

        Shards along hidden (output) dim by ``self._tp``. No vocab padding —
        embedding lookup never indexes past ``vocab - 1``.
        """
        self._add_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT)

    def add_lm_head(self, linear):
        """Pad output dim to ``round_up(vocab_size, tp)`` and commit to the
        ``output`` LinearWeight root child.

        Works for every checkpoint format in use today — trivial / AWQ /
        GPTQ / compressed-tensors / MXFP4 all have ``block_out is None``,
        so padding every tensor in the bundle along ``dim=-1`` keeps the
        format-specific block structure intact. FP8 ``lm_head``
        (``block_out == 128``) would misalign scales under naive padding
        but is not a configuration used by any released checkpoint.
        """
        padded_vocab = ((self._vocab_size + self._tp - 1)
                        // self._tp) * self._tp
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format)
        self._add_linear('output', padded,
                            split_side=SplitSide.OUTPUT)
