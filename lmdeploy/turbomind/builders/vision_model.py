# Copyright (c) OpenMMLab. All rights reserved.

from ._base import Builder, BuiltModule, ParallelGroup


class VisionModelBuilder(Builder):
    """Generic root builder for a VLM vision sub-graph.

    Counterpart to ``TextModelBuilder``. Attaches the constructed vision
    root weight as the ``visual_model`` sibling of ``text_model`` on each
    per-GPU ``ModelRoot``.
    """

    def __init__(self, config, ctx, *, root_handles, tp: ParallelGroup):
        super().__init__(config, ctx)
        self.tp = tp
        self._root_handles = root_handles

    def build(self) -> BuiltModule:
        built = super().build()
        for i, (root, vision) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._ctx.devices[i]:
                root.add_child_raw('visual_model', vision)
        return built
