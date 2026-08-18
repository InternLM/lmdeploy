# Copyright (c) OpenMMLab. All rights reserved.
"""StateDictLoader: queue-driven loader used by update_params.

The disk-backed Safetensors / Pytorch paths now live in
``lmdeploy.turbomind.checkpoint``. This module survives only because
``update_params`` (online RL weight updates) hands a Queue to the
loader factory and consumes per-layer chunks. update_params is
currently broken (see spec out-of-scope notes) and will be revisited
as a separate workstream.
"""
from __future__ import annotations

import re
from queue import Queue

import torch


class StateDictLoader:
    """This loader is used for `update_params`.

    Currently, the item in the queue should be full state dict of a decoder layer or the meta of the model (embedding,
    lm_head, norm).
    """

    def __init__(self, queue: Queue, pattern=None,
                 mappings: list | None = None):
        self.que = queue
        self.pattern = pattern

    def items(self):
        for data in iter(self.que.get, None):
            match = []
            if self.pattern:
                for k in data.keys():
                    match = re.findall(self.pattern, k)
                    break
            if not match:
                yield (-1, data)
            else:
                idx = int(match[0])
                yield (idx, data)
            torch.cuda.empty_cache()
            self.que.task_done()

    def all_items(self) -> dict:
        raise NotImplementedError(
            'StateDictLoader does not support all_items()')


def create_loader(model_path, pattern=None, mappings=None):
    """Legacy factory; only the Queue path survives.

    The disk-backed paths moved to :func:`lmdeploy.turbomind.checkpoint.create_checkpoint`.
    """
    if isinstance(model_path, Queue):
        return StateDictLoader(model_path, pattern, mappings)
    raise RuntimeError(
        f'create_loader() no longer supports paths; use '
        f'lmdeploy.turbomind.checkpoint.create_checkpoint instead. '
        f'Got: {type(model_path).__name__}')
