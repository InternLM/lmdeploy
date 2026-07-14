# Copyright (c) OpenMMLab. All rights reserved.
"""ModelLoader: coordinates loading a model's weights into the TurboMind runtime."""
import torch

from .builders._base import Context, ParallelGroup
from .checkpoint import Prefix, create_checkpoint


class ModelLoader:
    """Coordinates loading a model's weights into the TurboMind runtime.

    Holds the model, model_comm handle, and model_path. Extracts GPU topology handles from model_comm and binds them
    onto the model at construction time. Provides export() and export_iter() to load checkpoint weights and commit them
    to the C++ runtime.
    """

    def __init__(self, model, model_comm, gpu_count, model_path,
                 data_type, engine_config):
        self.model = model
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self.data_type = data_type
        self.engine_config = engine_config
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        ctx = Context(
            [mc.context(g) for g in range(self.gpu_count)],
            data_type=self.data_type,
        )
        ec = self.engine_config

        attn_tp = ParallelGroup(ec.attn_tp_size,
                                [mc.attn_tp_rank(g) for g in range(self.gpu_count)])
        mlp_tp = ParallelGroup(ec.mlp_tp_size,
                               [mc.mlp_tp_rank(g) for g in range(self.gpu_count)])
        model_tp = ParallelGroup(ec.attn_tp_size * ec.attn_cp_size,
                                 [mc.model_tp_rank(g) for g in range(self.gpu_count)])

        self.model.bind_runtime(
            ctx=ctx,
            root_handles=[mc.root(g) for g in range(self.gpu_count)],
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            model_tp=model_tp,
        )

    def export(self):
        ckpt = create_checkpoint(
            self.model_path,
            mappings=getattr(self.model, '_loader_mappings', []))
        try:
            self.model.model(Prefix(ckpt))
        finally:
            ckpt.close()
        torch.cuda.empty_cache()

    def export_iter(self):
        ckpt = create_checkpoint(
            self.model_path,
            mappings=getattr(self.model, '_loader_mappings', []))
        try:
            self.model.model(Prefix(ckpt))
            yield -1
        finally:
            ckpt.close()
        torch.cuda.empty_cache()
