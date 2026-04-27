# Copyright (c) OpenMMLab. All rights reserved.
"""ModelLoader: coordinates loading a model's weights into the TurboMind runtime."""
import torch

from .loader import create_loader


class ModelLoader:
    """Coordinates loading a model's weights into the TurboMind runtime.

    Holds the model, model_comm handle, and model_path. Extracts GPU topology handles from model_comm and binds them onto
    the model at construction time. Provides export() and export_iter() to load checkpoint weights and commit them to the
    C++ runtime.
    """

    def __init__(self, model, model_comm, gpu_count, model_path):
        self.model = model
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        attn_ranks = [mc.attn_tp_rank(g) for g in range(self.gpu_count)]
        mlp_ranks = [mc.mlp_tp_rank(g) for g in range(self.gpu_count)]
        model_tp = [mc.model_tp_rank(g) for g in range(self.gpu_count)]
        contexts = [mc.context(g) for g in range(self.gpu_count)]
        handles = [mc.root(g) for g in range(self.gpu_count)]
        self.model.bind_runtime(
            contexts=contexts,
            root_handles=handles,
            attn_ranks=attn_ranks,
            mlp_ranks=mlp_ranks,
            model_tp_ranks=model_tp,
        )

    def export(self):
        loader = create_loader(self.model_path, self.model._layer_pattern,
                               self.model._loader_mappings)
        self.model.set_params(loader.all_items())
        self.model.model()
        torch.cuda.empty_cache()

    def export_iter(self):
        loader = create_loader(self.model_path, self.model._layer_pattern,
                               self.model._loader_mappings)
        self.model.set_params(loader.all_items())
        self.model.model()
        yield -1
        torch.cuda.empty_cache()
