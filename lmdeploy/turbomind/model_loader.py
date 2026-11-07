# Copyright (c) OpenMMLab. All rights reserved.
"""ModelLoader: coordinates loading a model's weights into the TurboMind runtime."""
import torch

from . import _tm
from .builders._base import Context, ParallelGroup
from .checkpoint import Prefix, create_checkpoint


class ModelLoader:
    """Coordinates loading a model's weights into the TurboMind runtime.

    Holds the model, model_comm handle, and model_path. Extracts GPU topology handles from model_comm and binds them
    onto the model at construction time. Provides export() and export_iter() to load checkpoint weights and commit them
    to the C++ runtime.
    """

    def __init__(self,
                 model,
                 model_comm,
                 gpu_count,
                 model_path,
                 data_type,
                 engine_config,
                 *,
                 draft_model=None,
                 draft_model_path=None):
        self.model = model
        self.draft_model = draft_model
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.data_type = data_type
        self.engine_config = engine_config
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        gemm_input_dtype = {
            None: _tm.DataType.TYPE_INVALID,
            'float16': _tm.DataType.TYPE_FP16,
            'bfloat16': _tm.DataType.TYPE_BF16,
            'float8_e4m3': _tm.DataType.TYPE_FP8_E4M3,
        }[self.engine_config.gemm_input_dtype]
        ctx = Context(
            [mc.context(g) for g in range(self.gpu_count)],
            mc.gemm(0),
            data_type=self.data_type,
            gemm_input_dtype=gemm_input_dtype,
        )
        ec = self.engine_config

        attn_tp = ParallelGroup(ec.attn_tp_size,
                                [mc.attn_tp_rank(g) for g in range(self.gpu_count)])
        mlp_tp = ParallelGroup(ec.mlp_tp_size,
                               [mc.mlp_tp_rank(g) for g in range(self.gpu_count)])
        ep = ParallelGroup(ec.ep,
                           [mc.ep_rank(g) for g in range(self.gpu_count)])
        model_tp = ParallelGroup(ec.attn_tp_size * ec.attn_cp_size,
                                 [mc.model_tp_rank(g) for g in range(self.gpu_count)])

        # Dense (non-expert) FFN TP: node-local — one node's ranks within
        # the comm domain (shard index = inner_rank % domain_size,
        # inner_rank = ep_rank * mlp_tp_size + mlp_tp_rank).
        dense_size = min(mlp_tp.size * ep.size, self.gpu_count)
        dense_tp = ParallelGroup(
            dense_size,
            [(e * mlp_tp.size + m) % dense_size
             for e, m in zip(ep.ranks, mlp_tp.ranks)])

        models = (
            (self.model,)
            if self.draft_model is None
            else (self.model, self.draft_model))
        for model in models:
            model.bind_runtime(
                ctx=ctx,
                root_handles=[
                    mc.root(g)
                    for g in range(self.gpu_count)],
                attn_tp=attn_tp,
                mlp_tp=mlp_tp,
                ep=ep,
                model_tp=model_tp,
                dense_tp=dense_tp)

    @staticmethod
    def _export_one(model, model_path):
        checkpoint = create_checkpoint(
            model_path,
            mappings=getattr(
                model, '_loader_mappings', []))
        try:
            root = Prefix(checkpoint)
            if getattr(model, 'prefix', ''):
                root = root + model.prefix
            model.model(root)
        finally:
            checkpoint.close()

    def export(self):
        self._export_one(
            self.model, self.model_path)
        if self.draft_model is not None:
            self._export_one(
                self.draft_model,
                self.draft_model_path)
        torch.cuda.empty_cache()

    def export_iter(self):
        self._export_one(
            self.model, self.model_path)
        yield -1
        if self.draft_model is not None:
            self._export_one(
                self.draft_model,
                self.draft_model_path)
            yield -1
        torch.cuda.empty_cache()
