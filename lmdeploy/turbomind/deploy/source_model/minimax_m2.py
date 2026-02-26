# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MiniMaxM2Reader(LlamaReader):

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        for x in ['w1', 'w2', 'w3']:
            name = f'model.layers.{i}.block_sparse_moe.experts.{e}.{x}.{kind}'
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i, kind):
        if kind == 'bias':
            return self.params.get(
                f'model.layers.{i}.block_sparse_moe.e_score_correction_bias')
        return self.params.get(
            f'model.layers.{i}.block_sparse_moe.gate.{kind}')

    def qk_norm(self, i: int):
        result = []
        for x in ['q', 'k']:
            name = f'model.layers.{i}.self_attn.{x}_norm.weight'
            result.append(self.transform(self.params.get(name), 'weight'))
        return (*result, )


@INPUT_MODELS.register_module(name='minimax-m2')
class MiniMaxM2Model(LlamaModel):

    Reader = MiniMaxM2Reader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info.update(
            qk_norm=True,
            qk_norm_type='per_token',
            expert_num=cfg['num_local_experts'],
            expert_inter_size=cfg['intermediate_size'],
            experts_per_token=cfg['num_experts_per_tok'],
            inter_size=0,
            norm_topk_prob=True,
            expert_router_bias=True,
            scoring_func=cfg.get('scoring_func', 'sigmoid'),
        )
        rotary_dim = cfg.get('rotary_dim', None)
        if rotary_dim is not None:
            info['rope_param'].dim = rotary_dim
        return info
