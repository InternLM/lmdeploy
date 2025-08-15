# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MixtralReader(LlamaReader):

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
        return self.params.get(f'model.layers.{i}.block_sparse_moe.gate.{kind}')


@INPUT_MODELS.register_module(name='mixtral')
class MixtralModel(LlamaModel):

    Reader = MixtralReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['expert_num'] = cfg['num_local_experts']
        info['expert_inter_size'] = cfg['intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        info['norm_topk_prob'] = True
        info['inter_size'] = 0
        return info
