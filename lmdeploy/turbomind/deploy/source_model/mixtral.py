# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama import LlamaReader, LlamaModel

class MixtralReader(LlamaReader):

    def moe_ffn_expert(self, e, i):
        result = []
        for k in ['w1', 'w2', 'w3']:
            name = f'model.layers.{i}.block_sparse_moe.experts.{e}.{k}.weight'
            tensor = self.params.get(name)
            tensor = self.transform(tensor, 'weight')
            result.append(tensor)
        return (*result, )
    
    # This is only used to obtain `inter_size` in model config
    def ffn(self, i):
        return self.moe_ffn_expert(0, i)
        
    def moe_ffn_gate(self, i):
        return self.params.get(
            f'model.layers.{i}.block_sparse_moe.gate.weight')
    

@INPUT_MODELS.register_module(name='mixtral')
class MixtralModel(LlamaModel):

    Reader = MixtralReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['expert_num'] = cfg['num_local_experts']
        info['expert_inter_size'] = cfg['intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        return info
