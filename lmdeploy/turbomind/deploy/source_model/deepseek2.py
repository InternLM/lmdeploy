# Copyright (c) OpenMMLab. All rights reserved.
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class DeepSeek2Reader(LlamaReader):
    
    def moe_ffn_gate(self, i):
        return self.params.get(f'model.layers.{i}.mlp.gate.weight')
    
    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        for key in ['gate', 'down', 'up']:
            name = f'model.layers.{i}.mlp.experts.{e}.{key}_proj.{kind}'
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )
    
    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(r'mlp' if i == 0 else r'shared_expert\.')
        result = []
        for key in ['gate', 'down', 'up']:
            name = f'model.layers.{i}.mlp.shared_experts.{key}_proj.{kind}'
            if i == 0:
                name = name.replace('shared_experts.', '')
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )
    
    def mla(self, i: int, kind: str):
        if not kind:
            return self.filter(r'self_attn.*proj')
        result = []
        for key in ['q_a_proj', 'q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}.self_attn.{key}.{kind}'
            )
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )
    
    def mla_norm(self, i: int):
        result = []
        for k in ['q', 'kv']:
            result.append(self.params.get(f'{self.attn_layer_prefix}.{i}.self_attn.{k}_a_layernorm.weight'))
        return (*result, )
        

@INPUT_MODELS.register_module(name='deepseek2')
class DeepSeek2Model(LlamaModel):
    
    Reader = DeepSeek2Reader

    def tokenizer_info(self):
        n_words = self.model_config['vocab_size']
        bos_id = self.model_config['bos_token_id']
        eos_id = self.model_config['eos_token_id']
        return n_words, bos_id, eos_id

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        qk_nope_dim = cfg['qk_nope_head_dim']
        qk_rope_dim = cfg['qk_rope_head_dim']
        num_layer = cfg['num_hidden_layers']
        expert_num = cfg['n_routed_experts']
        expert_num = [expert_num] * num_layer
        expert_num[0] = 0
        n_shared_experts = cfg['n_shared_experts']
        expert_inter_size = cfg['moe_intermediate_size']
        experts_per_token = cfg['num_experts_per_tok']
        inter_size = [n_shared_experts * expert_inter_size] * num_layer
        inter_size[0] = cfg['intermediate_size']
        norm_topk_prob = cfg['norm_topk_prob']
        info.update(
            kv_lora_rank=cfg['kv_lora_rank'],
            q_lora_rank=cfg['q_lora_rank'] or 0,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=cfg['v_head_dim'],
            size_per_head=qk_rope_dim + qk_nope_dim,
            rotary_embedding=qk_rope_dim,
            expert_num=expert_num,
            expert_inter_size=expert_inter_size,
            experts_per_token=experts_per_token,
            inter_size=inter_size,
            norm_topk_prob=norm_topk_prob,
            tune_layer_num=2
        )
        return info

