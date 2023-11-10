# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class QwenReader(LlamaReader):
    """QwenReader."""

    attn_layer_patten = r'transformer.h.([0-9]+).'
    tok_embeddings_key = 'transformer.wte.weight'
    norm_weight_key = 'transformer.ln_f.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        qkv = self.params[f'transformer.h.{i}.attn.c_attn.{kind}']
        q, k, v = torch.split(qkv, qkv.size(size_dim) // 3, dim=dim)
        o = self.params.get(f'transformer.h.{i}.attn.c_proj.{kind}', None)
        if o is None:
            o = torch.zeros_like(q)
        return q, k, v, o

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, 'weight', 0, 0)

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        return self._attn(i, 'bias', -1, 0)

    def attn_zero(self, i: int):
        """Get q, k, v, o zero point for layer i."""
        return (None, ) * 4

    def attn_scale(self, i: int):
        """Get q, k, v, o scale for layer i."""
        return (None, ) * 4

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_1.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w2', 'c_proj', 'w1']:
            tensor = self.params[f'transformer.h.{i}.mlp.{key}.{kind}']
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, 'weight')

    def ffn_zero(self, i: int):
        """Get ffn zero point for layer i."""
        return (None, ) * 3

    def ffn_scale(self, i: int):
        """Get ffn scale for layer i."""
        return (None, ) * 3

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_2.weight']


@INPUT_MODELS.register_module(name='qwen')
class QwenModel(LlamaModel):
    """Qwen model in hf format."""

    Reader = QwenReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    def tokenizer_info(self):
        """Read tokenizer info."""
        n_words = 151851
        bos_id = 0
        eos_id = 151643
        return n_words, bos_id, eos_id

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            config = json.load(f)
            num_layer = config['num_hidden_layers']
            norm_eps = config['layer_norm_epsilon']
            rope_theta = float(config.get('rotary_emb_base', 10000.0))
            if 'num_key_value_heads' in config:
                kv_head_num = config['num_key_value_heads']
            else:
                kv_head_num = config['num_attention_heads']
            seq_length = config['seq_length']
            use_dynamic_ntk = int(config['use_dynamic_ntk'])
            use_logn_attn = int(config['use_logn_attn'])
        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=seq_length,
                    use_dynamic_ntk=int(use_dynamic_ntk),
                    use_logn_attn=use_logn_attn)
