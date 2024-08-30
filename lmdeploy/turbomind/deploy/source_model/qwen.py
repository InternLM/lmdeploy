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

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v, o = (None, ) * 4
        qkv = self.params[f'transformer.h.{i}.attn.c_attn.{kind}']
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            q, k, v = torch.split(qkv, qkv.size(0) // 3, dim=0)
        o = self.params.get(f'transformer.h.{i}.attn.c_proj.{kind}')
        o = self.transform(o, kind)
        if o is None:
            o = torch.zeros_like(q)
        return q, k, v, o

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_1.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w2', 'c_proj', 'w1']:
            tensor = self.params[f'transformer.h.{i}.mlp.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_2.weight']


@INPUT_MODELS.register_module(name='qwen')
class QwenModel(LlamaModel):
    """Qwen model in hf format."""

    Reader = QwenReader

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
            hidden_units = config['hidden_size']
            num_layer = config['num_hidden_layers']
            norm_eps = config['layer_norm_epsilon']
            rope_theta = float(config.get('rotary_emb_base', 10000.0))
            if 'num_key_value_heads' in config:
                kv_head_num = config['num_key_value_heads']
            else:
                kv_head_num = config['num_attention_heads']
            attn_head_num = config['num_attention_heads']
            seq_length = config['seq_length']
            use_dynamic_ntk = int(config['use_dynamic_ntk'])
            use_logn_attn = int(config['use_logn_attn'])
        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    hidden_units=hidden_units,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=seq_length,
                    use_dynamic_ntk=int(use_dynamic_ntk),
                    use_logn_attn=use_logn_attn)


@INPUT_MODELS.register_module(name='qwen2')
class Qwen2Model(LlamaModel):
    """Qwen model in hf format.

    The weight of qwen2 model is similar to Llama, except its attention bias
    doesn't include o_proj bias.
    """

    Reader = LlamaReader

    def tokenizer_info(self):
        """set tokenizer info.

        Refer to https://huggingface.co/Qwen/Qwen1.5-7B-Chat/blob/main/generation_config.json
        """  # noqa: E501
        n_words = 152064
        bos_id = 151643
        eos_id = 151645
        return n_words, bos_id, eos_id
