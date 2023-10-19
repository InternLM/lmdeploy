# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from lmdeploy.turbomind.deploy.source_model.base import (INPUT_MODELS,
                                                         BaseWeightFileMgr)
from lmdeploy.turbomind.deploy.source_model.hf import HfModel


class QwenWeightFileMgr(BaseWeightFileMgr):
    """QwenWeightFileMgr."""

    attn_layer_patten = r'transformer.h.([0-9]+).'

    def __init__(self, new_params: dict, unused_params: dict):
        super().__init__()
        self.params = unused_params
        self.params.update(new_params)
        self.init_layer_id()

    def init_layer_id(self):
        super().init_layer_id()

    def clean_up(self, last: bool) -> None:
        super().clean_up(last)

    @property
    def start_layer_id(self):
        return self._start_layer_id

    @property
    def end_layer_id(self):
        return self._end_layer_id

    def tok_embeddings(self):
        return self.params.get('transformer.wte.weight', None)

    def norm_weight(self):
        return self.params.get('transformer.ln_f.weight', None)

    def output_weight(self):
        return self.params.get('lm_head.weight', None)

    def attn(self, i: int):
        qkv_w = self.params[f'transformer.h.{i}.attn.c_attn.weight']
        qw, kw, vw = torch.split(qkv_w, qkv_w.size(0) // 3, dim=0)
        ow = self.params[f'transformer.h.{i}.attn.c_proj.weight']
        return qw, kw, vw, ow

    def attn_bias(self, i: int):
        qkv_b = self.params[f'transformer.h.{i}.attn.c_attn.bias']
        qb, kb, vb = torch.split(qkv_b, qkv_b.size(-1) // 3)
        ob = torch.zeros_like(qb)
        return qb, kb, vb, ob

    def attn_zero(self, i: int):
        return (None, ) * 4

    def attn_scale(self, i: int):
        return (None, ) * 4

    def attn_norm(self, i: int):
        return self.params[f'transformer.h.{i}.ln_1.weight']

    def ffn(self, i: int):
        result = []
        for key in ['w2', 'c_proj', 'w1']:
            tensor = self.params[f'transformer.h.{i}.mlp.{key}.weight']
            result.append(tensor)
        return (*result, )

    def ffn_zero(self, i: int):
        return (None, ) * 3

    def ffn_scale(self, i: int):
        return (None, ) * 3

    def ffn_norm(self, i: int):
        return self.params[f'transformer.h.{i}.ln_2.weight']


@INPUT_MODELS.register_module(name='qwen')
class QwenModel(HfModel):
    """Qwen model in hf format."""

    WeightFileMgr = QwenWeightFileMgr

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)

    def tokenizer_info(self):
        n_words = 151851
        bos_id = 0
        eos_id = 151643
        return n_words, bos_id, eos_id

    def model_info(self):
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
