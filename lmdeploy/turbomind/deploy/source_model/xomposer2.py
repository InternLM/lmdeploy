# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Xcomposer2Reader(LlamaReader):
    """Xcomposer2 model reader."""

    attn_layer_patten = r'model.layers.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        kv_head_num = self.model_cfg['kv_head_num']
        gs = int(self.model_cfg['attn_head_num'] / kv_head_num)
        qkv = self.params[f'model.layers.{i}.attention.wqkv.{kind}']
        qkv = qkv.view(kv_head_num, gs + 2, 128, -1)
        hidden_dim = qkv.shape[-1]
        q, k, v = torch.split(qkv, [gs, 1, 1], dim=1)
        q = q.reshape(-1, hidden_dim)
        k = k.reshape(-1, hidden_dim)
        v = v.reshape(-1, hidden_dim)
        o = self.params.get(f'model.layers.{i}.attention.wo.{kind}')
        return q, k, v, o

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, 'weight', 0, 0)

    def attn_lora_a(self, i):
        """Get attn lora_a."""
        qkv = self.params[f'model.layers.{i}.attention.wqkv.Plora_A.weight']
        o = self.params[f'model.layers.{i}.attention.wo.Plora_A.weight']
        return qkv, o

    def attn_lora_b(self, i):
        """Get attn lora_b."""
        return self._attn(i, 'Plora_B.weight', 0, 0)

    def attn_bias(self, i: int):
        return (None, ) * 4

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'model.layers.{i}.attention_norm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[f'model.layers.{i}.feed_forward.{key}.{kind}']
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, 'weight')

    def ffn_lora_a(self, i: int):
        """Get ffn lora_a weight for layer i."""
        return self._ffn(i, 'Plora_A.weight')

    def ffn_lora_b(self, i: int):
        """Get fnn lora_b weight for layer i."""
        return self._ffn(i, 'Plora_B.weight')

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'model.layers.{i}.ffn_norm.weight']


@INPUT_MODELS.register_module(name='xcomposer2')
class Xcomposer2Model(LlamaModel):
    """Xcomposer2 model in hf format."""

    Reader = Xcomposer2Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    def _lora_cfg_7b(self):
        """lora config for internlm-xcomposer2-7b."""
        return dict(lora_r=256, lora_scale=1.0, lora_policy=1)

    def _lora_cfg_4khd_7b(self, model_info: dict):
        """lora config for internlm-xcomposer2-4khd-7b."""
        num_layer = model_info['num_layer']
        rank_pattern = []
        scale_pattern = []
        rank_qkv = 8
        rank_wo = 256
        scale_qkv = 2.0
        scale_wo = 1.0
        for i in range(num_layer):
            for key, rank, scale in zip(['w_qkv', 'wo'], [rank_qkv, rank_wo],
                                        [scale_qkv, scale_wo]):
                rank_pattern.append(f'layers.{i}.attention.{key}:{rank}')
                scale_pattern.append(f'layers.{i}.attention.{key}:{scale}')
        rank_pattern = ','.join(rank_pattern)
        scale_pattern = ','.join(scale_pattern)
        return dict(lora_r=256,
                    lora_scale=1.0,
                    lora_policy=1,
                    lora_rank_pattern=rank_pattern,
                    lora_scale_pattern=scale_pattern)

    def model_info(self):
        out = super().model_info()
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
        if model_arg['max_length'] == 16384:
            out.update(self._lora_cfg_4khd_7b(out))
        else:
            out.update(self._lora_cfg_7b())
        return out
