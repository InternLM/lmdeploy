# Copyright (c) OpenMMLab. All rights reserved.

import re

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class InternLM2Reader(LlamaReader):
    """InternLM2 model reader."""

    attn_layer_prefix = 'model.layers'
    attn_layer_patten = r'model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    attn_pattern = r'attention'
    ffn_pattern = r'feed_forward'

    proj_pattern = 'w'

    def filter(self, pattern: str, i: int | None):
        params = []
        for k in self.params.keys():
            if re.search(pattern, k):
                params.append(k)

        if self.fp8_quant and pattern == self.attn_pattern:
            from lmdeploy.lite.quantization.weight.quant_utils import quant_blocked_fp8
            q, k, v = (None, ) * 3
            kv_head_num = self.model_cfg['num_key_value_heads']
            gs = int(self.model_cfg['num_attention_heads'] / kv_head_num)
            qkv = self.params.get(f'{self.attn_layer_prefix}.{i}.attention.wqkv.weight')

            if qkv is not None:
                qkv = qkv.view(kv_head_num, gs + 2, 128, -1)
                hidden_dim = qkv.shape[-1]
                q, k, v = torch.split(qkv, [gs, 1, 1], dim=1)

                tensors = [q.reshape(-1, hidden_dim), k.reshape(-1, hidden_dim), v.reshape(-1, hidden_dim)]
                split_sizes = [gs, 1, 1]
                keys = ['q', 'k', 'v']
                qkv_weight = []
                for tensor, split_size, key in zip(tensors, split_sizes, keys):
                    qweight, scale = quant_blocked_fp8(tensor, torch.float8_e4m3fn, block_size=128)
                    qweight = qweight.reshape(kv_head_num, split_size, 128, -1)
                    qkv_weight.append(qweight)

                    self.params[f'{self.attn_layer_prefix}.{i}.{self.attn_pattern}.w{key}.weight_scale_inv'] = scale
                    params.append(f'{self.attn_layer_prefix}.{i}.{self.attn_pattern}.w{key}.weight_scale_inv')

                qkv_weight = torch.cat(qkv_weight, dim=1)
                qkv_weight = qkv_weight.reshape(-1, hidden_dim)
                self.params[f'{self.attn_layer_prefix}.{i}.{self.attn_pattern}.wqkv.weight'] = qkv_weight

            return params
        else:
            return params

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        if self.fp8_quant and kind == 'weight_scale_inv':
            result = []
            for key in ['q', 'k', 'v', 'o']:
                tensor = self.params.get(f'{self.attn_layer_prefix}.{i}.{self.attn_pattern}.w{key}.{kind}')
                tensor = self.transform(tensor, kind)
                result.append(tensor)
            return (*result, )
        q, k, v = (None, ) * 3
        kv_head_num = self.model_cfg['num_key_value_heads']
        gs = int(self.model_cfg['num_attention_heads'] / kv_head_num)
        qkv = self.params.get(f'{self.attn_layer_prefix}.{i}.attention.wqkv.{kind}')
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            qkv = qkv.view(kv_head_num, gs + 2, 128, -1)
            hidden_dim = qkv.shape[-1]
            q, k, v = torch.split(qkv, [gs, 1, 1], dim=1)
            q = q.reshape(-1, hidden_dim)
            k = k.reshape(-1, hidden_dim)
            v = v.reshape(-1, hidden_dim)
        o = self.params.get(f'{self.attn_layer_prefix}.{i}.attention.wo.{kind}')
        o = self.transform(o, kind)
        return (q, k, v, o)

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.attention_norm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[f'{self.attn_layer_prefix}.{i}.feed_forward.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.ffn_norm.weight']


@INPUT_MODELS.register_module(name='internlm2')
class InternLM2Model(LlamaModel):
    """InternLM2 model in hf format."""

    Reader = InternLM2Reader
