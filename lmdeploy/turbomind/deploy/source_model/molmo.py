# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MolmoReader(LlamaReader):
    attn_layer_prefix = 'model.transformer.blocks'
    attn_layer_patten = r'model.transformer.blocks.([0-9]+).'
    norm_weight_key = 'model.transformer.ln_f.weight'
    output_weight_key = 'model.transformer.ff_out.weight'

    def tok_embeddings(self):
        key1 = 'model.transformer.wte.embedding'
        key2 = 'model.transformer.wte.new_embedding'
        embed1 = self.params.get(key1, None)
        embed2 = self.params.get(key2, None)
        if embed1 is not None and embed2 is not None:
            return torch.cat((embed1, embed2), dim=0)
        else:
            assert embed1 is None and embed2 is None
            return None

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.attn_norm.weight']

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind(weight, bias, qweight) for layer i."""
        q, k, v = (None, ) * 3
        hidden_size = self.model_cfg['hidden_size']
        head_num = self.model_cfg['num_attention_heads']
        kv_head_num = self.model_cfg['num_key_value_heads']
        head_dim = hidden_size // head_num
        assert head_dim == 128
        fused_dims = (hidden_size, kv_head_num * head_dim,
                      kv_head_num * head_dim)
        qkv = self.params.get(f'{self.attn_layer_prefix}.{i}.att_proj.{kind}')
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            _q, k, v = qkv.split(fused_dims, dim=0)
        o = self.params.get(f'{self.attn_layer_prefix}.{i}.attn_out.{kind}')
        o = self.transform(o, kind)
        if o is None:  # handle the case when qkv has bias but o doesn't
            o = torch.zeros_like(_q)
        return (_q, k, v, o)

    def _ffn(self, i: int, kind: str):
        """Get ffn kind(weight, qweight) for layer i."""
        up_and_gate = self.params[
            f'{self.attn_layer_prefix}.{i}.ff_proj.{kind}']
        up_and_gate = self.transform(up_and_gate, kind)
        gate, up = up_and_gate.chunk(2, dim=0)
        down = self.params[f'{self.attn_layer_prefix}.{i}.ff_out.{kind}']
        down = self.transform(down, kind)
        return (up, down, gate)

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.ff_norm.weight']


@INPUT_MODELS.register_module(name='molmo')
class MolmoModel(LlamaModel):
    Reader = MolmoReader

    def model_info(self):
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['layer_norm_eps']
            attn_head_num = model_arg['num_attention_heads']
            kv_head_num = model_arg['num_key_value_heads']
            hidden_units = model_arg['hidden_size']
            rope_theta = model_arg.get('rope_theta')
            max_position_embeddings = model_arg.get('max_position_embeddings')
        return dict(
            num_layer=num_layer,
            norm_eps=norm_eps,
            head_num=attn_head_num,
            kv_head_num=kv_head_num,
            hidden_units=hidden_units,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
