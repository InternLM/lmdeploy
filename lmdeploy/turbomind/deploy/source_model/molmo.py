# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from ..config import RopeParam
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class MolmoReader(LlamaReader):
    attn_layer_prefix = 'model.transformer.blocks'
    attn_layer_patten = r'model\.transformer\.blocks\.([0-9]+).'
    norm_weight_key = 'model.transformer.ln_f.weight'
    output_weight_key = 'model.transformer.ff_out.weight'

    # In molmo, names of attention parameters are "att_proj.bias",
    # "att_proj.weight", "attn_norm.weight", "attn_out.weight", and names
    # of ffn parameters are "ff_norm", "ff_out", "ff_proj", so we
    # make the patterns are r'att' and r'ffn_', respectively.
    attn_pattern = r'att'
    ffn_pattern = r'ff_'

    def tok_embeddings(self):
        embed1 = self.params.get('model.transformer.wte.embedding', None)
        embed2 = self.params.get('model.transformer.wte.new_embedding', None)
        if embed1 is not None and embed2 is not None:
            return torch.cat((embed1, embed2), dim=0)
        else:
            assert embed1 is None and embed2 is None
            return None

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.attn_norm.weight']

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind(weight, bias, qweight) for layer i.

        Args:
            i (int): layer id
            kind (str): can be one of ["weight", "bias", "qweight"]
        """
        q, k, v = (None, ) * 3
        hidden_size = self.model_cfg['hidden_size']
        head_num = self.model_cfg['num_attention_heads']
        kv_head_num = self.model_cfg['num_key_value_heads']
        head_dim = hidden_size // head_num
        assert head_dim == 128
        fused_dims = (hidden_size, kv_head_num * head_dim, kv_head_num * head_dim)
        qkv = self.params.get(f'{self.attn_layer_prefix}.{i}.att_proj.{kind}')
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            q, k, v = qkv.split(fused_dims, dim=0)
        o = self.params.get(f'{self.attn_layer_prefix}.{i}.attn_out.{kind}')
        o = self.transform(o, kind)
        if o is None:  # handle the case when qkv has bias but o doesn't
            o = torch.zeros_like(q)
        return (q, k, v, o)

    def _ffn(self, i: int, kind: str):
        """Get ffn kind(weight, qweight) for layer i."""
        up_and_gate = self.params[f'{self.attn_layer_prefix}.{i}.ff_proj.{kind}']
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

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        config_path = osp.join(self.model_path, 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)

    def model_info(self):
        config = self.config
        num_layer = config['num_hidden_layers']
        norm_eps = config['layer_norm_eps']
        attn_head_num = config['num_attention_heads']
        kv_head_num = config['num_key_value_heads']
        hidden_units = config['hidden_size']
        rope_theta = config['rope_theta']
        max_position_embeddings = config['max_position_embeddings']
        vocab_size = config['vocab_size']
        # https://huggingface.co/allenai/Molmo-7B-D-0924/blob/main/modeling_molmo.py#L2041
        additional_vocab_size = 128
        inter_size = config['intermediate_size'] // 2
        attn_bias = config['qkv_bias']
        rope_param = RopeParam(type='default', base=rope_theta, dim=hidden_units // attn_head_num)
        return dict(
            num_layer=num_layer,
            norm_eps=norm_eps,
            head_num=attn_head_num,
            kv_head_num=kv_head_num,
            hidden_units=hidden_units,
            attn_bias=int(attn_bias),
            inter_size=inter_size,
            vocab_size=vocab_size,
            # https://huggingface.co/allenai/Molmo-7B-D-0924/blob/main/modeling_molmo.py#L564
            embedding_size=vocab_size + additional_vocab_size,
            rope_param=rope_param,
            max_position_embeddings=max_position_embeddings,
        )
