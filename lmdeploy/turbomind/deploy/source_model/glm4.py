# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Glm4Reader(LlamaReader):
    """Glm4Reader."""

    attn_layer_patten = r'transformer.encoder.layers.([0-9]+).'
    tok_embeddings_key = 'transformer.embedding.word_embeddings.weight'
    norm_weight_key = 'transformer.encoder.final_layernorm.weight'
    output_weight_key = 'transformer.output_layer.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        qkv = self.params[f'transformer.encoder.layers.{i}'
                          f'.self_attention.query_key_value.{kind}']
        attn_head_num = self.model_cfg['attn_head_num']
        kv_head_num = self.model_cfg['kv_head_num']
        HEAD_DIM = 128
        q, k, v = torch.split(qkv, [
            attn_head_num * HEAD_DIM, kv_head_num * HEAD_DIM,
            kv_head_num * HEAD_DIM
        ],
                              dim=size_dim)
        o = self.params.get(
            f'transformer.encoder.layers.{i}.self_attention.dense.{kind}',
            None)
        if o is None:  # handle the case when qkv has bias but o doesn't
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
        return self.params[
            f'transformer.encoder.layers.{i}.input_layernorm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        up_and_gate = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.{kind}']
        up, gate = up_and_gate.chunk(2, dim=0)
        down = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.{kind}']

        return (up, down, gate)

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
        return self.params[
            f'transformer.encoder.layers.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='glm4')
class Glm4Model(LlamaModel):
    """Glm4 model in hf format."""

    Reader = Glm4Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    def tokenizer_info(self):
        """Read tokenizer info."""
        n_words = 151552
        bos_id = 0
        eos_id = 151329
        return n_words, bos_id, eos_id

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            config = json.load(f)
            num_layer = config['num_hidden_layers']
            norm_eps = config['layernorm_epsilon']
            rope_theta = float(config.get('rotary_emb_base', 10000.0))
            rope_ratio = float(config.get('rope_ratio'))
            rope_theta *= rope_ratio
            attn_head_num = config['num_attention_heads']
            kv_head_num = attn_head_num
            if config['multi_query_attention']:
                kv_head_num = config['multi_query_group_num']
            seq_length = config['seq_length']
        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    attn_head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=seq_length,
                    permute_qk=False)  # head layout is same as TM
