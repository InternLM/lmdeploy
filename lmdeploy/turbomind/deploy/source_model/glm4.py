# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Glm4Reader(LlamaReader):
    """Glm4Reader."""

    attn_layer_patten = r'transformer.encoder.layers.([0-9]+).'
    tok_embeddings_key = 'transformer.embedding.word_embeddings.weight'
    norm_weight_key = 'transformer.encoder.final_layernorm.weight'
    output_weight_key = 'transformer.output_layer.weight'

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        qkv = self.params[f'transformer.encoder.layers.{i}'
                          f'.self_attention.query_key_value.{kind}']
        qkv = self.transform(qkv, kind)
        attn_head_num = self.model_cfg['num_attention_heads']
        kv_head_num = attn_head_num
        if self.model_cfg.get('multi_query_attention', False):
            kv_head_num = self.model_cfg['multi_query_group_num']
        HEAD_DIM = 128
        q, k, v = torch.split(qkv, [
            attn_head_num * HEAD_DIM, kv_head_num * HEAD_DIM,
            kv_head_num * HEAD_DIM
        ],
                              dim=0)
        o = self.params.get(
            f'transformer.encoder.layers.{i}.self_attention.dense.{kind}')
        o = self.transform(o, kind)
        if o is None:  # handle the case when qkv has bias but o doesn't
            o = torch.zeros_like(q)
        return q, k, v, o

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'transformer.encoder.layers.{i}.input_layernorm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        up_and_gate = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.{kind}']
        up_and_gate = self.transform(up_and_gate, kind)
        up, gate = up_and_gate.chunk(2, dim=0)
        down = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.{kind}']
        down = self.transform(down, kind)
        return (up, down, gate)

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[
            f'transformer.encoder.layers.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='glm4')
class Glm4Model(LlamaModel):
    """Glm2/3/4 model in hf format."""

    Reader = Glm4Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        config_path = osp.join(self.model_path, 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)

    def tokenizer_info(self):
        """Read tokenizer info."""
        n_words = self.config['padded_vocab_size']
        bos_id = 0
        eos_id = self.config['eos_token_id']
        if isinstance(eos_id, List):
            eos_id = eos_id[0]
        return n_words, bos_id, eos_id

    def model_info(self):
        """Read model info."""
        config = self.config
        hidden_units = config.get('hidden_size', None)
        num_layer = config.get('num_hidden_layers', None)
        num_layer = config.get('num_layers', num_layer)
        norm_eps = config['layernorm_epsilon']
        rope_theta = float(config.get('rotary_emb_base', 10000.0))
        rope_ratio = float(config.get('rope_ratio', 1.0))
        rope_theta *= rope_ratio
        attn_head_num = config['num_attention_heads']
        kv_head_num = attn_head_num
        if config['multi_query_attention']:
            kv_head_num = config['multi_query_group_num']
        seq_length = config['seq_length']
        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    hidden_units=hidden_units,
                    rope_theta=rope_theta,
                    max_position_embeddings=seq_length,
                    rotary_embedding=64,
                    permute_qk=False)  # head layout is same as TM
