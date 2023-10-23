# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp

import torch
from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor

from lmdeploy.tokenizer import Tokenizer
from lmdeploy.turbomind.deploy.source_model.base import (INPUT_MODELS,
                                                         BaseInputModel,
                                                         BaseWeightFileMgr)


class HfWeightFileMgr(BaseWeightFileMgr):
    """HfWeightFileMgr."""

    attn_layer_patten = r'model.layers.([0-9]+).'

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
        return self.params.get('model.embed_tokens.weight', None)

    def norm_weight(self):
        return self.params.get('model.norm.weight', None)

    def output_weight(self):
        return self.params.get('lm_head.weight', None)

    def attn(self, i: int):
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params[
                f'model.layers.{i}.self_attn.{key}_proj.weight']
            result.append(tensor)
        return (*result, )

    def attn_bias(self, i: int):
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'model.layers.{i}.self_attn.{key}_proj.bias', None)
            result.append(tensor)
        return (*result, )

    def attn_zero(self, i: int):
        return (None, ) * 4

    def attn_scale(self, i: int):
        return (None, ) * 4

    def attn_norm(self, i: int):
        return self.params[f'model.layers.{i}.input_layernorm.weight']

    def ffn(self, i: int):
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[f'model.layers.{i}.mlp.{key}_proj.weight']
            result.append(tensor)
        return (*result, )

    def ffn_zero(self, i: int):
        return (None, ) * 3

    def ffn_scale(self, i: int):
        return (None, ) * 3

    def ffn_norm(self, i: int):
        return self.params[f'model.layers.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='hf')
class HfModel(BaseInputModel):
    """Llama model in hf format."""

    WeightFileMgr = HfWeightFileMgr

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path)
        self.ckpt_files = self.get_ckpt()

    def get_ckpt(self):
        suffixes = ['.safetensors', '.bin']
        for suffix in suffixes:
            files = [
                file for file in os.listdir(self.model_path)
                if file.endswith(suffix)
            ]
            if len(files) > 0:
                break
        assert len(
            files) > 0, f'could not find checkpoints in {self.model_path}'
        files = sorted(files)
        return files

    @property
    def nmgrs(self):
        return len(self.ckpt_files)

    def get_mgrs(self):
        unused_params = {}
        try:
            for i, ckpt in enumerate(self.ckpt_files):
                is_last_bin = i == len(self.ckpt_files) - 1
                if ckpt.endswith('.bin'):
                    new_params = torch.load(osp.join(self.model_path, ckpt),
                                            map_location='cpu')
                else:
                    new_params = load_file(osp.join(self.model_path, ckpt))
                ret = self.WeightFileMgr(new_params, unused_params)
                yield ret
                ret.clean_up(is_last_bin)
        except GeneratorExit:
            ret.clean_up(True)

    def tokenizer_info(self):
        assert osp.isfile(self.tokenizer_path), self.tokenizer_path
        try:
            tk_model = SentencePieceProcessor(model_file=self.tokenizer_path)
        except Exception as ex:
            tk_model = Tokenizer(self.model_path)
        # BOS / EOS token IDs
        n_words = tk_model.vocab_size
        bos_id = tk_model.bos_token_id
        eos_id = tk_model.eos_token_id
        return n_words, bos_id, eos_id

    def model_info(self):
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            repo_scaling = bool(model_arg.get('rope_scaling', False))

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    use_dynamic_ntk=int(repo_scaling))
