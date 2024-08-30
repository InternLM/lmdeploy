# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from glob import glob

import torch
from safetensors.torch import load_file

from lmdeploy.archs import get_model_arch
from lmdeploy.tokenizer import Tokenizer

from .base import INPUT_MODELS, BaseInputModel, BaseReader


class LlamaReader(BaseReader):
    """LlamaReader."""

    attn_layer_prefix = 'model.layers'
    attn_layer_patten = r'model.layers.([0-9]+).'
    tok_embeddings_key = 'model.embed_tokens.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict, policy):
        super().__init__()
        self.params = unused_params
        self.params.update(new_params)
        self.last_bin = last_bin
        self.model_cfg = model_cfg
        tie_word_embeddings = self.model_cfg.get('tie_word_embeddings', False)
        if tie_word_embeddings:
            self.output_weight_key = self.tok_embeddings_key
        self.weight_suffix, self.processor = policy
        self.init_layer_id()

    def init_layer_id(self):
        """Get start/end transformer layer id."""
        super().init_layer_id()

    def clean_up(self, last: bool) -> None:
        """Clean up unused params."""
        super().clean_up(last)

    @property
    def start_layer_id(self):
        """Get start transformer layer id."""
        return self._start_layer_id

    @property
    def end_layer_id(self):
        """Get end transformer layer id."""
        return self._end_layer_id

    def tok_embeddings(self):
        """Get embeddings."""
        return self.params.get(self.tok_embeddings_key, None)

    def norm_weight(self):
        """Get norm."""
        return self.params.get(self.norm_weight_key, None)

    def output_weight(self):
        """Get output."""
        return self.params.get(self.output_weight_key, None)

    def _transform(self, x: torch.Tensor, kind: str):
        return self.processor(x, kind)

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}.self_attn.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, self.weight_suffix)

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        return self._attn(i, 'bias')

    def attn_zero(self, i: int):
        """Get q, k, v, o zero point for layer i."""
        return self._attn(i, 'qzeros')

    def attn_scale(self, i: int):
        """Get q, k, v, o scale for layer i."""
        return self._attn(i, 'scales')

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.input_layernorm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.mlp.{key}_proj.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, self.weight_suffix)

    def ffn_zero(self, i: int):
        """Get ffn zero point for layer i."""
        return self._ffn(i, 'qzeros')

    def ffn_scale(self, i: int):
        """Get ffn scale for layer i."""
        return self._ffn(i, 'scales')

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='llama')
class LlamaModel(BaseInputModel):
    """Llama model in hf format."""

    Reader = LlamaReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path)
        ckpt_path = kwargs.get('ckpt_path')
        self.policy = kwargs.get('input_policy')
        if ckpt_path is None:
            ckpt_path = model_path
        self.ckpt_path = ckpt_path
        self.ckpt_files = self.get_ckpt()
        _, self.model_config = get_model_arch(model_path)
        self.model_config = self.model_config.to_dict()

    def get_ckpt(self):
        """Get weight files."""
        patterns = ['*.safetensors', 'pytorch_model*.bin']
        files = []
        for pattern in patterns:
            files = glob(os.path.join(self.ckpt_path, pattern))
            files = [os.path.basename(file) for file in files]
            if len(files) > 0:
                break
        files = sorted(files)
        return files

    @property
    def nmgrs(self):
        """Get number of checkpoint."""
        return len(self.ckpt_files)

    def get_mgrs(self):
        """Conctruct all Reader."""
        assert self.nmgrs > 0, \
            f'could not find checkpoints in {self.ckpt_path}'
        unused_params = {}
        try:
            for i, ckpt in enumerate(self.ckpt_files):
                is_last_bin = i == len(self.ckpt_files) - 1
                if ckpt.endswith('.bin'):
                    new_params = torch.load(osp.join(self.ckpt_path, ckpt),
                                            map_location='cpu')
                else:
                    new_params = load_file(osp.join(self.ckpt_path, ckpt))
                ret = self.Reader(new_params,
                                  unused_params,
                                  i == self.nmgrs - 1,
                                  self.model_config,
                                  policy=self.policy)
                yield ret
                ret.clean_up(is_last_bin)
        except GeneratorExit:
            ret.clean_up(True)

    def tokenizer_info(self):
        """Read tokenizer info."""
        assert osp.isdir(self.model_path), self.model_path
        tk_model = Tokenizer(self.model_path)
        n_words = tk_model.vocab_size
        bos_id = tk_model.bos_token_id
        eos_id = tk_model.eos_token_id
        # bos_id may be None
        bos_id = bos_id or 0
        return n_words, bos_id, eos_id

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            attn_head_num = model_arg['num_attention_heads']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            hidden_units = model_arg['hidden_size']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            use_dynamic_ntk = 0
            scaling_type = ''
            low_freq_factor = 1.0
            high_freq_factor = 1.0
            original_max_position_embeddings = 0
            if isinstance(rope_scaling, dict):
                llama2_scaling_type = model_arg['rope_scaling'].get('type', '')
                llama3_scaling_type = model_arg['rope_scaling'].get(
                    'rope_type', '')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
                low_freq_factor = model_arg['rope_scaling'].get(
                    'low_freq_factor', 1.0)
                high_freq_factor = model_arg['rope_scaling'].get(
                    'high_freq_factor', 1.0)
                original_max_position_embeddings = model_arg[
                    'rope_scaling'].get('original_max_position_embeddings', 0)
                if llama2_scaling_type and llama3_scaling_type:
                    raise ValueError(
                        f'Ambiguous rope_scaling in config: {model_arg}')
                scaling_type = llama2_scaling_type if llama2_scaling_type \
                    else llama3_scaling_type
                if scaling_type == 'dynamic':
                    use_dynamic_ntk = 1

        return dict(
            num_layer=num_layer,
            norm_eps=norm_eps,
            head_num=attn_head_num,
            kv_head_num=kv_head_num,
            hidden_units=hidden_units,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            use_dynamic_ntk=use_dynamic_ntk,
            rope_scaling_type=scaling_type,
            rope_scaling_factor=scaling_factor,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor)
