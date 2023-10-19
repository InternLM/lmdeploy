# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor

from lmdeploy.turbomind.deploy.source_model.base import (INPUT_MODELS,
                                                         BaseInputModel,
                                                         BaseWeightFileMgr)


def reverse_permute(x: torch.Tensor):
    SIZE_PER_HEAD = 128
    if x.shape[-1] > 1:
        dim = x.shape[-1]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(-1, n_heads, dim // n_heads // 2,
                      2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(n_heads, dim // n_heads // 2, 2,
                      1).transpose(1, 2).reshape(dim, 1)


class LlamaWeightFileMgr(BaseWeightFileMgr):
    """LlamaWeightFileMgr."""

    def __init__(self, model_path: str, start_layer_id: int,
                 end_layer_id: int):
        super().__init__()
        self._start_layer_id = start_layer_id
        self._end_layer_id = end_layer_id
        self.params = self.load_model(model_path)

    def init_layer_id(self):
        pass

    def load_model(self, model_path):
        checkpoints = []
        for pattern in ['*.pth', '*.pt']:
            checkpoints += sorted(Path(model_path).glob(pattern))
        n_ckpt = len(checkpoints)
        model_params = {}

        def get_param(_name, _size):
            if _name not in model_params:
                model_params[_name] = torch.zeros(_size,
                                                  dtype=torch.float16,
                                                  device='cpu')
            return model_params[_name]

        for i, ckpt_path in enumerate(checkpoints):
            ckpt = torch.load(ckpt_path, map_location='cpu')

            for i, ckpt_path in enumerate(checkpoints):
                ckpt = torch.load(ckpt_path, map_location='cpu')

                for param_name, param_data in ckpt.items():
                    key, ext = param_name.split('.')[-2:]
                    # column-parallel
                    if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'output']:
                        size = param_data.size(0)
                        if ext == 'weight':
                            param = get_param(
                                param_name,
                                [size * n_ckpt,
                                 param_data.size(1)])
                            param.data[size * i:size * (i + 1), :] = param_data
                        else:  # bias
                            param = get_param(param_name, [size * n_ckpt])
                            param.data[size * i:size * (i + 1)] = param_data
                    # row-parallel
                    elif key in ['w2', 'wo', 'tok_embeddings']:
                        size = param_data.size(-1)
                        if ext == 'weight':
                            param = get_param(
                                param_name,
                                [param_data.size(0), size * n_ckpt])
                            param.data[:, size * i:size * (i + 1)] = param_data
                        else:  # bias
                            param = get_param(param_name, [size])
                            param.data = param_data
                    elif i == 0:
                        param = get_param(param_name, param_data.size())
                        param.data = param_data
                del ckpt

        for name, param in model_params.items():
            # transpose all weights as TurboMind is expecting column-major
            # (output_dims, input_dims) -> (input_dims, output_dims)
            key = name.split('.')[-2]
            if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'w2', 'wo']:
                param.data = param.data.t()
                if key in ['wq', 'wk']:
                    param.data = reverse_permute(param.data)
        return model_params

    def clean_up(self, last: bool) -> None:
        self.params.clear()

    @property
    def start_layer_id(self):
        return self._start_layer_id

    @property
    def end_layer_id(self):
        return self._end_layer_id

    def tok_embeddings(self):
        return self.params.get('tok_embeddings.weight')

    def norm_weight(self):
        return self.params.get('norm.weight')

    def output_weight(self):
        return self.params.get('output.weight')

    def attn(self, i: int):
        result = []
        for key in ['wq', 'wk', 'wv', 'wo']:
            tensor = self.params.pop(f'layers.{i}.attention.{key}.weight')
            tensor = tensor.t() if tensor is not None else None
            result.append(tensor)
        return (*result, )

    def attn_bias(self, i: int):
        result = []
        for key in ['wq', 'wk', 'wv', 'wo']:
            tensor = self.params.pop(f'layers.{i}.attention.{key}.bias', None)
            tensor = tensor.t() if tensor is not None else None
            result.append(tensor)
        return (*result, )

    def attn_zero(self, i: int):
        return (None, ) * 4

    def attn_scale(self, i: int):
        return (None, ) * 4

    def attn_norm(self, i: int):
        return self.params.pop(f'layers.{i}.attention_norm.weight')

    def ffn(self, i: int):
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params.pop(f'layers.{i}.feed_forward.{key}.weight')
            result.append(tensor.t())
        return (*result, )

    def ffn_zero(self, i: int):
        return (None, ) * 3

    def ffn_scale(self, i: int):
        return (None, ) * 3

    def ffn_norm(self, i: int):
        return self.params.pop(f'layers.{i}.ffn_norm.weight')


@INPUT_MODELS.register_module(name='llama')
class LlamaModel(BaseInputModel):
    """Llama model in fb format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)

    @property
    def nmgrs(self):
        return 1

    def get_mgrs(self):
        end_layer_id = self.model_info()['num_layer']
        try:
            for _ in range(1):
                ret = LlamaWeightFileMgr(self.model_path, 0, end_layer_id)
                yield ret
                ret.clean_up(True)
        except GeneratorExit:
            ret.clean_up(True)

    def tokenizer_info(self):
        assert osp.isfile(self.tokenizer_path), self.tokenizer_path
        sp_model = SentencePieceProcessor(model_file=self.tokenizer_path)
        # BOS / EOS token IDs
        n_words = sp_model.vocab_size()
        bos_id = sp_model.bos_id()
        eos_id = sp_model.eos_id()
        return n_words, bos_id, eos_id

    def model_info(self):
        params_path = osp.join(self.model_path, 'params.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['n_layers']
            norm_eps = model_arg['norm_eps']
            head_num = model_arg.get('n_heads', 32)
            kv_head_num = model_arg.get('n_kv_heads', head_num)

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=head_num,
                    kv_head_num=kv_head_num)
