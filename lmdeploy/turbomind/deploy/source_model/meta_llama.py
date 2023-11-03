# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor

from .base import INPUT_MODELS, BaseInputModel, BaseReader


def reverse_permute(x: torch.Tensor, size_per_head: int = 128):
    """reverse permute to hf format."""
    if x.shape[-1] > 1:
        dim = x.shape[-1]
        n_heads = dim // size_per_head
        return x.view(-1, n_heads, dim // n_heads // 2,
                      2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // size_per_head
        return x.view(n_heads, dim // n_heads // 2, 2,
                      1).transpose(1, 2).reshape(dim, 1)


class MetaLlamaReader(BaseReader):
    """MetaLlamaReader."""

    def __init__(self, model_path: str, start_layer_id: int,
                 end_layer_id: int):
        super().__init__()
        self._start_layer_id = start_layer_id
        self._end_layer_id = end_layer_id
        self.params = self.load_model(model_path)

    def init_layer_id(self):
        """Empty."""
        pass

    def load_model(self, model_path):
        """Load all parameters."""
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

        from tqdm import tqdm
        pbar = tqdm(total=n_ckpt, desc='load meta ckpt', leave=False)
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
                            [size * n_ckpt, param_data.size(1)])
                        param.data[size * i:size * (i + 1), :] = param_data
                    else:  # bias
                        param = get_param(param_name, [size * n_ckpt])
                        param.data[size * i:size * (i + 1)] = param_data
                # row-parallel
                elif key in ['w2', 'wo', 'tok_embeddings']:
                    size = param_data.size(-1)
                    if ext == 'weight':
                        param = get_param(param_name,
                                          [param_data.size(0), size * n_ckpt])
                        param.data[:, size * i:size * (i + 1)] = param_data
                    else:  # bias
                        param = get_param(param_name, [size])
                        param.data = param_data
                elif i == 0:
                    param = get_param(param_name, param_data.size())
                    param.data = param_data
            del ckpt
            pbar.update(1)
        pbar.close()

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
        """Clean up unused params."""
        self.params.clear()

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
        return self.params.get('tok_embeddings.weight')

    def norm_weight(self):
        """Get norm."""
        return self.params.get('norm.weight')

    def output_weight(self):
        """Get output."""
        return self.params.get('output.weight')

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        result = []
        for key in ['wq', 'wk', 'wv', 'wo']:
            tensor = self.params[f'layers.{i}.attention.{key}.weight']
            tensor = tensor.t() if tensor is not None else None
            result.append(tensor)
        return (*result, )

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        result = []
        for key in ['wq', 'wk', 'wv', 'wo']:
            tensor = self.params.get(f'layers.{i}.attention.{key}.bias')
            tensor = tensor.t() if tensor is not None else None
            result.append(tensor)
        return (*result, )

    def attn_zero(self, i: int):
        """Get q, k, v, o zero point for layer i."""
        return (None, ) * 4

    def attn_scale(self, i: int):
        """Get q, k, v, o scale for layer i."""
        return (None, ) * 4

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'layers.{i}.attention_norm.weight']

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[f'layers.{i}.feed_forward.{key}.weight']
            result.append(tensor.t())
        return (*result, )

    def ffn_zero(self, i: int):
        """Get ffn zero point for layer i."""
        return (None, ) * 3

    def ffn_scale(self, i: int):
        """Get ffn scale for layer i."""
        return (None, ) * 3

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'layers.{i}.ffn_norm.weight']


@INPUT_MODELS.register_module(name='llama')
class MetaLlamaModel(BaseInputModel):
    """Llama model in fb format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    @property
    def nmgrs(self):
        """Get number of checkpoint."""
        return 1

    def get_mgrs(self):
        """Conctruct all BaseReader."""
        end_layer_id = self.model_info()['num_layer']
        try:
            if hasattr(self, 'meta_reader'):
                yield self.meta_reader
            else:
                self.meta_reader = MetaLlamaReader(self.model_path, 0,
                                                   end_layer_id)
                yield self.meta_reader
        except GeneratorExit:
            pass

    def tokenizer_info(self):
        """Read tokenizer info."""
        assert osp.isfile(self.tokenizer_path), self.tokenizer_path
        sp_model = SentencePieceProcessor(model_file=self.tokenizer_path)
        # BOS / EOS token IDs
        n_words = sp_model.vocab_size()
        bos_id = sp_model.bos_id()
        eos_id = sp_model.eos_id()
        return n_words, bos_id, eos_id

    def model_info(self):
        """Read model info."""
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
