# Copyright (c) OpenMMLab. All rights reserved.

import torch
import fire
import os.path as osp
from os import makedirs
from pathlib import Path
import safetensors
from typing import List
from tqdm import tqdm


def import_fb(ckpt_dir: str):
    checkpoints = []
    for pattern in ['*.pth', '*.pt']:
        checkpoints += sorted(Path(ckpt_dir).glob(pattern))
    print(checkpoints)
    n_ckpt = len(checkpoints)
    model_params = {}

    def get_param(name, size):
        print(name, size)
        if name not in model_params:
            model_params[name] = torch.zeros(
                size, dtype=torch.float16, device='cpu')
        return model_params[name]
    for i, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for param_name, param_data in ckpt.items():
            key = param_name.split('.')[-2]
            if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'output']:  # column-parallel
                size = param_data.size(0)
                param = get_param(
                    param_name, [size * n_ckpt, param_data.size(1)])
                param.data[size * i: size * (i + 1), :] = param_data
            elif key in ['w2', 'wo', 'tok_embeddings']:          # row-parallel
                size = param_data.size(-1)
                param = get_param(
                    param_name, [param_data.size(0), size * n_ckpt])
                param.data[:, size * i: size * (i + 1)] = param_data
            elif i == 0:
                param = get_param(param_name, param_data.size())
                param.data = param_data
        del ckpt

    for name, param in model_params.items():
        # transpose all weights as FasterTransformer is expecting column-major weights
        # (output_dims, input_dims) -> (input_dims, output_dims)
        key = name.split('.')[-2]
        if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'w2', 'wo']:
            param.data = param.data.t()

    # concat qkv projection
    for i in range(1000):
        _qkv = [f'layers.{i}.attention.{k}.weight' for k in ['wq', 'wk', 'wv']]
        try:
            qkv = tuple(map(model_params.pop, _qkv))
        except KeyError:
            break
        qkv = torch.stack(qkv, dim=1)
        model_params[f'layers.{i}.attention.w_qkv.weight'] = qkv
        print(qkv.shape, qkv.dtype)

    return model_params


def permute(x: torch.Tensor):
    SIZE_PER_HEAD = 128
    if x.shape[-1] > 1:  # qweights
        dim = x.shape[-1]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(-1, n_heads, 2, dim // n_heads // 2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(n_heads, 2, dim // n_heads // 2, 1).transpose(1, 2).reshape(dim, 1)


def check_zero(x: torch.Tensor):
    sum = x.flatten().sum().item()
    assert sum == 0, str(sum)


def import_gptq(path: str):
    model_params = {}

    _qweight = 'weight'
    _suffixes = [_qweight]
    n_split = 3
    if True:
        _params = {}
        for i in tqdm(range(0, n_split)):
            filename = "pytorch_model-{:05d}-of-{:05d}.bin".format(i + 1, n_split)
            _tmp = torch.load(osp.join(path, filename), map_location='cpu')
            _params.update(_tmp)
        # print('\n'.join(_params.keys()))
        def get_tensor(name):
            return _params[name]
        def get_tensor_transposed(name):
            return _params[name].t()

    # _qweight = 'qweight'
    # _suffixes = [_qweight, 'bias', 'scales', 'zeros']
    # with safetensors.safe_open(path, framework='pt') as f:
    #     get_tensor = f.get_tensor
    #     # quantized weights are already in column major, no need to transpose
    #     get_tensor_transposed = get_tensor
        for i in range(1000):
            try:
                # attention weights
                _qkvo = [f'model.layers.{i}.self_attn.{t}_proj' for t in 'qkvo']
                for suffix in _suffixes:
                    q, k, v, o = map(get_tensor_transposed, map(('{}.' + suffix).format, _qkvo))
                    if suffix == 'bias':
                        check_zero(q), check_zero(k), check_zero(v), check_zero(o)
                    else:
                        # q, k has different layout for fb & hf, convert to fb's layout
                        q = permute(q)
                        k = permute(k)
                        if suffix == _qweight:  # weight, qweight
                            # insert a dimension for splitting heads later
                            # qkv = torch.cat([q[:, None, :], k[:, None, :], v[:, None, :]], dim=1)
                            qkv = torch.stack((q, k, v), dim=1)
                        else:  # scales, zeros
                            # qkv = torch.cat([q[None, :], k[None, :], v[None, :]], dim=0).squeeze(dim=-1)
                            qkv = torch.stack((q, k, v), dim=0).squeeze(dim=-1)
                        for k, v in [('w_qkv', qkv), ('wo', o)]:
                            model_params[f'layers.{i}.attention.{k}.{suffix}'] = v
                # ffn weights
                _w123 = [f'model.layers.{i}.mlp.{t}_proj' for t in ['gate', 'down', 'up']]
                for suffix in _suffixes:
                    w1, w2, w3 = map(get_tensor_transposed, map(('{}.' + suffix).format, _w123))
                    if suffix == 'bias':
                        check_zero(w1), check_zero(w2), check_zero(w3)
                    else:
                        if suffix in ['scales', 'zeros']:
                            w1, w2, w3 = map(lambda x: x.squeeze(dim=-1), [w1, w2, w3])
                        for k, v in [('w1', w1), ('w2', w2), ('w3', w3)]:
                            model_params[f'layers.{i}.feed_forward.{k}.{suffix}'] = v
                other = [('attention_norm.weight', 'input_layernorm.weight'),
                         ('ffn_norm.weight', 'post_attention_layernorm.weight')]
                for ours, theirs in other:
                    model_params[f'layers.{i}.' + ours] = get_tensor(f'model.layers.{i}.' + theirs)
            except safetensors.SafetensorError:
                break
            except KeyError:
                break
            print(i)

        other = [('tok_embeddings.weight', 'model.embed_tokens.weight'),
                 ('norm.weight', 'model.norm.weight'),
                 ('output.weight', 'lm_head.weight')]
        for ours, theirs in other:
            model_params[ours] = get_tensor(theirs)

        return model_params


def export(model_params: dict, out_dir: str, n_inference: int):
    makedirs(out_dir, exist_ok=True)

    def save_bin(param: torch.Tensor, name):
        print(name, param.shape)
        if param.dtype in [torch.float, torch.bfloat16]:
            param = param.half()
        param.contiguous().numpy().tofile(osp.join(out_dir, name))

    # reverse the spliting axes since the weights are transposed above
    for param_name, param_data in model_params.items():
        split_dim = None
        key, ext = param_name.split('.')[-2:]
        copy = False
        if key in ['w1', 'w3', 'w_qkv']:
            split_dim = -1
        elif key in ['w2', 'wo']:
            if ext in ['scales', 'zeros']:
                copy = True
            else:
                split_dim = 0
        if split_dim is not None:
            print(f'*** spliting {param_name}, shape={param_data.shape}, split_dim={split_dim}')
            assert param_data.shape[split_dim] % n_inference == 0
            split_size = param_data.shape[split_dim] // n_inference
            splits = torch.split(param_data, split_size, dim=split_dim)
            for i, split in enumerate(splits):
                prefix, ext = osp.splitext(param_name)
                save_bin(split, f'{prefix}.{i}{ext}')
        elif copy:
            print(f'### copying {param_name}, shape={param_data.shape}')
            copies = [param_data] * n_inference
            for i, copy in enumerate(copies):
                prefix, ext = osp.splitext(param_name)
                save_bin(copy, f'{prefix}.{i}{ext}')
        else:
            save_bin(param_data, param_name)


def main(kind: str, input_path: str, out_dir: str, n_inference: int = 1):
    if kind == 'fb':
        model_params = import_fb(input_path)
    elif kind == 'gptq':
        model_params = import_gptq(input_path)
    else:
        raise RuntimeError(f'Unsupported kind: {kind}')

    export(model_params, out_dir, n_inference)


if __name__ == '__main__':
    fire.Fire(main)