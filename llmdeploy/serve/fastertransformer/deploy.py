# Copyright (c) OpenMMLab. All rights reserved.
import configparser
import os
import os.path as osp
import shutil
from pathlib import Path

import fire
import safetensors
import torch
from sentencepiece import SentencePieceProcessor

supported_models = [
    'vicuna-7b', 'vicuna-13b', 'llama-7b', 'llama-13b', 'llama-30b',
    'llama-65b'
]
supported_formats = ['llama', 'hf']


def create_workspace(_path: str):
    try:
        if osp.exists(_path):
            shutil.rmtree(_path)
        os.makedirs(_path)
        print(f'create workspace in directory {_path}')
        return True
    except Exception as e:
        print(f'exception happened: {e}')
        return False


def copy_triton_model_templates(_path: str):
    try:
        cur_path = osp.abspath(__file__)
        dir_path = osp.dirname(cur_path)
        triton_models_path = osp.join(dir_path, 'triton_models')
        dst_path = osp.join(_path, 'triton_models')
        shutil.copytree(triton_models_path, dst_path, symlinks=True)
        print(f'copy triton model templates from "{triton_models_path}" to '
              f'"{dst_path}" successfully')
        shutil.copy(osp.join(dir_path, 'service_docker_up.sh'), _path)
        return True
    except Exception as e:
        print(f'copy triton model templates from "{triton_models_path}"'
              f' to "{dst_path}" failed: {e}')
        return False


def tokenizer_info(model_path: str):
    assert os.path.isfile(model_path), model_path
    sp_model = SentencePieceProcessor(model_file=model_path)
    # BOS / EOS token IDs
    n_words = sp_model.vocab_size()
    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()
    return n_words, bos_id, eos_id


def export(model_name: str, model_params: dict, tokenizer_path: str,
           out_dir: str, tp: int):
    out_dir = osp.join(out_dir, 'weights')
    os.makedirs(out_dir, exist_ok=True)

    def save_bin(param: torch.Tensor, name):
        print(name, param.shape)
        if param.dtype in [torch.float, torch.bfloat16]:
            param = param.half()
        param.contiguous().numpy().tofile(osp.join(out_dir, name))

    # reverse the splitting axes since the weights are transposed above
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
            print(f'*** splitting {param_name}, shape={param_data.shape}, '
                  f'split_dim={split_dim}')
            assert param_data.shape[split_dim] % tp == 0
            split_size = param_data.shape[split_dim] // tp
            splits = torch.split(param_data, split_size, dim=split_dim)
            for i, split in enumerate(splits):
                prefix, ext = osp.splitext(param_name)
                save_bin(split, f'{prefix}.{i}{ext}')
        elif copy:
            print(f'### copying {param_name}, shape={param_data.shape}')
            copies = [param_data] * tp
            for i, copy in enumerate(copies):
                prefix, ext = osp.splitext(param_name)
                save_bin(copy, f'{prefix}.{i}{ext}')
        else:
            save_bin(param_data, param_name)

    # export config and save it to {out_dir}/config.ini
    vocab_size, bos_id, eos_id = tokenizer_info(tokenizer_path)
    cfg = dict(llama=dict(model_name=model_name,
                          vocab_size=vocab_size,
                          bos_id=bos_id,
                          eos_id=eos_id,
                          size_per_head=128,
                          norm_eps=1e-6,
                          rotary_embedding=128))
    special = dict()
    if model_name.endswith('7b'):
        special = dict(head_num=32, num_layer=32, inter_size=11008)
    elif model_name.endswith('13b'):
        special = dict(head_num=40, num_layer=40, inter_size=13824)
    elif model_name.endswith('30b'):
        special = dict(head_num=52, num_layer=60, inter_size=17920)
    elif model_name.endswith('65b'):
        special = dict(head_num=64, num_layer=80, inter_size=22016)
    else:
        return False

    cfg['llama'].update(special)
    cfg['llama'].update(
        dict(weight_type='fp16',
             max_batch_size=32,
             max_context_token_num=4,
             session_len=2048,
             step_length=1,
             cache_max_entry_count=48,
             cache_chunk_size=8,
             use_context_fmha=1))

    config = configparser.ConfigParser()
    for section, key_values in cfg.items():
        config[section] = key_values

    config_path = osp.join(out_dir, 'config.ini')
    with open(config_path, 'w') as f:
        config.write(f)
    return True


def deploy_llama(model_name: str, model_path: str, tokenizer_path: str,
                 dst_path: str, tp: int):
    if osp.exists(tokenizer_path):
        shutil.copy(
            tokenizer_path,
            osp.join(dst_path, 'triton_models/tokenizer/tokenizer.model'))
    else:
        print('tokenizer model {tokenizer_path} does not exist')
        return -1
    # convert weights from llama to fastertransformer
    checkpoints = []
    for pattern in ['*.pth', '*.pt']:
        checkpoints += sorted(Path(model_path).glob(pattern))
    print(checkpoints)
    n_ckpt = len(checkpoints)
    model_params = {}

    def get_param(_name, _size):
        print(_name, _size)
        if _name not in model_params:
            model_params[_name] = torch.zeros(_size,
                                              dtype=torch.float16,
                                              device='cpu')
        return model_params[_name]

    for i, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for param_name, param_data in ckpt.items():
            key = param_name.split('.')[-2]
            # column-parallel
            if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'output']:
                size = param_data.size(0)
                param = get_param(
                    param_name,
                    [size * n_ckpt, param_data.size(1)])
                param.data[size * i:size * (i + 1), :] = param_data
            # row-parallel
            elif key in ['w2', 'wo', 'tok_embeddings']:
                size = param_data.size(-1)
                param = get_param(param_name,
                                  [param_data.size(0), size * n_ckpt])
                param.data[:, size * i:size * (i + 1)] = param_data
            elif i == 0:
                param = get_param(param_name, param_data.size())
                param.data = param_data
        del ckpt

    for name, param in model_params.items():
        # transpose all weights as FasterTransformer is expecting column-major
        # weights: (output_dims, input_dims) -> (input_dims, output_dims)
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

    return export(model_name, model_params, tokenizer_path, dst_path, tp)


def permute(x: torch.Tensor):
    SIZE_PER_HEAD = 128
    if x.shape[-1] > 1:  # qweights
        dim = x.shape[-1]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(-1, n_heads, 2,
                      dim // n_heads // 2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(n_heads, 2, dim // n_heads // 2,
                      1).transpose(1, 2).reshape(dim, 1)


def check_zero(x: torch.Tensor):
    _sum = x.flatten().sum().item()
    assert _sum == 0, str(_sum)


def deploy_hf(model_name: str, model_path: str, tokenizer_path: str,
              dst_path: str, tp: int):
    if tokenizer_path is None:
        tokenizer_path = osp.join(model_path, 'tokenizer.model')
    if osp.exists(tokenizer_path):
        shutil.copy(
            tokenizer_path,
            osp.join(dst_path, 'triton_models/tokenizer/tokenizer.model'))
    else:
        print('tokenizer model {tokenizer_path} does not exist')
        exit(-1)
    # convert weights from hf to fastertransformer
    model_params = {}

    _qweight = 'weight'
    _suffixes = [_qweight]

    _files = [file for file in os.listdir(model_path) if file.endswith('.bin')]
    _files = sorted(_files)

    _params = {}
    for _file in _files:
        _tmp = torch.load(osp.join(model_path, _file), map_location='cpu')
        _params.update(_tmp)

    def get_tensor(name):
        return _params[name]

    def get_tensor_transposed(name):
        return _params[name].t()

    for i in range(1000):
        try:
            # attention weights
            _qkvo = [f'model.layers.{i}.self_attn.{t}_proj' for t in 'qkvo']
            for suffix in _suffixes:
                q, k, v, o = map(get_tensor_transposed,
                                 map(('{}.' + suffix).format, _qkvo))
                if suffix == 'bias':
                    check_zero(q), check_zero(k), check_zero(v), check_zero(o)
                else:
                    # q, k has different layout for fb & hf, convert to fb's
                    # layout
                    q = permute(q)
                    k = permute(k)
                    if suffix == _qweight:  # weight, qweight
                        # insert a dimension for splitting heads later
                        qkv = torch.stack((q, k, v), dim=1)
                    else:  # scales, zeros
                        qkv = torch.stack((q, k, v), dim=0).squeeze(dim=-1)
                    for k, v in [('w_qkv', qkv), ('wo', o)]:
                        model_params[f'layers.{i}.attention.{k}.{suffix}'] = v
            # ffn weights
            _w123 = [
                f'model.layers.{i}.mlp.{t}_proj'
                for t in ['gate', 'down', 'up']
            ]
            for suffix in _suffixes:
                w1, w2, w3 = map(get_tensor_transposed,
                                 map(('{}.' + suffix).format, _w123))
                if suffix == 'bias':
                    check_zero(w1), check_zero(w2), check_zero(w3)
                else:
                    if suffix in ['scales', 'zeros']:
                        w1, w2, w3 = map(lambda x: x.squeeze(dim=-1),
                                         [w1, w2, w3])
                    for k, v in [('w1', w1), ('w2', w2), ('w3', w3)]:
                        model_params[
                            f'layers.{i}.feed_forward.{k}.{suffix}'] = v
            other = [('attention_norm.weight', 'input_layernorm.weight'),
                     ('ffn_norm.weight', 'post_attention_layernorm.weight')]
            for ft, hf in other:
                model_params[f'layers.{i}.' +
                             ft] = get_tensor(f'model.layers.{i}.' + hf)
        except safetensors.SafetensorError:
            break
        except KeyError:
            break

    other = [('tok_embeddings.weight', 'model.embed_tokens.weight'),
             ('norm.weight', 'model.norm.weight'),
             ('output.weight', 'lm_head.weight')]
    for ft, hf in other:
        model_params[ft] = get_tensor(hf)

    return export(model_name, model_params, tokenizer_path, dst_path, tp)


def pack_model_repository(workspace_path: str, triton_model_path):
    model_repo_dir = osp.join(workspace_path, 'model_repository')
    os.makedirs(model_repo_dir, exist_ok=True)
    os.symlink(src=osp.join('../triton_models/interactive'),
               dst=osp.join(model_repo_dir, 'fastertransformer'))
    os.symlink(src=osp.join('../triton_models/preprocessing'),
               dst=osp.join(model_repo_dir, 'preprocessing'))
    os.symlink(src=osp.join('../triton_models/postprocessing'),
               dst=osp.join(model_repo_dir, 'postprocessing'))


def main(model_name: str,
         model_path: str,
         model_format: str,
         tokenizer_path: str = None,
         dst_path: str = './workspace',
         tp: int = 1):
    """deploy llama family models via fastertransformer.

    Args:
        model_name (str): the name of the to-be-deployed model, such as
            llama-7b, llama-13b and etc
        model_path (str): the directory path of the model
        model_format (str): the format of the model, fb or hf. 'fb' stands for
            META's llama format, and 'hf' means huggingface format
        tokenizer_path (str): the path of tokenizer model
        dst_path (str): the destination path that saves outputs
        tp (int): the number of GPUs used for tensor parallelism
    """
    if model_name.lower() not in supported_models:
        print(f'"{model_name}" is not supported. The supported models are: '
              f'{supported_models}')
        exit(-1)

    if model_format not in supported_formats:
        print(f'the model format "{model_format}" is not supported. '
              f'The supported format are: {supported_formats}')
        exit(-1)

    if model_format == 'llama' and tokenizer_path is None:
        print('The model is llama. Its tokenizer model path should be '
              'specified')
        exit(-1)

    if not create_workspace(dst_path):
        exit(-1)

    if not copy_triton_model_templates(dst_path):
        exit(-1)

    model_name = model_name.lower()
    if model_format == 'llama':
        deploy_llama(model_name, model_path, tokenizer_path, dst_path, tp)
    else:
        deploy_hf(model_name, model_path, tokenizer_path, dst_path, tp)

    # pack model repository for triton inference server
    triton_model_path = osp.join(dst_path, 'triton_models')
    pack_model_repository(dst_path, triton_model_path)


if __name__ == '__main__':
    fire.Fire(main)
