# Copyright (c) OpenMMLab. All rights reserved.
import configparser
import json
import os
import os.path as osp
import re
import shutil
import sys
from pathlib import Path

import fire
import safetensors
import torch
from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor

import lmdeploy
from lmdeploy.model import MODELS

supported_formats = ['llama', 'hf', 'awq', 'qwen']


def get_package_root_path():
    import lmdeploy
    return Path(lmdeploy.__file__).parent


def create_workspace(_path: str):
    """Create a workspace.

    Args:
        _path (str): the path of the workspace
    Returns:
        bool: success or not
    """
    try:
        if osp.exists(_path):
            shutil.rmtree(_path)
        os.makedirs(_path)
        print(f'create workspace in directory {_path}')
        return True
    except Exception as e:
        print(f'create workspace in {_path} failed: {e}')
        return False


def destroy_workspace(_path: str):
    """destroy workspace.

    Args:
        _path(str): the path of the workspace
    Returns:
        bool: success or not
    """
    try:
        shutil.rmtree(_path)
        print(f'destroy workspace in directory {_path}')
        return True
    except Exception as e:
        print(f'destroy workspace in {_path} failed: {e}')
        return False


def copy_triton_model_templates(_path: str):
    """copy triton model templates to the specified path.

    Args:
        _path (str): the target path
    Returns:
        str: the path of the triton models
    """
    try:
        cur_path = osp.abspath(__file__)
        dir_path = osp.dirname(cur_path)
        triton_models_path = osp.join(dir_path, 'triton_models')
        dst_path = osp.join(_path, 'triton_models')
        shutil.copytree(triton_models_path, dst_path, symlinks=True)
        print(f'copy triton model templates from "{triton_models_path}" to '
              f'"{dst_path}" successfully')
        shutil.copy(osp.join(dir_path, 'service_docker_up.sh'), _path)
        return dst_path
    except Exception as e:
        print(f'copy triton model templates from "{triton_models_path}"'
              f' to "{dst_path}" failed: {e}')
        return None


def tokenizer_info_sp(model_path: str):
    """Return the vocabulary size, bos token id and eos token id.

    Args:
        model_path (str): the tokenizer model's path
    Returns:
        tuple: vocabulary size, bos token id and eos token id
    """
    assert os.path.isfile(model_path), model_path
    sp_model = SentencePieceProcessor(model_file=model_path)
    # BOS / EOS token IDs
    n_words = sp_model.vocab_size()
    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()
    return n_words, bos_id, eos_id


def tokenizer_info_qwen(model_dir: str):
    n_words = 151851
    bos_id = 0
    eos_id = 151643
    return n_words, bos_id, eos_id


def load_checkpoint(model_path):
    """Load checkpoint files into torch format.

    Args:
        model_path (str): the checkpoint folder
    Returns:
        Dict[str, torch.Tensor]: weight in torch format
    """
    suffixes = ['.safetensors', '.bin']
    for suffix in suffixes:
        files = [
            file for file in os.listdir(model_path) if file.endswith(suffix)
        ]
        if len(files) > 0:
            break

    assert len(files) > 0, f'could not find checkpoints in {model_path}'
    files = sorted(files)
    print(files)
    params = {}
    for file in files:
        if file.endswith('.bin'):
            tmp = torch.load(osp.join(model_path, file), map_location='cpu')
        else:
            tmp = load_file(osp.join(model_path, file))
        params.update(tmp)
    return params


def export(model_name: str,
           num_layer: int,
           norm_eps: float,
           kv_head_num: int,
           model_params: dict,
           tokenizer_path: str,
           out_dir: str,
           tp: int,
           size_per_head: int = 128,
           group_size: int = 0,
           weight_type: str = 'fp16',
           max_position_embeddings: int = 0,
           use_dynamic_ntk: int = 0,
           use_logn_attn: int = 0,
           rope_theta: float = 10000.0,
           tokenizer_info=tokenizer_info_sp):
    """Export deploying information to a config file.

    Args:
        model_name (str): model's name
        num_layer (int): the number of transformer blocks
        norm_eps (float): norm epsilon
        model_params (dict): parameters of a model
        tokenizer_path (str): the tokenizer model's path
        out_dir (str): the path of the output directory
        tp (int): the number of tensor parallelism
        size_per_head (int): the dimension of each head
    """
    out_dir = osp.join(out_dir, 'weights')
    os.makedirs(out_dir, exist_ok=True)

    def save_bin(param: torch.Tensor, name):
        print(name, param.shape)
        if param.dtype in [torch.float, torch.bfloat16]:
            param = param.half()
        param.contiguous().cpu().numpy().tofile(osp.join(out_dir, name))

    attn_bias = False
    inter_size = 0

    tok_embeddings = model_params['tok_embeddings.weight']
    _vocab_size, dim = tok_embeddings.shape
    head_num = dim // size_per_head
    if _vocab_size % tp != 0:
        # Resolve https://github.com/InternLM/lmdeploy/issues/266
        # Pad tok_embeddings and output weights, making their shape divisible by TP # noqa: E501
        pad_size = (_vocab_size + tp - 1) // tp * tp - _vocab_size
        # Pad weight at the bottom of dim 0
        model_params['tok_embeddings.weight'] = torch.nn.functional.pad(
            tok_embeddings, (0, 0, 0, pad_size), 'constant', 0)
        # Pad output weight at the bottom of dim 0
        model_params['output.weight'] = torch.nn.functional.pad(
            model_params['output.weight'], (0, 0, 0, pad_size), 'constant', 0)

    # reverse the splitting axes since the weights are transposed above
    for param_name, param_data in model_params.items():
        split_dim = None
        key, ext = param_name.split('.')[-2:]
        if key == 'w_qkv' and ext == 'bias':
            attn_bias = True
        copy = False
        if key in ['w1', 'w3', 'w13', 'w_qkv']:
            split_dim = -1
            # TODO: move parameter extraction outside of the loop
            if key == 'w1':
                inter_size = max(inter_size, param_data.shape[-1])
            elif key == 'w13':
                inter_size = max(inter_size, param_data.shape[-1] // 2)
        elif key in ['w2', 'wo']:
            if ext in ['bias']:
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

    assert inter_size > 0

    # export config and save it to {out_dir}/config.ini
    model = MODELS.get(model_name)()
    vocab_size, bos_id, eos_id = tokenizer_info(tokenizer_path)
    assert _vocab_size >= vocab_size, \
        f'different vocab size {_vocab_size} vs {vocab_size}'
    cfg = dict(llama=dict(
        model_name=model_name,
        head_num=head_num,
        kv_head_num=kv_head_num,
        size_per_head=size_per_head,
        vocab_size=_vocab_size,
        num_layer=num_layer,
        rotary_embedding=size_per_head,
        rope_theta=rope_theta,
        inter_size=inter_size,
        norm_eps=norm_eps,
        attn_bias=int(attn_bias),
        start_id=bos_id,
        end_id=eos_id,
        weight_type=weight_type,
        group_size=group_size,
        # parameters for turbomind
        max_batch_size=32,
        max_context_token_num=4,
        session_len=model.session_len + 8,
        step_length=1,
        cache_max_entry_count=48,
        cache_chunk_size=1,
        use_context_fmha=1,
        quant_policy=0,
        tensor_para_size=tp,
        # extra attention params
        max_position_embeddings=max_position_embeddings,
        use_dynamic_ntk=int(use_dynamic_ntk),
        use_logn_attn=int(use_logn_attn),
    ))

    config = configparser.ConfigParser()
    for section, key_values in cfg.items():
        config[section] = key_values

    config_path = osp.join(out_dir, 'config.ini')
    with open(config_path, 'w') as f:
        config.write(f)
    return True


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int,
              dim: int):

    def reshape(x):
        return x.view(x.size(0), tp, -1) if dim == 2 else x.view(tp, -1)

    qkv = torch.cat((reshape(q), reshape(k), reshape(v)), dim=-1)

    # (input_dim, head_num + 2 * kv_head_num)
    return qkv.view(q.size(0), -1)


def deploy_llama(model_name: str, model_path: str, tokenizer_path: str,
                 triton_models_path: str, tp: int):
    """Deploy a model with huggingface transformers' format.

    Args:
        model_name (str): the name of the to-be-deployed model
        model_path (str): the path of the directory where the model weight
          files are
        tokenizer_path (str): the path of the tokenizer model path
        triton_models_path (str): the path of the exported triton models
        tp (int): the number of tensor parallelism
    """
    if osp.exists(tokenizer_path):
        shutil.copy(tokenizer_path,
                    osp.join(triton_models_path, 'tokenizer/tokenizer.model'))
        with get_package_root_path() as root_path:
            shutil.copy(osp.join(root_path, 'tokenizer.py'),
                        osp.join(triton_models_path, 'tokenizer'))
    else:
        print(f'tokenizer model {tokenizer_path} does not exist')
        return False
    # read model arguments from params.json
    try:
        params_path = osp.join(model_path, 'params.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['n_layers']
            norm_eps = model_arg['norm_eps']
            head_num = model_arg.get('n_heads', 32)
            kv_head_num = model_arg.get('n_kv_heads', head_num)
    except Exception as e:
        print(f'get "n_layers" and "norm_eps" from {params_path} failed: {e}')
        return False

    # convert weights from llama to turbomind format
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

    for name, param in model_params.items():
        # transpose all weights as TurboMind is expecting column-major
        # weights: (output_dims, input_dims) -> (input_dims, output_dims)
        key = name.split('.')[-2]
        if key in ['w1', 'w3', 'wq', 'wk', 'wv', 'w2', 'wo']:
            param.data = param.data.t()

    # concat qkv projection
    for t in ['weight', 'bias']:
        for i in range(1000):
            _qkv = [
                f'layers.{i}.attention.{k}.{t}' for k in ['wq', 'wk', 'wv']
            ]
            try:
                qkv = tuple(map(model_params.pop, _qkv))
            except KeyError:
                break
            # concat by heads
            qkv = merge_qkv(*qkv, tp, dim=2 if t == 'weight' else 1)
            print(f'layers.{i}.attention.w_qkv.{t}', qkv.shape)
            model_params[f'layers.{i}.attention.w_qkv.{t}'] = qkv

    assert i == 0 or num_layer == i, f'miss matched layers: {num_layer} vs {i}'

    return export(model_name, num_layer, norm_eps, kv_head_num, model_params,
                  tokenizer_path, triton_models_path, tp)


def permute(x: torch.Tensor):
    SIZE_PER_HEAD = 128
    if x.shape[-1] > 1:
        dim = x.shape[-1]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(-1, n_heads, 2,
                      dim // n_heads // 2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // SIZE_PER_HEAD
        return x.view(n_heads, 2, dim // n_heads // 2,
                      1).transpose(1, 2).reshape(dim, 1)


def deploy_hf(model_name: str, model_path: str, tokenizer_path: str,
              triton_models_path: str, tp: int):
    """Deploy a model with huggingface transformers' format.

    Args:
        model_name (str): the name of the to-be-deployed model
        model_path (str): the path of the directory where the model weight
          files are
        tokenizer_path (str): the path of the tokenizer model path
        triton_models_path (str): the path of the exported triton models
        tp (int): the number of tensor parallelism
    """
    if tokenizer_path is None:
        tokenizer_path = osp.join(model_path, 'tokenizer.model')
    if osp.exists(tokenizer_path):
        shutil.copy(tokenizer_path,
                    osp.join(triton_models_path, 'tokenizer/tokenizer.model'))
        for _file in os.listdir(model_path):
            if _file.endswith('.json') or _file.endswith('.py'):
                json_path = osp.join(model_path, _file)
                shutil.copy(json_path,
                            osp.join(triton_models_path, 'tokenizer', _file))
        with get_package_root_path() as root_path:
            shutil.copy(osp.join(root_path, 'tokenizer.py'),
                        osp.join(triton_models_path, 'tokenizer'))
    else:
        print(f'tokenizer model {tokenizer_path} does not exist')
        exit(-1)

    # read model arguments from params.json
    try:
        params_path = osp.join(model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            repo_scaling = bool(model_arg.get('rope_scaling', False))
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
    except Exception as e:
        print(f'get "num_hidden_layers" and "rms_norm_eps" from '
              f'{params_path} failed: {e}')
        return False

    # convert weights from hf to turbomind
    model_params = {}

    _qweight = 'weight'
    _suffixes = [_qweight, 'bias']

    _params = load_checkpoint(model_path)

    def get_tensor(name):
        """return tensor according its name."""
        return _params[name]

    def get_tensor_transposed(name: str):
        """return a transposed tensor according its name."""
        if name not in _params and name.find('bias'):
            return None
        return _params[name].t()

    w_pack = False
    if 'model.layers.0.self_attn.W_pack.weight' in _params:
        w_pack = True

    for i in range(1000):
        try:
            # attention weights
            for suffix in _suffixes:
                if w_pack:
                    _qkvo = [
                        f'model.layers.{i}.self_attn.{t}'
                        for t in ['W_pack', 'o_proj']
                    ]
                    qkv, o = map(get_tensor_transposed,
                                 map(('{}.' + suffix).format, _qkvo))

                    if qkv is None:
                        continue
                    _shape = qkv.shape[1] // 3
                    _qkv = torch.split(qkv, [_shape, _shape, _shape], dim=1)
                    q = _qkv[0]
                    k = _qkv[1]
                    v = _qkv[2]

                else:
                    _qkvo = [
                        f'model.layers.{i}.self_attn.{t}_proj' for t in 'qkvo'
                    ]
                    q, k, v, o = map(get_tensor_transposed,
                                     map(('{}.' + suffix).format, _qkvo))
                if q is None:
                    continue
                # q, k has different layout for fb & hf, convert to fb's
                # layout
                q = permute(q)
                k = permute(k)
                if suffix == _qweight:  # weight, qweight
                    qkv = merge_qkv(q, k, v, tp, dim=2)
                    print(suffix, qkv.shape)
                else:  # scales, zeros, bias
                    qkv = merge_qkv(q, k, v, tp, dim=1)
                    print(suffix, qkv.shape)
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
                if w1 is None:
                    continue
                if suffix in ['scales', 'zeros', 'bias']:
                    w1, w2, w3 = map(lambda x: x.squeeze(dim=-1), [w1, w2, w3])
                for k, v in [('w1', w1), ('w2', w2), ('w3', w3)]:
                    model_params[f'layers.{i}.feed_forward.{k}.{suffix}'] = v
            other = [('attention_norm.weight', 'input_layernorm.weight'),
                     ('ffn_norm.weight', 'post_attention_layernorm.weight')]
            for ft, hf in other:
                model_params[f'layers.{i}.' +
                             ft] = get_tensor(f'model.layers.{i}.' + hf)
        except safetensors.SafetensorError:
            break
        except KeyError:
            break

    assert num_layer == i, f'miss matched layers: {num_layer} vs {i}'

    other = [('tok_embeddings.weight', 'model.embed_tokens.weight'),
             ('norm.weight', 'model.norm.weight'),
             ('output.weight', 'lm_head.weight')]
    for ft, hf in other:
        model_params[ft] = get_tensor(hf)

    if model_name == 'baichuan2-7b':
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/modeling_baichuan.py#L507
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        model_params['output.weight'] = torch.nn.functional.normalize(
            model_params['output.weight'])

    return export(model_name,
                  num_layer,
                  norm_eps,
                  kv_head_num,
                  model_params,
                  tokenizer_path,
                  triton_models_path,
                  tp,
                  max_position_embeddings=max_position_embeddings,
                  use_dynamic_ntk=repo_scaling,
                  rope_theta=rope_theta)


def deploy_awq(model_name: str, model_path: str, tokenizer_path: str,
               triton_models_path: str, tp: int, quant_path: str,
               group_size: int):
    """Deploy a model with huggingface transformers' format.

    Args:
        model_name (str): the name of the to-be-deployed model
        model_path (str): the path of the directory where the model weight
          files are
        tokenizer_path (str): the path of the tokenizer model path
        triton_models_path (str): the path of the exported triton models
        tp (int): the number of tensor parallelism
        quant_path (str): path of the quantized model, which can be None
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
    """
    if tokenizer_path is None:
        tokenizer_path = osp.join(model_path, 'tokenizer.model')
    if osp.exists(tokenizer_path):
        shutil.copy(tokenizer_path,
                    osp.join(triton_models_path, 'tokenizer/tokenizer.model'))
        for _file in os.listdir(model_path):
            if _file.endswith('.json') or _file.endswith('.py'):
                json_path = osp.join(model_path, _file)
                shutil.copy(json_path,
                            osp.join(triton_models_path, 'tokenizer', _file))
        with get_package_root_path() as root_path:
            shutil.copy(osp.join(root_path, 'tokenizer.py'),
                        osp.join(triton_models_path, 'tokenizer'))
    else:
        print(f'tokenizer model {tokenizer_path} does not exist')
        exit(-1)

    # read model arguments from params.json
    try:
        params_path = osp.join(model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
    except Exception as e:
        print(f'get "num_hidden_layers" and "rms_norm_eps" from '
              f'{params_path} failed: {e}')
        return False

    # convert weights from hf to turbomind
    if quant_path is None:
        _files = [
            osp.join(model_path, file) for file in os.listdir(model_path)
            if file.endswith('.bin')
        ]
        _files = sorted(_files)
    else:
        _files = [quant_path]

    model_params = {}

    _params = {}
    for _file in _files:
        _tmp = torch.load(_file, map_location='cpu')
        _params.update(_tmp)

    def get_tensor(name):
        """return tensor according its name."""
        return _params[name].cuda().contiguous()

    # import _turbomind as _tm
    # TODO: find another way import _turbomind
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    sys.path.append(osp.join(lmdeploy_dir, 'lib'))
    import _turbomind as _tm  # noqa: E402

    def transpose_qk_s4(src: torch.Tensor):
        assert src.is_contiguous()
        dst = torch.zeros_like(src)
        _tm.transpose_qk_s4_k_m8(src, dst,
                                 src.size(-1) * 8, src.size(0), group_size)
        return dst

    def fuse_w1_w3_s4(w1_qw: torch.Tensor, w1_qz: torch.Tensor,
                      w1_s: torch.Tensor, w3_qw: torch.Tensor,
                      w3_qz: torch.Tensor, w3_s: torch.Tensor):

        def fuse(a: torch.Tensor, b: torch.Tensor):
            ab = torch.cat((a, b)).contiguous()
            _ab = torch.zeros_like(ab)
            _tm.fuse_w1_w3_s4_k_m8(ab, _ab, a.size(-1) * 8, a.size(0))
            return _ab.view(a.size(0), -1)

        w13_qw = fuse(w1_qw, w3_qw)
        w13_qz = fuse(w1_qz, w3_qz)

        w13_s = torch.cat((w1_s, w3_s)).view(2, w1_s.size(0), -1)
        w13_s = w13_s.permute(1, 2, 0).contiguous().view(w1_s.size(0), -1)

        return w13_qw, w13_qz, w13_s

    def convert_s4(qw: torch.Tensor, qz: torch.Tensor, s: torch.Tensor,
                   group_size: int):
        assert qw.is_contiguous()
        assert qz.is_contiguous()
        assert s.is_contiguous()
        _qw = torch.zeros_like(qw)
        _sz = torch.zeros_like(s, dtype=torch.int32)  # half2
        _ws = torch.zeros_like(s)
        _tm.convert_s4_k_m8(_qw, _sz, _ws, qw, s, qz,
                            qw.size(-1) * 8, qw.size(0), group_size)
        return _qw, _sz

    def tp_m_s4(x: torch.Tensor, tp: int):
        return x.view(x.size(0) // 32, tp, -1, 128).permute(0, 2, 3,
                                                            1).contiguous()

    attn_bias = False

    for i in range(num_layer):
        print(i)

        # attention weights
        q_qw = get_tensor(f'model.layers.{i}.self_attn.q_proj.qweight')
        k_qw = get_tensor(f'model.layers.{i}.self_attn.k_proj.qweight')
        v_qw = get_tensor(f'model.layers.{i}.self_attn.v_proj.qweight')
        o_qw = get_tensor(f'model.layers.{i}.self_attn.o_proj.qweight')

        q_qz = get_tensor(f'model.layers.{i}.self_attn.q_proj.qzeros')
        k_qz = get_tensor(f'model.layers.{i}.self_attn.k_proj.qzeros')
        v_qz = get_tensor(f'model.layers.{i}.self_attn.v_proj.qzeros')
        o_qz = get_tensor(f'model.layers.{i}.self_attn.o_proj.qzeros')

        q_s = get_tensor(f'model.layers.{i}.self_attn.q_proj.scales')
        k_s = get_tensor(f'model.layers.{i}.self_attn.k_proj.scales')
        v_s = get_tensor(f'model.layers.{i}.self_attn.v_proj.scales')
        o_s = get_tensor(f'model.layers.{i}.self_attn.o_proj.scales')

        try:
            q_b = get_tensor(f'model.layers.{i}.self_attn.q_proj.bias')
            k_b = get_tensor(f'model.layers.{i}.self_attn.k_proj.bias')
            v_b = get_tensor(f'model.layers.{i}.self_attn.v_proj.bias')
            o_b = get_tensor(f'model.layers.{i}.self_attn.o_proj.bias')
            attn_bias = True
        except:  # noqa: E722
            pass

        q_qw = transpose_qk_s4(q_qw)
        k_qw = transpose_qk_s4(k_qw)
        q_qz = transpose_qk_s4(q_qz)
        k_qz = transpose_qk_s4(k_qz)
        q_s = permute(q_s)
        k_s = permute(k_s)

        qkv_qw = merge_qkv(q_qw, k_qw, v_qw, tp, dim=2)
        qkv_qz = merge_qkv(q_qz, k_qz, v_qz, tp, dim=2)
        qkv_s = merge_qkv(q_s, k_s, v_s, tp, dim=2)

        qkv_qw, qkv_sz = convert_s4(qkv_qw, qkv_qz, qkv_s, group_size)

        qkv_qw = tp_m_s4(qkv_qw, tp)

        model_params[f'layers.{i}.attention.w_qkv.qweight'] = qkv_qw
        model_params[f'layers.{i}.attention.w_qkv.scales_zeros'] = qkv_sz

        o_qw, o_sz = convert_s4(o_qw, o_qz, o_s, group_size)

        model_params[f'layers.{i}.attention.wo.qweight'] = o_qw
        model_params[f'layers.{i}.attention.wo.scales_zeros'] = o_sz

        if attn_bias:
            q_b = permute(q_b)
            k_b = permute(k_b)
            qkv_b = merge_qkv(q_b, k_b, v_b, tp, dim=1)
            model_params[f'layers.{i}.attention.w_qkv.bias'] = qkv_b
            model_params[f'layers.{i}.attention.wo.bias'] = o_b

        # ffn weights
        w1_qw = get_tensor(f'model.layers.{i}.mlp.gate_proj.qweight')
        w2_qw = get_tensor(f'model.layers.{i}.mlp.down_proj.qweight')
        w3_qw = get_tensor(f'model.layers.{i}.mlp.up_proj.qweight')

        w1_qz = get_tensor(f'model.layers.{i}.mlp.gate_proj.qzeros')
        w2_qz = get_tensor(f'model.layers.{i}.mlp.down_proj.qzeros')
        w3_qz = get_tensor(f'model.layers.{i}.mlp.up_proj.qzeros')

        w1_s = get_tensor(f'model.layers.{i}.mlp.gate_proj.scales')
        w2_s = get_tensor(f'model.layers.{i}.mlp.down_proj.scales')
        w3_s = get_tensor(f'model.layers.{i}.mlp.up_proj.scales')

        w13_qw, w13_qz, w13_s = fuse_w1_w3_s4(w1_qw, w1_qz, w1_s, w3_qw, w3_qz,
                                              w3_s)

        w13_qw, w13_sz = convert_s4(w13_qw, w13_qz, w13_s, group_size)
        w2_qw, w2_sz = convert_s4(w2_qw, w2_qz, w2_s, group_size)

        w13_qw = tp_m_s4(w13_qw, tp)

        model_params[f'layers.{i}.feed_forward.w13.qweight'] = w13_qw
        model_params[f'layers.{i}.feed_forward.w13.scales_zeros'] = w13_sz

        model_params[f'layers.{i}.feed_forward.w2.qweight'] = w2_qw
        model_params[f'layers.{i}.feed_forward.w2.scales_zeros'] = w2_sz

        # norm weights
        attn_norm = get_tensor(f'model.layers.{i}.input_layernorm.weight')
        ffn_norm = get_tensor(
            f'model.layers.{i}.post_attention_layernorm.weight')

        model_params[f'layers.{i}.attention_norm.weight'] = attn_norm
        model_params[f'layers.{i}.ffn_norm.weight'] = ffn_norm

    other = [('tok_embeddings.weight', 'model.embed_tokens.weight'),
             ('norm.weight', 'model.norm.weight'),
             ('output.weight', 'lm_head.weight')]
    for ft, hf in other:
        model_params[ft] = get_tensor(hf)

    return export(model_name,
                  num_layer,
                  norm_eps,
                  kv_head_num,
                  model_params,
                  tokenizer_path,
                  triton_models_path,
                  tp,
                  weight_type='int4',
                  group_size=group_size,
                  rope_theta=rope_theta)


def deploy_qwen(model_name: str, model_path: str, tokenizer_path: str,
                triton_models_path: str, tp: int):
    """Deploy a model with huggingface transformers' format.

    Args:
        model_name (str): the name of the to-be-deployed model
        model_path (str): the path of the directory where the model weight
          files are
        tokenizer_path (str): the path of the tokenizer model path
        triton_models_path (str): the path of the exported triton models
        tp (int): the number of tensor parallelism
        quant_path (str): path of the quantized model, which can be None
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
    """

    if osp.exists(model_path):
        shutil.copy(osp.join(model_path, 'qwen.tiktoken'),
                    osp.join(triton_models_path, 'tokenizer'))
        for _file in os.listdir(model_path):
            if _file.endswith('.json') or _file.endswith('.py'):
                json_path = osp.join(model_path, _file)
                shutil.copy(json_path,
                            osp.join(triton_models_path, 'tokenizer', _file))
        with get_package_root_path() as root_path:
            shutil.copy(osp.join(root_path, 'tokenizer.py'),
                        osp.join(triton_models_path, 'tokenizer'))
    else:
        print(f'tokenizer model {tokenizer_path} does not exist')
        exit(-1)

    # read model arguments from params.json
    try:
        params_path = osp.join(model_path, 'config.json')
        with open(params_path) as f:
            config = json.load(f)
            num_layer = config['num_hidden_layers']
            norm_eps = config['layer_norm_epsilon']
            rope_theta = float(config.get('rotary_emb_base', 10000.0))
            if 'num_key_value_heads' in config:
                kv_head_num = config['num_key_value_heads']
            else:
                kv_head_num = config['num_attention_heads']
            seq_length = config['seq_length']
            use_dynamic_ntk = config['use_dynamic_ntk']
            use_logn_attn = config['use_logn_attn']
    except Exception as e:
        print(f'get "num_hidden_layers" and "layer_norm_epsilon" from '
              f'{params_path} failed: {e}')
        return False

    # convert weights from hf to turbomind
    model_params = {}

    _params = load_checkpoint(model_path)

    def get_tensor(name, trans=True):
        """return a transposed tensor according its name."""
        if trans:
            return _params[name].cuda().t()
        else:
            return _params[name].cuda()

    for i in range(num_layer):
        print(i)

        # qkv weights
        qkv_w = get_tensor(f'transformer.h.{i}.attn.c_attn.weight')
        q_w, k_w, v_w = torch.split(qkv_w, qkv_w.size(-1) // 3, dim=-1)
        q_w, k_w = permute(q_w), permute(k_w)
        qkv_w = merge_qkv(q_w, k_w, v_w, tp, dim=2)
        model_params[f'layers.{i}.attention.w_qkv.weight'] = qkv_w

        # qkv bias
        qkv_b = get_tensor(f'transformer.h.{i}.attn.c_attn.bias')
        q_b, k_b, v_b = torch.split(qkv_b, qkv_b.size(-1) // 3)
        q_b, k_b = permute(q_b), permute(k_b)
        qkv_b = merge_qkv(q_b, k_b, v_b, tp, dim=1)
        model_params[f'layers.{i}.attention.w_qkv.bias'] = qkv_b

        # o weights
        o_w = get_tensor(f'transformer.h.{i}.attn.c_proj.weight')
        model_params[f'layers.{i}.attention.wo.weight'] = o_w
        model_params[f'layers.{i}.attention.wo.bias'] = torch.zeros_like(q_b)

        # ffn weights
        # ours: w2(silu(w1(x)) * w3(x))
        # qwen: c_proj(w1(x) * silu(w2(x)))
        w1 = get_tensor(f'transformer.h.{i}.mlp.w2.weight')
        w3 = get_tensor(f'transformer.h.{i}.mlp.w1.weight')
        w2 = get_tensor(f'transformer.h.{i}.mlp.c_proj.weight')
        model_params[f'layers.{i}.feed_forward.w1.weight'] = w1
        model_params[f'layers.{i}.feed_forward.w2.weight'] = w2
        model_params[f'layers.{i}.feed_forward.w3.weight'] = w3

        # norm weights
        attn_norm = get_tensor(f'transformer.h.{i}.ln_1.weight')
        ffn_norm = get_tensor(f'transformer.h.{i}.ln_2.weight')

        model_params[f'layers.{i}.attention_norm.weight'] = attn_norm
        model_params[f'layers.{i}.ffn_norm.weight'] = ffn_norm

    other = [('tok_embeddings.weight', 'transformer.wte.weight'),
             ('norm.weight', 'transformer.ln_f.weight'),
             ('output.weight', 'lm_head.weight')]
    for ft, hf in other:
        model_params[ft] = get_tensor(hf, trans=False)

    return export(model_name,
                  num_layer,
                  norm_eps,
                  kv_head_num,
                  model_params,
                  model_path,
                  triton_models_path,
                  tp,
                  max_position_embeddings=seq_length,
                  use_dynamic_ntk=use_dynamic_ntk,
                  use_logn_attn=use_logn_attn,
                  rope_theta=rope_theta,
                  tokenizer_info=tokenizer_info_qwen)


def pack_model_repository(workspace_path: str):
    """package the model repository.

    Args:
        workspace_path: the path of workspace
    """
    os.symlink(src='../../tokenizer',
               dst=osp.join(workspace_path, 'triton_models', 'preprocessing',
                            '1', 'tokenizer'))
    os.symlink(src='../../tokenizer',
               dst=osp.join(workspace_path, 'triton_models', 'postprocessing',
                            '1', 'tokenizer'))
    os.symlink(src='../../weights',
               dst=osp.join(workspace_path, 'triton_models', 'interactive',
                            '1', 'weights'))
    model_repo_dir = osp.join(workspace_path, 'model_repository')
    os.makedirs(model_repo_dir, exist_ok=True)
    os.symlink(src=osp.join('../triton_models/interactive'),
               dst=osp.join(model_repo_dir, 'turbomind'))
    os.symlink(src=osp.join('../triton_models/preprocessing'),
               dst=osp.join(model_repo_dir, 'preprocessing'))
    os.symlink(src=osp.join('../triton_models/postprocessing'),
               dst=osp.join(model_repo_dir, 'postprocessing'))


def main(model_name: str,
         model_path: str,
         model_format: str = None,
         tokenizer_path: str = None,
         dst_path: str = './workspace',
         tp: int = 1,
         quant_path: str = None,
         group_size: int = 0):
    """deploy llama family models via turbomind.

    Args:
        model_name (str): the name of the to-be-deployed model, such as
            llama-7b, llama-13b, vicuna-7b and etc
        model_path (str): the directory path of the model
        model_format (str): the format of the model, fb or hf. 'fb' stands for
            META's llama format, and 'hf' means huggingface format
        tokenizer_path (str): the path of tokenizer model
        dst_path (str): the destination path that saves outputs
        tp (int): the number of GPUs used for tensor parallelism, should be 2^n
        quant_path (str): path of the quantized model, which can be None
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
    """
    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'

    assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'

    if model_format is None:
        model_format = 'qwen' if model_name == 'qwen-7b' else 'hf'

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

    triton_models_path = copy_triton_model_templates(dst_path)
    if triton_models_path is None:
        exit(-1)

    if model_format == 'llama':
        res = deploy_llama(model_name, model_path, tokenizer_path,
                           triton_models_path, tp)
    elif model_format == 'hf':
        res = deploy_hf(model_name, model_path, tokenizer_path,
                        triton_models_path, tp)
    elif model_format == 'awq':
        res = deploy_awq(model_name, model_path, tokenizer_path,
                         triton_models_path, tp, quant_path, group_size)
    elif model_format == 'qwen':
        res = deploy_qwen(model_name, model_path, tokenizer_path,
                          triton_models_path, tp)

    # update `tensor_para_size` in `triton_models/interactive/config.pbtxt`
    with open(osp.join(triton_models_path, 'interactive/config.pbtxt'),
              'a') as f:
        param = \
            'parameters {\n  key: "tensor_para_size"\n  value: {\n    ' \
            'string_value: ' + f'"{tp}"\n' + '  }\n}\n' + \
            'parameters {\n  key: "model_name"\n  value: {\n    ' \
            'string_value: ' + f'"{model_name}"\n' + '  }\n}\n'
        f.write(param)
    if not res:
        print(f'deploy model "{model_name}" via turbomind failed')
        destroy_workspace(dst_path)
        exit(-1)

    # pack model repository for triton inference server
    pack_model_repository(dst_path)

    # update the value of $TP in `service_docker_up.sh`
    file_path = osp.join(dst_path, 'service_docker_up.sh')
    with open(file_path, 'r') as f:
        content = f.read()
        content = re.sub('TP=1', f'TP={tp}', content)
    with open(file_path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    fire.Fire(main)
