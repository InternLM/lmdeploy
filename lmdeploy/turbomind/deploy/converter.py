# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import shutil
from pathlib import Path

import fire
import torch

from lmdeploy.model import MODELS
from lmdeploy.utils import get_model

from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS, TurbomindModelConfig

supported_formats = ['llama', 'hf', 'awq', None]
special_input_model_map = {
    'qwen': 'qwen',
    'qwen2': 'qwen2',
    'baichuan': 'baichuan',
    'baichuan2': 'baichuan2',
    'internlm2': 'internlm2',
    'xcomposer2': 'xcomposer2',
    'deepseekvl': 'deepseekvl',
    'internvl': 'internvl'
}


def get_package_root_path():
    """Get lmdeploy root path."""
    import lmdeploy
    return Path(lmdeploy.__file__).parent


def get_model_format(model_name: str, model_format: str):
    """Get model format if not given or equal awq."""
    # get model name prefix
    if model_name.find('-') != -1:
        model_name = model_name[:model_name.find('-')]
    # rules:
    # 1) llama -> match special -> hf (if not matched)
    # 2) append awq (if model_format is awq)
    inferred_model_format = model_format
    if model_format in [None, 'hf']:
        inferred_model_format = special_input_model_map.get(model_name, 'hf')
    elif model_format == 'awq':
        inferred_model_format = special_input_model_map.get(model_name,
                                                            'hf') + '-awq'
    return inferred_model_format


def create_workspace(_path: str):
    """Create a workspace.

    Args:
        _path (str): the path of the workspace
    """
    if osp.exists(_path):
        print(f'remove workspace in directory {_path}')
        shutil.rmtree(_path)
    print(f'create workspace in directory {_path}')
    os.makedirs(_path)


def copy_triton_model_templates(_path: str):
    """copy triton model templates to the specified path.

    Args:
        _path (str): the target path
    Returns:
        str: the path of the triton models
    """

    root = get_package_root_path()
    dir_path = osp.join(root, 'serve', 'turbomind')
    triton_models_path = osp.join(dir_path, 'triton_models')
    dst_path = osp.join(_path, 'triton_models')
    print(f'copy triton model templates from "{triton_models_path}" to '
          f'"{dst_path}"')
    shutil.copytree(triton_models_path, dst_path, symlinks=True)
    service_docker_up_file = osp.join(dir_path, 'service_docker_up.sh')
    print(f'copy service_docker_up.sh from "{service_docker_up_file}" to '
          f'"{_path}"')
    shutil.copy(osp.join(dir_path, 'service_docker_up.sh'), _path)
    return dst_path


def copy_tokenizer(model_path: str, tokenizer_path: str,
                   triton_models_path: str, trust_remote_code: bool):
    """Copy tokenizer."""
    if tokenizer_path is not None:
        assert osp.exists(tokenizer_path), f'{tokenizer_path} does not exists.'

        shutil.copy(
            tokenizer_path,
            osp.join(triton_models_path,
                     osp.join('tokenizer', osp.basename(tokenizer_path))))
    else:
        from transformers import AutoTokenizer
        try:
            _ = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception:
            assert 0, (
                f'Failed to load tokenizer model from path {model_path}.'
                'please specify tokenizer path by --tokenizer-path')

    # move tokenizer model to the target path
    candidate = ['tokenizer.model', 'qwen.tiktoken', 'merges.txt']
    for name in candidate:
        tmp_path = osp.join(model_path, name)
        if osp.exists(tmp_path):
            shutil.copy(tmp_path,
                        osp.join(triton_models_path, 'tokenizer', name))
    # move py/json files that are related to tokenizer to the target path
    for _file in os.listdir(model_path):
        if _file.endswith('.json') or _file.endswith('.py'):
            json_path = osp.join(model_path, _file)
            shutil.copy(json_path,
                        osp.join(triton_models_path, 'tokenizer', _file))
    with get_package_root_path() as root_path:
        shutil.copy(osp.join(root_path, 'tokenizer.py'),
                    osp.join(triton_models_path, 'tokenizer'))


def update_output_format(model_name: str, model_format: str, model_path: str,
                         output_format: str):
    """Update output format according to model info."""
    TORCH_DTYPE_MAP = {torch.bfloat16: 'bf16'}
    MODEL_NAME_MAP = {'qwen': 'bf16', 'llama': 'half'}
    model_name = model_name.split('-')[0]

    def _fix_device_support(output_format):
        """fix device support."""
        if output_format == 'bf16':
            if not torch.cuda.is_bf16_supported():
                # device does not support bf16
                print('Device does not support bf16.')
                output_format = 'fp16'
        return output_format

    def _infer_output_format(config):
        """_infer_output_format."""
        torch_dtype = getattr(config, 'torch_dtype', None)
        if torch_dtype:
            updated_output_format = TORCH_DTYPE_MAP.get(
                torch_dtype, output_format)
        else:
            # get model name prefix
            updated_output_format = MODEL_NAME_MAP.get(model_name,
                                                       output_format)
        return _fix_device_support(updated_output_format)

    if model_format in MODEL_NAME_MAP:
        updated_output_format = MODEL_NAME_MAP.get(model_name, output_format)
        return _fix_device_support(updated_output_format)
    else:
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_path,
                                                trust_remote_code=True)
        except Exception as e:  # noqa
            from transformers import PretrainedConfig
            config = PretrainedConfig.get_config_dict(model_path)[0]
        return _infer_output_format(config)


def update_config_weight_type(output_format: str,
                              config: TurbomindModelConfig):
    WEIGHT_TYPE_MAP = {
        'fp32': 'fp32',
        'fp16': 'fp16',
        'bf16': 'bf16',
        'w4': 'int4',
        'w8': 'int8'
    }
    config.weight_type = WEIGHT_TYPE_MAP[output_format]


def pack_model_repository(workspace_path: str):
    """package the model repository.

    Args:
        workspace_path: the path of workspace
    """
    os.symlink(src=osp.join('..', '..', 'tokenizer'),
               dst=osp.join(workspace_path, 'triton_models', 'preprocessing',
                            '1', 'tokenizer'))
    os.symlink(src=osp.join('..', '..', 'tokenizer'),
               dst=osp.join(workspace_path, 'triton_models', 'postprocessing',
                            '1', 'tokenizer'))
    os.symlink(src=osp.join('..', '..', 'weights'),
               dst=osp.join(workspace_path, 'triton_models', 'interactive',
                            '1', 'weights'))
    model_repo_dir = osp.join(workspace_path, 'model_repository')
    os.makedirs(model_repo_dir, exist_ok=True)
    os.symlink(src=osp.join('..', 'triton_models', 'interactive'),
               dst=osp.join(model_repo_dir, 'turbomind'))
    os.symlink(src=osp.join('..', 'triton_models', 'preprocessing'),
               dst=osp.join(model_repo_dir, 'preprocessing'))
    os.symlink(src=osp.join('..', 'triton_models', 'postprocessing'),
               dst=osp.join(model_repo_dir, 'postprocessing'))


def main(model_name: str,
         model_path: str,
         model_format: str = None,
         tokenizer_path: str = None,
         dst_path: str = 'workspace',
         tp: int = 1,
         quant_path: str = None,
         group_size: int = 0,
         trust_remote_code: bool = False,
         **kwargs):
    """deploy llama family models via turbomind.

    Args:
        model_name (str): the name of the to-be-deployed model, such as
            llama-7b, llama-13b, vicuna-7b and etc
        model_path (str): the directory path of the model
        model_format (str): the format of the model, should choose from
            ['llama', 'hf', 'awq', None]. 'llama' stands for META's llama
            format, 'hf' means huggingface llama format, and 'awq' means
            llama(hf) model quantized by lmdeploy/lite/quantization/awq.py.
            the default value is None, which means the model_format will be
            inferred based on model_name
        tokenizer_path (str): the path of tokenizer model
        dst_path (str): the destination path that saves outputs
        tp (int): the number of GPUs used for tensor parallelism, should be 2^n
        quant_path (str): Path of the quantized model, which can be None.
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
        trust_remote_code (bool):  Whether or not to allow for custom models
            defined on the Hub in their own modeling files. Defaults to False
        kwargs (dict): other params for convert
    """

    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'

    from lmdeploy.turbomind.supported_models import (SUPPORTED_ARCHS,
                                                     get_model_arch,
                                                     is_supported)
    assert is_supported(model_path), (
        f'turbomind does not support {model_path}. '
        'Plz try pytorch engine instead.')

    arch, _ = get_model_arch(model_path)

    assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'

    output_format = 'fp16'

    # get input model format
    assert model_format in supported_formats, 'the model format ' \
        f'should be in {supported_formats}'

    inferred_model_format = get_model_format(SUPPORTED_ARCHS[arch],
                                             model_format)
    if inferred_model_format not in INPUT_MODELS.module_dict.keys():
        supported_keys = list(INPUT_MODELS.module_dict.keys())
        print(f'with model name {model_name} and model formst {model_format}, '
              f'the inferred model format is {inferred_model_format}, '
              f'which is not in supported list {supported_keys}')
        exit(-1)

    if not os.path.exists(model_path):
        print(f'can\'t find model from local_path {model_path}, '
              'try to download from huggingface')
        model_path = get_model(model_path)
        print(f'load model from {model_path}')

    # create workspace
    create_workspace(dst_path)

    triton_models_path = copy_triton_model_templates(dst_path)

    copy_tokenizer(model_path, tokenizer_path, triton_models_path,
                   trust_remote_code)

    # turbomind config
    cfg = TurbomindModelConfig.from_dict({}, allow_none=True)
    cfg.model_name = model_name
    cfg.tensor_para_size = tp
    cfg.rotary_embedding = cfg.size_per_head
    cfg.group_size = group_size
    if inferred_model_format.find('awq') != -1:
        cfg.weight_type = 'int4'
        output_format = 'w4'
        if 'xcomposer2' in inferred_model_format:
            output_format = 'plora-w4'
        assert group_size > 0, f'group_size: {group_size} should > 0'
    else:
        output_format = update_output_format(model_name, inferred_model_format,
                                             model_path, output_format)
        update_config_weight_type(output_format, cfg)

    # convert
    print('model_name            ', model_name)
    print('model_format          ', model_format)
    print('inferred_model_format ', inferred_model_format)
    print('model_path            ', model_path)
    print('tokenizer_path        ', tokenizer_path)
    print('output_format         ', output_format)
    weight_path = osp.join(triton_models_path, 'weights')
    input_model = INPUT_MODELS.get(inferred_model_format)(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        ckpt_path=quant_path)
    output_model = OUTPUT_MODELS.get(output_format)(input_model=input_model,
                                                    cfg=cfg,
                                                    to_file=True,
                                                    out_dir=weight_path)
    output_model.export()

    # update `tensor_para_size` in `triton_models/interactive/config.pbtxt`
    with open(osp.join(triton_models_path, 'interactive', 'config.pbtxt'),
              'a') as f:
        param = \
            'parameters {\n  key: "tensor_para_size"\n  value: {\n    ' \
            'string_value: ' + f'"{tp}"\n' + '  }\n}\n' + \
            'parameters {\n  key: "model_name"\n  value: {\n    ' \
            'string_value: ' + f'"{model_name}"\n' + '  }\n}\n'
        f.write(param)

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
