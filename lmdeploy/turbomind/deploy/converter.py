# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import shutil
from pathlib import Path

import fire

from lmdeploy.model import MODELS
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
from lmdeploy.turbomind.deploy.target_model.base import (OUTPUT_MODELS,
                                                         TurbomindModelConfig)

supported_formats = ['llama', 'hf', 'awq', 'qwen']


def get_package_root_path():
    import lmdeploy
    return Path(lmdeploy.__file__).parent


def guess_tokenizer_path(model_path: str):
    tokenizer_path = None
    candidate = ['tokenizer.model', 'qwen.tiktoken']
    for name in candidate:
        tmp_path = osp.join(model_path, name)
        if osp.exists(tmp_path):
            tokenizer_path = tmp_path
            break
    assert tokenizer_path, 'please supply tokenizer path by --tokenizer-path'
    return tokenizer_path


def guess_model_format(model_name: str):
    model_format = 'qwen' if model_name.startswith('qwen') else 'hf'
    return model_format


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


def copy_triton_model_templates(_path: str):
    """copy triton model templates to the specified path.

    Args:
        _path (str): the target path
    Returns:
        str: the path of the triton models
    """
    try:
        root = get_package_root_path()
        dir_path = osp.join(root, 'serve/turbomind')
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
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
    """

    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'

    assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'

    output_format = 'fp16'

    if model_format is None:
        model_format = guess_model_format(model_name)
    if model_format not in supported_formats:
        print(f'the model format "{model_format}" is not supported. '
              f'The supported format are: {supported_formats}')
        exit(-1)

    if tokenizer_path is None:
        tokenizer_path = guess_tokenizer_path(model_path)

    # create workspace
    if not create_workspace(dst_path):
        print(f'Can\'t create dst path {dst_path}')
        exit(-1)

    triton_models_path = copy_triton_model_templates(dst_path)
    if not triton_models_path:
        print('Can\'t copy triton model templates')
        exit(-1)

    # turbomind config
    cfg = TurbomindModelConfig.from_dict({}, allow_none=True)
    cfg.model_name = model_name
    cfg.tensor_para_size = tp
    cfg.rotary_embedding = cfg.size_per_head
    cfg.group_size = group_size
    if model_format == 'awq':
        cfg.weight_type = 'int4'
        output_format = 'w4a16'

    # convert
    weight_path = osp.join(triton_models_path, 'weights')
    input_model = INPUT_MODELS.get(model_format)(model_path, tokenizer_path)
    output_model = OUTPUT_MODELS.get(output_format)(input_model, cfg, True,
                                                    weight_path)
    output_model.export()

    # update `tensor_para_size` in `triton_models/interactive/config.pbtxt`
    with open(osp.join(triton_models_path, 'interactive/config.pbtxt'),
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
