# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil

import fire
import torch

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.model import MODELS, best_match_model
from lmdeploy.utils import get_logger, get_model

from ...utils import _get_and_verify_max_len
from ..supported_models import SUPPORTED_ARCHS, is_supported
from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS, TurbomindModelConfig

SUPPORTED_FORMATS = ['meta_llama', 'hf', 'awq', None]
logger = get_logger('lmdeploy')


def get_input_model_registered_name(model_path: str, model_format: str):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['meta_llama',  'hf', 'awq']
    """
    arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    if model_format == 'awq':
        register_name = register_name + '-awq'
    return register_name


def create_workspace(_path: str):
    """Create a workspace.

    Args:
        _path (str): the path of the workspace
    """
    if osp.exists(_path):
        print(f'remove workspace in directory {_path}')
        shutil.rmtree(_path)
    print(f'create workspace in directory {_path}')
    weight_path = osp.join(_path, 'triton_models', 'weights')
    tokenizer_path = osp.join(_path, 'triton_models', 'tokenizer')
    os.makedirs(weight_path)
    os.makedirs(tokenizer_path)
    return weight_path, tokenizer_path


def copy_tokenizer(model_path: str, tokenizer_path: str,
                   tm_tokenizer_path: str):
    """Copy tokenizer."""

    if tokenizer_path is not None:
        assert osp.exists(tokenizer_path), f'{tokenizer_path} does not exists.'

        shutil.copy(tokenizer_path,
                    osp.join(tm_tokenizer_path, osp.basename(tokenizer_path)))
    else:
        from transformers import AutoTokenizer
        try:
            _ = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
        except Exception as e:
            assert 0, f'{e}'

    # move tokenizer model to the target path
    candidate = ['tokenizer.model', 'qwen.tiktoken', 'merges.txt']
    for name in candidate:
        tmp_path = osp.join(model_path, name)
        if osp.exists(tmp_path):
            shutil.copy(tmp_path, osp.join(tm_tokenizer_path, name))
    # copy py/json files that are related to tokenizer to the target path
    for _file in os.listdir(model_path):
        if _file.endswith('.json') or _file.endswith('.py'):
            json_path = osp.join(model_path, _file)
            shutil.copy(json_path, osp.join(tm_tokenizer_path, _file))


def get_output_model_registered_name_and_config(model_path: str,
                                                model_format: str,
                                                group_size: int):
    """Get the registered name of the turbomind model and its configuration
    according to the input model path, format and user-input config. The name
    will be used to access the OUTPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['meta_llama',  'hf', 'awq']
        group_size (int): the size of group used by awq model
    """
    register_name = 'fp16'
    turbomind_model_arch = 'llama'
    weight_type = 'fp16'

    config = TurbomindModelConfig.from_dict({}, allow_none=True)

    if model_format == 'meta_llama':
        session_len = 2048
    else:  # hf, awq, None
        register_name = 'fp16'
        model_arch, model_config = get_model_arch(model_path)
        turbomind_model_arch = SUPPORTED_ARCHS[model_arch]
        session_len = _get_and_verify_max_len(model_config, None)
        if model_format == 'awq':
            weight_type = 'int4'
            register_name = 'plora-w4' \
                if turbomind_model_arch == 'xcomposer2' else 'w4'
            group_size = 128 if group_size == 0 else group_size
        else:
            torch_dtype = getattr(model_config, 'torch_dtype', 'float16')
            TORCH_DTYPE_MAP = {torch.bfloat16: 'bf16', torch.float16: 'fp16'}
            weight_type = TORCH_DTYPE_MAP.get(torch_dtype, 'fp16')

            # Qwen-1 didn't set torch_dtype. It used bf16 as default
            if model_arch == 'QWenLMHeadModel':
                weight_type = 'bf16'
            if not torch.cuda.is_bf16_supported():
                print(
                    'Device does not support bfloat16. Set float16 forcefully')
                weight_type = 'fp16'

            register_name = weight_type
            if turbomind_model_arch == 'xcomposer2':
                register_name = 'plora'

    config.model_arch = model_arch
    config.session_len = session_len + 8
    config.weight_type = weight_type
    config.group_size = group_size

    return register_name, config


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


def get_tm_model(model_path,
                 model_name,
                 chat_template_name,
                 group_size,
                 engine_config,
                 out_dir: str = None):
    # TODO: open the following condition check in another PR,
    # CLI needs to be updated
    # if model_format == 'awq' and group_size <= 0:
    #     raise RuntimeError(
    #         'group_size should be specified when the model is awq')

    input_model_name = get_input_model_registered_name(
        model_path, engine_config.model_format)
    input_model = INPUT_MODELS.get(input_model_name)(model_path=model_path,
                                                     tokenizer_path=model_path)

    output_model_name, cfg = get_output_model_registered_name_and_config(
        model_path=model_path,
        model_format=engine_config.model_format,
        group_size=group_size)

    cfg.chat_template = chat_template_name
    cfg.model_name = model_name
    cfg.update_from_engine_config(engine_config)

    output_model = OUTPUT_MODELS.get(output_model_name)(
        input_model=input_model, cfg=cfg, out_dir=out_dir)

    return output_model


def main(model_name: str,
         model_path: str,
         model_format: str = None,
         chat_template: str = None,
         tokenizer_path: str = None,
         dst_path: str = 'workspace',
         tp: int = 1,
         group_size: int = 0,
         revision: str = None,
         download_dir: str = None,
         **kwargs):
    """deploy llama family models via turbomind.

    Args:
        model_name (str): unused any longer
        model_path (str): the directory path of the model
        model_format (str): the format of the model, should choose from
            ['meta_llama', 'hf', 'awq', None]. 'meta_llama' stands for META's
            llama format, 'hf' means huggingface llama format, and 'awq' means
            llama(hf) model quantized by lmdeploy/lite/quantization/awq.py.
            The default value is None
        chat_template (str): the name of the built-in chat template.
        tokenizer_path (str): the path of tokenizer model
        dst_path (str): the destination path that saves outputs
        tp (int): the number of GPUs used for tensor parallelism, should be 2^n
        quant_path (str): Path of the quantized model, which can be None.
        group_size (int): a parameter used in AWQ to quantize fp16 weights
            to 4 bits
        revision (str): The specific model version to use. It can be a branch
            name, a tag name, or a commit id. If unspecified, will use
            the default version.
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
        kwargs (dict): other params for convert
    """
    if model_name:
        logger.warning(
            'The argument `<model_name>` is deprecated and unused now. '
            'It will be removed on 2024.12.31. It was originally used to '
            'specify the name of the built-in chat template, but now it '
            'is substituted with a clearer parameter `--chat-template`')
    if chat_template is None:
        chat_template = best_match_model(model_path)
    assert chat_template in MODELS.module_dict.keys(), \
        f"chat template '{chat_template}' is not a built-in template. " \
        f'The built-ins are: {MODELS.module_dict.keys()}'
    assert is_supported(model_path), (
        f'turbomind does not support {model_path}. '
        'Plz try pytorch engine instead.')

    assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'

    assert model_format in SUPPORTED_FORMATS, 'the model format ' \
        f'should be in {SUPPORTED_FORMATS}'

    if not os.path.exists(model_path):
        print(f"can't find model from local_path {model_path}, "
              'try to download from huggingface')
        model_path = get_model(
            model_path,
            revision=revision,
            download_dir=download_dir,
        )
        print(f'load model from {model_path}')

    tm_weight_path, tm_tokenizer_path = create_workspace(dst_path)
    copy_tokenizer(model_path, tokenizer_path, tm_tokenizer_path)

    engine_config = TurbomindEngineConfig(tp=tp, model_format=model_format)
    tm_model = get_tm_model(model_path, model_name, chat_template, group_size,
                            engine_config, tm_weight_path)
    tm_model.export()


if __name__ == '__main__':
    fire.Fire(main)
