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
from .config import TurbomindModelConfig
from .exporter import get_exporter_factory
from .policy import get_input_policy
from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS

SUPPORTED_FORMATS = ['meta_llama', 'hf', 'awq', 'gptq', None]
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
                                                model_format: str, dtype: str,
                                                group_size: int):
    """Get the registered name of the turbomind model and its configuration
    according to the input model path, format and user-input config. The name
    will be used to access the OUTPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['meta_llama',  'hf', 'awq', 'gptq']
        dtype (str): the data type of the model's weights and activations
        group_size (int): the size of group used by awq model
    """
    register_name = 'tm'
    turbomind_model_arch = 'llama'
    weight_type = 'float16'

    config = TurbomindModelConfig.from_dict()

    if model_format == 'meta_llama':
        session_len = 2048
    else:  # hf, awq, None
        model_arch, model_config = get_model_arch(model_path)
        turbomind_model_arch = SUPPORTED_ARCHS[model_arch]
        session_len = _get_and_verify_max_len(model_config, None)
        if model_format in ['awq', 'gptq']:
            weight_type = 'int4'
            group_size = 128 if group_size == 0 else group_size
        else:
            torch_dtype = getattr(model_config, 'torch_dtype', 'float16')
            TORCH_DTYPE_MAP = {
                torch.bfloat16: 'bfloat16',
                torch.float16: 'float16'
            }
            weight_type = TORCH_DTYPE_MAP.get(torch_dtype, 'float16')

            # Qwen-1 didn't set torch_dtype. It used bf16 as default
            if model_arch == 'QWenLMHeadModel':
                weight_type = 'bfloat16'

    if dtype == 'auto':
        weight_type = weight_type if weight_type in [
            'float16', 'bfloat16', 'int4'
        ] else 'float16'
    elif dtype in ['float16', 'bfloat16']:
        if weight_type == 'int4':
            logger.warn(f'The model {model_path} is a quantized model, so the '
                        f'specified data type {dtype} is ignored')
        else:
            weight_type = dtype
    else:
        assert 0, f'unsupported specified data type {dtype}'

    config.model_config.model_arch = model_arch
    config.model_config.weight_type = weight_type
    config.model_config.model_format = model_format
    config.model_config.group_size = group_size
    config.model_config.session_len = session_len

    lora_type = 'plora' if turbomind_model_arch == 'xcomposer2' else ''

    exporter_factory = get_exporter_factory(weight_type, lora_type)

    return register_name, config, exporter_factory


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


def find_quantization_config(nested, target_key):
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key == target_key:
                return value
            if isinstance(value, (dict, list)):
                result = find_quantization_config(value, target_key)
                if result is not None:
                    return result
    elif isinstance(nested, list):
        for item in nested:
            result = find_quantization_config(item, target_key)
            if result is not None:
                return result
    return None


def get_tm_model(model_path,
                 model_name,
                 chat_template_name,
                 engine_config: TurbomindEngineConfig,
                 group_size: int = None,
                 out_dir: str = None):
    """Create turbomind model.

    Args:
        model_path (str): the path of the input model, which is supposed
            to be a local path, or huggingface hub repo_id, or modelscope
            hub repo_id
        model_name (str): user customized model name
        chat_template_name (str): the name of the chat template of
            the input model
        engine_config(TurbomindEngineConfig): user input engine config
        group_size(int): refers to the group_size if the input model
            is a w4a16(awq or gptq) quantized model
        out_dir(str): the output directory where to save to turbomind model.
            If it is None, the turbomind model won't be saved
    """
    _, cfg = get_model_arch(model_path)
    quant_config = find_quantization_config(cfg.to_dict(),
                                            'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or \
            engine_config.model_format == quant_method, \
            f'mismatched quant method: user input ' \
            f'"{engine_config.model_format}" ' \
            f'vs model quant_config "{quant_method}"'
        assert group_size is None or group_size == _group_size, \
            f'mismatched quant group size: user input "{group_size}" ' \
            f'vs model quant_config "{_group_size}"'

        if quant_method == 'awq':
            assert version == 'gemm', \
                f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and \
                quant_config.get('sym', True), \
                f'unsupported quant config: {quant_config}'
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    # Compatible to awq models that are quantized by lmdeploy (<=v0.3.0)
    if not group_size:
        group_size = 128

    if engine_config.model_format in ['awq', 'gptq']:
        assert group_size == 128, \
            f'model format is "{engine_config.model_format}" ' \
            f'but group_size is {group_size}. Currently, only 128 ' \
            'is supported'

    input_model_name = get_input_model_registered_name(
        model_path, engine_config.model_format)
    input_policy = get_input_policy(engine_config.model_format)
    input_model = INPUT_MODELS.get(input_model_name)(model_path=model_path,
                                                     tokenizer_path=model_path,
                                                     input_policy=input_policy)

    output_model_name, tm_cfg, exporter_factory = \
        get_output_model_registered_name_and_config(
            model_path=model_path,
            model_format=engine_config.model_format,
            dtype=engine_config.dtype,
            group_size=group_size)

    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name
    tm_cfg.model_config.tp = engine_config.tp

    output_model = OUTPUT_MODELS.get(output_model_name)(
        input_model=input_model,
        cfg=tm_cfg,
        exporter_factory=exporter_factory,
        out_dir=out_dir)

    return output_model


def main(model_name: str,
         model_path: str,
         model_format: str = 'hf',
         dtype: str = 'auto',
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
            ['meta_llama', 'hf', 'awq', 'gptq']. 'meta_llama' stands for META's
            llama format, 'hf' means huggingface model, and 'awq', `gptq`
            means models quantized by `autoawq` and `autogptq` respectively.
            The default value is hf
        dtype (str): data type for model weights and activations. It can be
            one of the following values, ['auto', 'float16', 'bfloat16']
            The `auto` option will use FP16 precision for FP32 and FP16
            models, and BF16 precision for BF16 models.
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
    engine_config = TurbomindEngineConfig(tp=tp,
                                          model_format=model_format,
                                          dtype=dtype)
    tm_model = get_tm_model(model_path, model_name, chat_template,
                            engine_config, group_size, tm_weight_path)
    tm_model.export()


if __name__ == '__main__':
    fire.Fire(main)
