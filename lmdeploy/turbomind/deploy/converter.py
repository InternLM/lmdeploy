# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .config import TurbomindModelConfig
from .module import Transformer
from .policy import get_input_policy
from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS, BaseOutputModel

SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'fp8', None]
logger = get_logger('lmdeploy')


def get_input_model_registered_name(model_path: str, model_format: str):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq']
    """
    arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    return register_name


def get_output_model_registered_name_and_config(model_path: str, model_format: str, dtype: str, group_size: int):
    """Get the registered name of the turbomind model and its configuration
    according to the input model path, format and user-input config. The name
    will be used to access the OUTPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq']
        dtype (str): the data type of the model's weights and activations
        group_size (int): the size of group used by awq model
    """
    register_name = 'tm'

    has_bf16 = is_bf16_supported()

    model_arch, model_config = get_model_arch(model_path)

    # infer dtype from device and model config
    if dtype == 'auto':
        # pick dtype by device as default
        dtype = 'bfloat16' if has_bf16 else 'float16'
        # dtype from model
        torch_dtype = getattr(model_config, 'torch_dtype', None)
        if not torch_dtype:
            if model_arch in ['QWenLMHeadModel', 'GptOssForCausalLM']:
                torch_dtype = torch.bfloat16
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(torch_dtype, dtype)

    if dtype == 'bfloat16' and not has_bf16:
        logger.warning('data type fallback to float16 since '
                       'torch.cuda.is_bf16_supported is False')
        dtype = 'float16'

    weight_type = dtype

    config = TurbomindModelConfig.from_dict()

    session_len = _get_and_verify_max_len(model_config, None)

    if model_format in ['awq', 'gptq', 'compressed-tensors']:
        weight_type = 'int4'
        dtype = 'float16'  # force float16 for int4 quantized weights
        group_size = 128 if group_size == 0 else group_size
        if model_format == 'compressed-tensors':
            model_format = 'awq'
    elif model_format == 'fp8':
        weight_type = 'fp8'
        group_size = 128
    elif model_format == 'mxfp4':
        weight_type = 'e2m1'
        group_size = 32

    expert_weight_type = weight_type

    # ONLY experts are in mxfp4
    if model_arch == 'GptOssForCausalLM':
        weight_type = dtype

    config.model_config.model_arch = model_arch
    config.model_config.data_type = dtype
    config.model_config.weight_type = weight_type
    config.model_config.expert_weight_type = expert_weight_type
    config.model_config.model_format = model_format
    config.model_config.group_size = group_size
    config.model_config.session_len = session_len

    return register_name, config


def get_tm_model(model_path,
                 model_name,
                 chat_template_name,
                 engine_config: TurbomindEngineConfig,
                 group_size: int = None,
                 out_dir: str = None) -> BaseOutputModel:
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
    quant_config = search_nested_config(cfg.to_dict(), 'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or engine_config.model_format == quant_method, (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (f'mismatched quant group size: user input "{group_size}" '
                                                             f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', ('compressed-tennsors only supports pack-quantized format, '
                                                 f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', ('pack-quantized requires 4-bit int, '
                                                       f'but got {_num_bits}-bit {_type}')
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    if engine_config.model_format in ['awq', 'gptq', 'compressed-tensors']:
        # Compatible to awq models that are quantized by lmdeploy (<=v0.3.0)
        if not group_size:
            group_size = 128
        assert group_size == 128, (f'model format is "{engine_config.model_format}" '
                                   f'but group_size is {group_size}. Currently, only 128 '
                                   'is supported')

    input_model_name = get_input_model_registered_name(model_path, engine_config.model_format)
    input_policy = get_input_policy(engine_config.model_format)
    input_model = INPUT_MODELS.get(input_model_name)(model_path=model_path,
                                                     tokenizer_path=model_path,
                                                     input_policy=input_policy)

    output_model_name, tm_cfg = get_output_model_registered_name_and_config(model_path=model_path,
                                                                            model_format=engine_config.model_format,
                                                                            dtype=engine_config.dtype,
                                                                            group_size=group_size)

    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name

    tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size
    tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size
    tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size

    output_model = OUTPUT_MODELS.get(output_model_name)(input_model=input_model,
                                                        cfg=tm_cfg,
                                                        model_cls=Transformer,
                                                        out_dir=out_dir)

    return output_model
