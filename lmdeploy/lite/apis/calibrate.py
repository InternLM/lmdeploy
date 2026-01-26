# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Literal, Union

import torch
from torch import nn
from transformers import AutoTokenizer

from lmdeploy.archs import get_task
from lmdeploy.lite.quantization import CalibrationContext, CalibrationContextV2
from lmdeploy.lite.utils import collect_target_modules, get_calib_loaders, load_hf_from_pretrained
from lmdeploy.vl.model.builder import load_vl_model

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'InternLM2ForCausalLM': 'InternLM2DecoderLayer',
    'InternLM3ForCausalLM': 'InternLM3DecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'Qwen2ForCausalLM': 'Qwen2DecoderLayer',
    'Qwen3ForCausalLM': 'Qwen3DecoderLayer',
    'BaiChuanForCausalLM': 'DecoderLayer',  # Baichuan 7B
    'BaichuanForCausalLM': 'DecoderLayer',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaDecoderLayer',
    'LlavaLlamaForCausalLM': 'LlamaDecoderLayer',
    'MGMLlamaForCausalLM': 'LlamaDecoderLayer',  # mini gemini
    'InternLMXComposer2ForCausalLM': 'InternLM2DecoderLayer',
    'Phi3ForCausalLM': 'Phi3DecoderLayer',
    'ChatGLMForConditionalGeneration': 'GLMBlock',
    'MixtralForCausalLM': 'MixtralDecoderLayer',
    'Qwen2VLForConditionalGeneration': 'Qwen2VLDecoderLayer',
    'Qwen2_5_VLForConditionalGeneration': 'Qwen2_5_VLDecoderLayer',
    'MistralForCausalLM': 'MistralDecoderLayer',
}

NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'InternLM2ForCausalLM': 'InternLM2RMSNorm',
    'InternLM3ForCausalLM': 'InternLM3RMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'Qwen2ForCausalLM': 'Qwen2RMSNorm',
    'Qwen3ForCausalLM': 'Qwen3RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',  # Baichuan 7B
    'BaichuanForCausalLM': 'RMSNorm',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaRMSNorm',
    'LlavaLlamaForCausalLM': 'LlamaRMSNorm',
    'MGMLlamaForCausalLM': 'LlamaRMSNorm',  # mini gemini
    'InternLMXComposer2ForCausalLM': 'InternLM2RMSNorm',
    'Phi3ForCausalLM': 'Phi3RMSNorm',
    'ChatGLMForConditionalGeneration': 'RMSNorm',
    'MixtralForCausalLM': 'MixtralRMSNorm',
    'Qwen2VLForConditionalGeneration': 'Qwen2RMSNorm',
    'Qwen2_5_VLForConditionalGeneration': 'Qwen2RMSNorm',
    'MistralForCausalLM': 'MistralRMSNorm',
}

HEAD_NAME_MAP = {
    'InternLMForCausalLM': 'lm_head',
    'InternLM2ForCausalLM': 'output',
    'InternLM3ForCausalLM': 'output',
    'QWenLMHeadModel': 'lm_head',
    'Qwen2ForCausalLM': 'lm_head',
    'Qwen3ForCausalLM': 'lm_head',
    'BaiChuanForCausalLM': 'lm_head',  # Baichuan 7B
    'BaichuanForCausalLM': 'lm_head',  # Baichuan2 7B
    'LlamaForCausalLM': 'lm_head',
    'LlavaLlamaForCausalLM': 'lm_head',
    'MGMLlamaForCausalLM': 'lm_head',  # mini gemini
    'InternLMXComposer2ForCausalLM': 'output',
    'Phi3ForCausalLM': 'lm_head',
    'ChatGLMForConditionalGeneration': 'output_layer',
    'MixtralForCausalLM': 'lm_head',
    'Qwen2VLForConditionalGeneration': 'lm_head',
    'Qwen2_5_VLForConditionalGeneration': 'lm_head',
    'MistralForCausalLM': 'lm_head',
}


def _prepare_for_calibrate(model: nn.Module,
                           layer_type: Union[str, type],
                           head_name: str = 'lm_head',
                           device: str = 'cuda',
                           prefix: str = '') -> None:
    """Prepare the model for calibration by moving specific modules to CPU.

    This function goes through each child of a given model and checks whether
    it is an instance of a certain layer type or has the name equal to
    `head_name`.
    If yes, it moves the module to CPU, otherwise to the specified device
    (default is CUDA).

    If the child contains the target layer type in its sub-modules, the
    function performs the same operation recursively.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to prepare for calibration.
    layer_type : Union[str, Type]
        The type of the layer to be moved to CPU. Can be either a string of
        class name or the class type itself.
    head_name : str, optional
        The name of the module to be moved to CPU. Default is 'lm_head'.
    device : str, optional
        The device to which modules not matching the `layer_type` or
        `head_name` will be moved. Default is 'cuda'.
    prefix : str, optional
        The prefix used when printing the names of the moved modules.
        Default is ''.

    Raises
    ------
    TypeError
        If `layer_type` is neither a string nor a type.
    """

    for name, child in model.named_children():

        # Check if the child is an instance of the given layer type
        if isinstance(layer_type, str):
            is_layer = type(child).__name__ == layer_type
        elif isinstance(layer_type, type):
            is_layer = isinstance(child, layer_type)
        else:
            raise TypeError('layer_type should be a string (class name) or a type')

        # Check if the child contains the target module type
        contain_layer = len(collect_target_modules(child, layer_type, [head_name]).keys()) > 0

        # Check if the child matches the head name
        is_head = name == head_name
        # skip moving head layer to CPU when tie_word_embeddings is True
        is_head = is_head and not getattr(model.config, 'tie_word_embeddings', False)

        mod_name = f'{prefix}.{name}' if prefix else name

        # If the child is either an instance of the layer type or has the
        # head name, move it to CPU, otherwise move it to the specified device
        if is_layer or is_head:
            child.to('cpu')
            print(f'Move {mod_name} to CPU.')
        elif contain_layer:
            _prepare_for_calibrate(child, layer_type, head_name, device, mod_name)
        else:
            child.to(device)
            print(f'Move {mod_name} to GPU.')


# TODO to be removed
def make_compatible_internvl_config(model_path):
    """Patch model.config since after transformers v4.45.0, InternVL models
    can't use `save_pretrained`"""
    from lmdeploy.archs import get_model_arch
    arch, _ = get_model_arch(model_path)
    if arch == 'InternVLChatModel':
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.45.0'):

            def _get_non_default_generation_parameters(self):
                return {}

            from transformers import PretrainedConfig
            PretrainedConfig._get_non_default_generation_parameters = _get_non_default_generation_parameters  # noqa


def update_moe_mapping(model, model_type):
    """Update moe mapping."""
    from lmdeploy.lite.quantization.awq import FC_FCS_MAP, NORM_FCS_MAP

    # get experts num
    num_experts = 0
    for n, m in model.named_modules():
        if type(m).__name__ == LAYER_TYPE_MAP[model_type]:
            fc2fcs = FC_FCS_MAP[LAYER_TYPE_MAP[model_type]]
            for k, v in fc2fcs.items():
                if '{i}' in k:
                    break
            num_experts = len(m.get_submodule(k.split('.{i}')[0]))
            break

    # update FC_FCS_MAP
    updated_fc2fcs = dict()
    for prev_fc, post_fc in fc2fcs.items():
        if '{i}' in prev_fc:
            for i in range(num_experts):
                updated_fc2fcs.update({prev_fc.format(i=i): [v.format(i=i) for v in post_fc]})
        else:
            updated_fc2fcs.update({prev_fc: post_fc})
    FC_FCS_MAP[LAYER_TYPE_MAP[model_type]] = updated_fc2fcs
    # update NORM_FCS_MAP
    norm2fcs = NORM_FCS_MAP[LAYER_TYPE_MAP[model_type]]
    updated_norm2fcs = dict()
    for norm, fc in norm2fcs.items():
        updated_norm2fcs.update({norm: list(set([v.format(i=i) for v in fc for i in range(num_experts)]))})
    NORM_FCS_MAP[LAYER_TYPE_MAP[model_type]] = updated_norm2fcs


def calibrate(model: str,
              calib_dataset: str = 'wikitext2',
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              work_dir: str = './work_dir',
              device: str = 'cuda',
              w_bits: int = 4,
              w_group_size: int = 128,
              search_scale: bool = False,
              dtype: Literal['float16', 'bfloat16', 'auto'] = 'auto',
              batch_size: int = 1) -> None:
    """The main function for loading the model and performing calibration on a
    given dataset.

    Args:
        model (str): The name or path of the model to be loaded.
        calib_dataset (str, optional): The calibration dataset name.
            Defaults to 'wikitext2'.
        calib_samples (int, optional): The number of samples for calibration.
            Defaults to 128.
        calib_seqlen (int, optional): The sequence length for calibration.
            Defaults to 2048.
        work_dir (str): The working directory for outputs.
            Defaults to './work_dir'.
        device (str, optional): The device to be used for calculation.
            Defaults to 'cuda'.
        w_bits (int): Bit number for weight quantization.
        w_group_size (int): Group size for weight quantization statistics.
        search_scale (bool): Whether search scale ratio. Default to False,
            which means only smooth quant with 0.5 ratio will be applied.
        dtype (str): Data type for loading model weights and calib infer.
        batch_size (int): The batch size for running the calib samples.
            Low GPU mem requires small batch_size. Large batch_size
            reduces the calibration time while costs more VRAM.

    Returns:
        model (nn.Module): The loaded huggingface model.
        tokenizer : The loaded hugginface tokenizer.
        work_dir (str): The working directory for outputs.
    """

    assert calib_dataset in ['wikitext2', 'c4', 'pileval',
                             'gsm8k', 'neuralmagic_calibration', 'open-platypus', 'openwebtext'], \
        'Support only `wikitext2`, `c4`, `pileval`, `gsm8k`, ' \
        '`neuralmagic_calibration`, `open-platypus`, `openwebtext`.'

    model_type, _ = get_task(model)
    make_compatible_internvl_config(model)

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    if model_type == 'llm':
        model = load_hf_from_pretrained(model, dtype=dtype, trust_remote_code=True)
        vl_model = None
    elif model_type == 'vlm':
        vl_model = load_vl_model(model, backend=None, with_llm=True).vl_model
        model = vl_model
        if hasattr(vl_model, 'language_model'):  # deepseek-vl, ...
            model = vl_model.language_model
        if hasattr(vl_model, 'llm'):  # MiniCPMV, ...
            model = vl_model.llm
        model.config.use_cache = False
        if dtype == 'float16':
            model.half()
        elif dtype == 'bfloat16':
            assert torch.cuda.is_bf16_supported(
            ), 'your device does not support bfloat16 please set --dtype float16'  # noqa
            model.to(torch.bfloat16)
        elif dtype == 'auto' and model.config.torch_dtype == torch.bfloat16:
            print('Warning: we cast model to float16 to prevent OOM. You'
                  ' may enforce it bfloat16 by `--dtype bfloat16`')
            model.half()
        model.eval()

    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(f'Currently, quantification and calibration of {model_type} are '
                           f'not supported. The supported model types are '
                           f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type in ['MixtralForCausalLM']:
        update_moe_mapping(model, model_type)

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError('When using Qwen, you need to `pip install flash-attn` first, '
                               'otherwise calibration and quantification will not work '
                               'properly.')

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    _prepare_for_calibrate(model, layer_type, HEAD_NAME_MAP[type(model).__name__], device)

    print('Loading calibrate dataset ...')
    calib_loader = get_calib_loaders(calib_dataset, tokenizer, nsamples=calib_samples, seqlen=calib_seqlen)

    # Initialize calibration context
    if search_scale:
        calib_ctx = CalibrationContextV2(model,
                                         tokenizer,
                                         layer_type=layer_type,
                                         norm_type=norm_type,
                                         device=device,
                                         w_bits=w_bits,
                                         w_group_size=w_group_size,
                                         batch_size=batch_size,
                                         search_scale=search_scale)
    else:
        calib_ctx = CalibrationContext(model,
                                       tokenizer,
                                       layer_type=layer_type,
                                       norm_type=norm_type,
                                       batch_size=batch_size,
                                       device=device)

    with calib_ctx:
        all_data = torch.cat(calib_loader).to(device)
        calib_ctx.calibrate(all_data)

    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calib_ctx.export(work_dir)

    return vl_model, model, tokenizer, work_dir


if __name__ == '__main__':
    import fire

    fire.Fire(calibrate)
