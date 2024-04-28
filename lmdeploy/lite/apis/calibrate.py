# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path
from typing import Union

import torch
from torch import nn
from transformers import AutoTokenizer

from lmdeploy.lite.quantization import CalibrationContext
from lmdeploy.lite.utils import (collect_target_modules, get_calib_loaders,
                                 load_hf_from_pretrained)

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'InternLM2ForCausalLM': 'InternLM2DecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'Qwen2ForCausalLM': 'Qwen2DecoderLayer',
    'BaiChuanForCausalLM': 'DecoderLayer',  # Baichuan 7B
    'BaichuanForCausalLM': 'DecoderLayer',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}

NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'InternLM2ForCausalLM': 'InternLM2RMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'Qwen2ForCausalLM': 'Qwen2RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',  # Baichuan 7B
    'BaichuanForCausalLM': 'RMSNorm',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaRMSNorm',
}

HEAD_NAME_MAP = {
    'InternLMForCausalLM': 'lm_head',
    'InternLM2ForCausalLM': 'output',
    'QWenLMHeadModel': 'lm_head',
    'Qwen2ForCausalLM': 'lm_head',
    'BaiChuanForCausalLM': 'lm_head',  # Baichuan 7B
    'BaichuanForCausalLM': 'lm_head',  # Baichuan2 7B
    'LlamaForCausalLM': 'lm_head',
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
            raise TypeError(
                'layer_type should be a string (class name) or a type')

        # Check if the child contains the target module type
        contain_layer = len(
            collect_target_modules(child, layer_type, [head_name]).keys()) > 0

        # Check if the child matches the head name
        is_head = name == head_name

        mod_name = f'{prefix}.{name}' if prefix else name

        # If the child is either an instance of the layer type or has the
        # head name, move it to CPU, otherwise move it to the specified device
        if is_layer or is_head:
            child.to('cpu')
            print(f'Move {mod_name} to CPU.')
        elif contain_layer:
            _prepare_for_calibrate(child, layer_type, head_name, device,
                                   mod_name)
        else:
            child.to(device)
            print(f'Move {mod_name} to GPU.')


import logging
from typing import List, Union

from datasets import load_dataset


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = 'pileval',
    tokenizer=None,
    n_samples=512,
    block_size=512,
    split='train',
    text_column='text',
):
    if isinstance(data, str):
        if data == 'pileval':
            dataset = load_dataset('mit-han-lab/pile-val-backup',
                                   split='validation')
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                'Either pass a string to a huggingface dataset or a list'
                'that is preprocessed with one sample of text per element'
                ' or a list of list of int for tokenized words.')
    else:
        raise NotImplementedError(
            'Either pass a string to a huggingface dataset or a list'
            'that is preprocessed with one sample of text per element'
            ' or a list of list of int for tokenized words.')

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f' * Split into {n_split} blocks')
    samples = [
        cat_samples[:, i * block_size:(i + 1) * block_size]
        for i in range(n_split)
    ]
    return samples


def calibrate(model: str,
              calib_dataset: str = 'ptb',
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              work_dir: str = './work_dir',
              device: str = 'cuda') -> None:
    """The main function for loading the model and performing calibration on a
    given dataset.

    Args:
        model (str): The name or path of the model to be loaded.
        calib_dataset (str, optional): The calibration dataset name.
            Defaults to 'ptb'.
        calib_samples (int, optional): The number of samples for calibration.
            Defaults to 128.
        calib_seqlen (int, optional): The sequence length for calibration.
            Defaults to 2048.
        work_dir (str): The working directory for outputs.
            Defaults to './work_dir'.
        device (str, optional): The device to be used for calculation.
            Defaults to 'cuda'.

    Returns:
        model (nn.Module): The loaded huggingface model.
        tokenizer : The loaded hugginface tokenizer.
        work_dir (str): The working directory for outputs.
    """

    assert calib_dataset in ['c4', 'ptb', 'wikitext2', 'pileval'], \
        'Support only `c4`, `ptb`, `wikitext2` or `pileval`.'

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)

    model = load_hf_from_pretrained(model,
                                    torch_dtype=torch.float16,
                                    trust_remote_code=True)

    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(
            f'Currently, quantification and calibration of {model_type} are '
            f'not supported. The supported model types are '
            f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError(
                'When using Qwen, you need to `pip install flash-attn` first, '
                'otherwise calibration and quantification will not work '
                'properly.')

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    _prepare_for_calibrate(model, layer_type,
                           HEAD_NAME_MAP[type(model).__name__], device)

    print('Loading calibrate dataset ...')
    # calib_loader, _ = get_calib_loaders(calib_dataset,
    #                                     tokenizer,
    #                                     nsamples=calib_samples,
    #                                     seqlen=calib_seqlen)

    calib_loader = get_calib_dataset(
        data='pileval',
        tokenizer=tokenizer,
        n_samples=128,
        block_size=512,
        split='train',
        text_column='text',
    )
    # Initialize calibration context
    from lmdeploy.lite.quantization.calibration import AWQCalibrationContext
    calib_ctx = AWQCalibrationContext(model,
                                      tokenizer,
                                      layer_type=layer_type,
                                      norm_type=norm_type,
                                      device=device)

    with calib_ctx:
        all_data = torch.cat([
            data if isinstance(data, torch.Tensor) else data[0]
            for data in calib_loader
        ]).to(device)
        calib_ctx.calibrate(all_data)

    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calib_ctx.export(work_dir)

    return model, tokenizer, work_dir


if __name__ == '__main__':
    import fire

    fire.Fire(calibrate)
