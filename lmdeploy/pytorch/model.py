# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dist import get_local_rank

logger = logging.getLogger(__name__)


class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_


def init_model(model_path: str,
               tokenizer_path: Optional[str] = None,
               use_fast_tokenizer=True):
    """Initialize model and tokenizer from given model path.

    Args:
        model_path (str): Path to model.
        tokenizer_path (str): Path to tokenizer.
        use_fast_tokenizer (bool): Whether to use fast tokenizer.

    Note:
        If the model is converted from new version of transformers,
            use_fast_tokenizer should be True.
        If using depodaca/llama-xb-hf, use_fast_tokenizer should be False.
    """

    start = time.monotonic()

    if not tokenizer_path:
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=use_fast_tokenizer,
                                              trust_remote_code=True)

    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)

    logger.info(f'Model loaded in {time.monotonic() - start:.1f} seconds')
    logger.info(f'Model loaded from {model_path}')
    logger.debug(model)

    return model, tokenizer


def accel_model(model,
                accel: Optional[str] = None,
                gpu_id=None,
                max_alloc=2048,
                tp_size=1):
    """Accelerate model with given accelerator.

    Note:
        Currently we support only deepspeed or just no acceleration.
    """

    logger.info(f'Accelerate model with {accel}')

    if accel is None:
        # No acceleration, just to cuda
        # assume single gpu single process
        # user is responsible to assign the gpu id via CUDA_VISIBLE_DEVICES # noqa: E501
        gpu_id = gpu_id if gpu_id is not None else get_local_rank()
        model = model.cuda(gpu_id)

    elif accel.lower() == 'deepspeed':
        # Use deepspeed inference inject fast kernel and/or tensor parallel

        try:
            import deepspeed
        except ImportError as e:
            raise ImportError('--accel=deepspeed is specified but '
                              'deepspeed is not installed.\n'
                              'Install with `pip install deepspeed`.') from e

        config = dict(
            tensor_parallel=dict(tp_size=tp_size),  # Use world size in general
            dtype=torch.float16,
            replace_with_kernel_inject=True,
            max_out_tokens=max_alloc,
        )

        if 'InternLM' in model.__class__.__name__:
            try:
                # Use customized deepspeed supporting InternLM
                # https://github.com/wangruohui/DeepSpeed/tree/support_internlm_0.10.0 (commit cdef2ce)  # noqa: E501
                from deepspeed.module_inject.containers.internlm import \
                    InternLMLayerPolicy  # noqa: E501
            except ImportError:
                # InternLM is not officially supported by DeepSpeed
                # Set replace_with_kernel_inject=False to use AutoTP
                config.update({'replace_with_kernel_inject': False})
                warnings.warn(
                    '\033[0;93m'
                    'Current installation of deepspeed does not '
                    'support InternLM. Disable kernel injection. '
                    'To support InternLM, install customized deepspeed with '
                    '`pip install git+https://github.com/wangruohui/DeepSpeed@support_internlm_0.10.0`'  # noqa: E501
                    '\033[0m')
            else:
                for module in model.modules():
                    # Since remote code is dynamically located,
                    # we need to do this dynamically
                    if module.__class__.__name__ == 'InternLMDecoderLayer':
                        InternLMLayerPolicy._orig_layer_class = module.__class__  # noqa: E501
                        break

        logger.debug(f'Using deepspeed config\n{config}')

        model = deepspeed.init_inference(
            model=model,  # Transformers models
            config=config,
        )

        # for k, v in model.named_parameters():
        #     logger.debug(f"{k}: v.device")
    else:
        raise ValueError(f'Unsupported accelerator {accel}.')

    logger.debug(model)

    return model
