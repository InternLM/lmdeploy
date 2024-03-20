# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger
from typing import List

from lmdeploy.utils import get_logger


def _handle_exception(e: Exception,
                      mod_name: str,
                      logger: Logger,
                      message: str = None):
    red_color = '\033[31m'
    reset_color = '\033[0m'
    if message is None:
        message = 'Please ensure it has been installed correctly.'
    logger.debug('Exception', exc_info=1)
    logger.error(f'{type(e).__name__}: {e}')
    logger.error(f'{red_color}'
                 f'<{mod_name}> test failed!\n'
                 f'{message}'
                 f'{reset_color}')
    exit(1)


def check_env_torch():
    """check PyTorch environment."""
    logger = get_logger('lmdeploy')

    try:
        logger.debug('Checking <PyTorch> environment.')
        import torch

        a = torch.tensor([1, 2], device='cuda')
        b = a.new_tensor([3, 4], device='cuda')
        c = a + b
        torch.testing.assert_close(c, a.new_tensor([4, 6]))
    except Exception as e:
        _handle_exception(e, 'PyTorch', logger)


def check_env_triton():
    """check OpenAI Triton environment."""
    logger = get_logger('lmdeploy')

    try:
        logger.debug('Checking <Triton> environment.')
        import torch

        from .triton_custom_add import custom_add
        a = torch.tensor([1, 2], device='cuda')
        b = a.new_tensor([3, 4], device='cuda')
        c = custom_add(a, b)
        torch.testing.assert_close(c, a + b)
    except Exception as e:
        _handle_exception(e, 'Triton', logger)


def check_env():
    """check all environment."""
    logger = get_logger('lmdeploy')
    logger.info('Checking environment for PyTorch Engine.')
    check_env_torch()
    check_env_triton()


def check_transformers_version(model_path: str,
                               trust_remote_code: bool = True):
    """check transformers version."""
    from packaging import version
    logger = get_logger('lmdeploy')

    def __check_transformers_version():
        """check transformers version."""
        logger.debug('Checking <transformers> version.')
        trans_version = None
        try:
            import transformers
            trans_version = version.parse(transformers.__version__)
        except Exception as e:
            _handle_exception(e, 'transformers', logger)
        return transformers, trans_version

    def __check_config(trans_version):
        """check config."""
        logger.debug('Checking <Model> AutoConfig.from_pretrained.')
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            message = (
                f'Load model config with transformers=={trans_version}'
                ' failed. '
                'Please make sure model can be loaded with transformers API.')
            _handle_exception(e, 'transformers', logger, message=message)
        return config

    def __check_model_transformers_version(config, trans_version):
        """check model transformers version."""
        logger.debug('Checking <Model> required transformers version.')
        try:
            model_trans_version = getattr(config, 'transformers_version')
            model_trans_version = version.parse(model_trans_version)
            assert trans_version >= model_trans_version, 'Version mismatch.'
        except Exception as e:
            message = (f'model `{model_path}` requires '
                       f'transformers version {model_trans_version} '
                       f'but transformers {trans_version} is installed.')
            _handle_exception(e, 'transformers', logger, message=message)

    def __check_model_dtype_support(config):
        """Checking model dtype support."""
        logger.debug('Checking <Model> dtype support.')

        import torch

        from lmdeploy.pytorch.config import ModelConfig

        try:
            model_config = ModelConfig.from_hf_config(config,
                                                      model_path=model_path)
            if model_config.dtype == torch.bfloat16:
                assert torch.cuda.is_bf16_supported(), (
                    'bf16 is not supported on your device')
        except AssertionError as e:
            message = (f'Your device does not support `{model_config.dtype}`. '
                       'Try edit `torch_dtype` in `config.json`.\n'
                       'Note that this might have negative effect!')
            _handle_exception(e, 'Model', logger, message=message)
        except Exception as e:
            message = (f'Checking failed with error {e}',
                       'Please send issue to LMDeploy with error logs.')
            _handle_exception(e, 'Model', logger, message=message)

        return model_config

    _, trans_version = __check_transformers_version()
    config = __check_config(trans_version)
    __check_model_transformers_version(config, trans_version)
    __check_model_dtype_support(config)


def check_model(model_path: str, trust_remote_code: bool = True):
    """check model requirements."""
    logger = get_logger('lmdeploy')
    logger.info('Checking model.')
    check_transformers_version(model_path, trust_remote_code)


def check_adapter(path: str):
    """check adapter."""
    logger = get_logger('lmdeploy')
    logger.debug(f'Checking <Adapter>: {path}.')

    try:
        from peft import PeftConfig
        PeftConfig.from_pretrained(path)
    except Exception as e:
        message = ('Please make sure the adapter can be loaded with '
                   '`peft.PeftConfig.from_pretrained`\n')
        err_msg = '' if len(e.args) == 0 else e.args[0]
        if 'got an unexpected keyword argument' in err_msg:
            message += ('Or try remove all unexpected keywords '
                        'in `adapter_config.json`.')
        _handle_exception(e, 'Model', logger, message=message)


def check_adapters(adapter_paths: List[str]):
    """check adapters."""
    if len(adapter_paths) <= 0:
        return
    logger = get_logger('lmdeploy')
    logger.info('Checking adapters.')
    for path in adapter_paths:
        check_adapter(path)
