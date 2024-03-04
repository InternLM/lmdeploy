# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger

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
                 f'<{mod_name}> test failed!\n {message}'
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


def check_transformers_version(model_path: str):
    """check transformers version."""
    from packaging import version
    logger = get_logger('lmdeploy')
    logger.debug('Checking <transformers> version.')

    trans_version = None
    try:
        import transformers
        trans_version = version.parse(transformers.__version__)
    except Exception as e:
        _handle_exception(e, 'transformers', logger)

    model_trans_version = None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_trans_version = getattr(config, 'transformers_version')
    except Exception as e:
        message = (
            f'Load model config with transformers=={trans_version} failed. '
            'Please make sure model can be loaded with transformers API.')
        _handle_exception(e, 'transformers', logger, message=message)

    try:
        model_trans_version = version.parse(model_trans_version)
        assert trans_version >= model_trans_version
    except Exception as e:
        message = {
            f'model `{model_path}` requires '
            f'transformers version {model_trans_version} '
            f'but transformers {trans_version} is installed.'
        }
        _handle_exception(e, 'transformers', logger, message=message)


def check_model(model_path: str):
    """check model requirements."""
    logger = get_logger('lmdeploy')
    logger.info('Checking model.')
    check_transformers_version(model_path)
