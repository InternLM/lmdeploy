# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger

from lmdeploy.utils import get_logger


def _handle_exception(e: Exception, mod_name: str, logger: Logger):
    logger.debug('Exception', exc_info=1)
    logger.error(f'{type(e).__name__}: {e}')
    logger.error(f'<{mod_name}> environment test failed. '
                 'Please ensure it has been installed correctly.')
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
