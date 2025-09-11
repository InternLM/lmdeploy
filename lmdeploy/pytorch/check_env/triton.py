# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version

from .base import BaseChecker

MAX_TRITON_VERSION = '3.4.0'
MIN_TRITON_VERSION = '3.0.0'


class TritonChecker(BaseChecker):
    """Check triton is available."""

    def check_version(self):
        """Check version."""
        logger = self.get_logger()

        # version check
        import triton
        max_version = version.parse(MAX_TRITON_VERSION)
        min_version = version.parse(MIN_TRITON_VERSION)
        triton_version = version.parse(triton.__version__)

        if triton_version > max_version:
            logger.warning('PytorchEngine has not been tested on '
                           f'triton>{MAX_TRITON_VERSION}.')
        if triton_version < min_version:
            msg = (f'triton>={MIN_TRITON_VERSION} is required. '
                   f'Found triton=={triton_version}')
            self.log_and_exit(mod_name='Triton', message=msg)

    def check(self):
        """check."""
        logger = self.get_logger()

        msg = (
            'Please ensure that your device is functioning properly with <Triton>.\n'  # noqa: E501
            'You can verify your environment by running '
            '`python -m lmdeploy.pytorch.check_env.triton_custom_add`.')
        try:
            logger.debug('Checking <Triton> environment.')
            import torch

            from .triton_custom_add import custom_add
            a = torch.tensor([1, 2], device='cuda')
            b = a.new_tensor([3, 4], device='cuda')
            c = custom_add(a, b)
            torch.testing.assert_close(c, a + b)
        except RuntimeError as e:
            ptxas_error = 'device kernel image is invalid'
            if len(e.args) > 0 and ptxas_error in e.args[0]:
                msg = (
                    'This Error might caused by mismatching between NVIDIA Driver and nvcc compiler. \n'  # noqa: E501
                    'Try solution https://github.com/triton-lang/triton/issues/1955#issuecomment-1929908209'  # noqa: E501
                    ' or reinstall the driver.')
            self.log_and_exit(e, 'Triton', msg)
        except Exception as e:
            self.log_and_exit(e, 'Triton', msg)

        # version check
        self.check_version()
