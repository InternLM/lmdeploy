# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseChecker


class CudaChecker(BaseChecker):
    """Check pytorch is available."""

    def __init__(self, weight_quant: str = None, logger=None) -> None:
        super().__init__(logger=logger)
        self.weight_quant = weight_quant

    def check(self):
        """check."""
        import torch

        if not torch.cuda.is_available():
            self.log_and_exit(mod_name='CUDA', message='cuda is not available.')

        if self.weight_quant == 'fp8':
            props = torch.cuda.get_device_properties(0)
            if props.major < 9:
                self.log_and_exit(mod_name='CUDA',
                                  message='weight_quant=fp8 requires sm>=9.0.')
