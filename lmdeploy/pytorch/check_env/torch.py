# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseChecker


class TorchChecker(BaseChecker):
    """Check pytorch is available."""

    def __init__(self, device: str = 'cuda', logger=None) -> None:
        super().__init__(logger=logger)
        self.device = device

    def check(self):
        """check."""
        try:
            import torch
            a = torch.tensor([1, 2], device=self.device)
            b = a.new_tensor([3, 4], device=self.device)
            c = a + b
            torch.testing.assert_close(c, a.new_tensor([4, 6]))
        except Exception as e:
            self.log_and_exit(e, 'PyTorch', 'PyTorch is not available.')
