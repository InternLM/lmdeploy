# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


@dataclass
class AdapterInfo:
    r: dict
    lora_A: nn.ModuleDict
    lora_B: nn.ModuleDict
    scaling: dict
    base_slice: slice
    in_features: int
    out_features: int

    @staticmethod
    def from_lora_linear(mod: nn.Module, base_slice: slice = None):
        if base_slice is None:
            base_slice = slice(None)
        return AdapterInfo(
            r=mod.r,
            lora_A=mod.lora_A,
            lora_B=mod.lora_B,
            scaling=mod.scaling,
            base_slice=base_slice,
            in_features=mod.in_features,
            out_features=mod.out_features,
        )


class SLoRAImpl(ABC):

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                base_output: torch.Tensor,
                target_name: str,
                layer_idx: int,
                is_tp: bool = True):
        raise NotImplementedError


class SLoRABuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(adapter_info: AdapterInfo,
              ctx_mgr: StepContextManager,
              colwise: bool = True):
        raise NotImplementedError
