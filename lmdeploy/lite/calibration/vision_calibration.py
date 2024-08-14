# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from lmdeploy.lite.utils import bimap_name_mod, collect_target_modules

from .calibration import CalibrationContext


class VisionCalibrationContext(CalibrationContext):

    inp_obs_group = 'vision_inputs'
    out_obs_group = 'vision_outputs'

    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 layer_type: Union[str, type],
                 norm_type: Union[str, type],
                 batch_size: int = 1,
                 device: str = 'cuda',
                 **kwargs) -> None:
        self.layer_type = layer_type
        self.norm_type = norm_type
        self.batch_size = batch_size
        # TODO models other than InternVL
        self.model = model.vl_model.vision_model
        self.vl_model = model

        self.tokenizer = tokenizer
        # Collect modules to observe
        self.name2layer = collect_target_modules(self.model, layer_type)
        self.name2norm = collect_target_modules(self.model, norm_type)

        self.name2fc = {}
        for l_name, layer in self.name2layer.items():
            name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
            self.name2fc.update(name2fc)
        maps = bimap_name_mod([self.name2layer, self.name2fc, self.name2norm])
        self.name2mod, self.mod2name = maps

        # Initialize observers
        self._init_input_observers(self.name2fc)
        self._init_output_observers(self.name2norm)
        self._init_output_observers(self.name2fc)

        self.device = device

    def calibrate(self, images):
        """Forward pass through the model in inference mode with given
        images."""
        with torch.inference_mode():
            _ = self.vl_model.forward(images)

    def __enter__(self):
        """Prepares the Calibration object for a 'with' statement by
        registering hooks and wrapping layer forward methods."""

        self._hooks = list()

        self._ori_forwards = {}
        for layer in self.name2layer.values():
            self._ori_forwards[layer] = layer.forward

        self._insert_input_observers()
        self._insert_output_observers()

    def export(self, out_dir):
        """Export the calibration statistics (inputs, outputs, keys and values)
        to specified directory.

        Args:
            out_dir (Union[str, Path]): The directory path where the stats
                will be saved.
        """
        inp_stats = self.collect_inputs_stats()
        torch.save(inp_stats, out_dir / 'vision_inputs_stats.pth')
        out_stats = self.collect_outputs_stats()
        torch.save(out_stats, out_dir / 'vision_outputs_stats.pth')
