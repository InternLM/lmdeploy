# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Union

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from lmdeploy.lite.quantization.activation import ActivationObserver
from lmdeploy.lite.utils import (bimap_name_mod, collect_target_modules,
                                 concat_decoder_layer_outputs,
                                 split_decoder_layer_inputs)


class CalibrationContext():
    """Calibration context manager for model quantization.

    Parameters:
      - model: The target model to be calibrated and quantized
      - tokenizer: The tokenizer used in the model training
      - layer_type: Layer type to be targeted for calibration
      - norm_type: Normalization type used for calibration
      - device: Device on which model is to be calibrated ('cpu' or 'cuda')
    """

    inp_obs_group = 'inputs'
    out_obs_group = 'outputs'

    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 layer_type: Union[str, type],
                 norm_type: Union[str, type],
                 batch_size: int = 1,
                 device: str = 'cuda',
                 **kwargs) -> None:
        """Initiate calibration context.

        Args:
            model (nn.Module): Model to be calibrated.
            tokenizer (PreTrainedTokenizer): Tokenizer of the given model.
            layer_type (Union[str, type]): Type of the layers to be observed.
            norm_type (Union[str, type]): Norm type used in the model.
            device (str, optional): Device where the model should run.
                Defaults to 'cuda'.
        """

        self.layer_type = layer_type
        self.norm_type = norm_type
        self.batch_size = batch_size
        self.model = model

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

    def _init_input_observers(self, name2mod):
        """Initialize input observers for given modules."""
        for name, mod in name2mod.items():
            obs = ActivationObserver(mod.weight.size(-1))
            obs.global_available(name, group=self.inp_obs_group)

    def _init_output_observers(self, name2mod):
        """Initialize output observers for given modules."""
        for name, mod in name2mod.items():
            obs = ActivationObserver(mod.weight.size(0))
            obs.global_available(name, group=self.out_obs_group)

    def _insert_input_observers(self):
        """Insert input observers into the target modules.

        This function registers a forward pre-hook on each target module to
        observe the inputs.
        """

        def _input_hook(mod: nn.Module, inp: torch.Tensor):
            m_name = self.mod2name[mod]
            obs = ActivationObserver.find(m_name, group=self.inp_obs_group)
            obs.observe(inp[0])

        group = ActivationObserver.find_group(self.inp_obs_group)
        for name in group.keys():
            mod = self.name2mod[name]
            hook_fn = mod.register_forward_pre_hook(_input_hook)
            self._hooks.append(hook_fn)

    def _insert_output_observers(self):
        """Insert output observers into the target modules.

        This function registers a forward hook on each target module to observe
        the outputs.
        """

        def _output_hook(mod: nn.Module, inp: torch.Tensor, out: torch.Tensor):
            m_name = self.mod2name[mod]
            obs = ActivationObserver.find(m_name, group=self.out_obs_group)
            obs.observe(out)

        group = ActivationObserver.find_group(self.out_obs_group)
        for name in group.keys():
            mod = self.name2mod[name]
            hook_fn = mod.register_forward_hook(_output_hook)
            self._hooks.append(hook_fn)

    def _wrap_decoder_layers(self):
        """Method to wrap the decoder layers' forward functions for observing
        their key/value cache during batched forward passes."""

        def _forward(mod, *args, **kwargs):

            mod.to(self.device)
            batch_args, batch_kwargs = split_decoder_layer_inputs(
                self.batch_size, *args, **kwargs)
            batch_outputs = []
            samples = len(batch_args)

            m_name = self.mod2name[mod]

            for i in range(len(batch_args)):
                batch_outputs.append(self._ori_forwards[mod](
                    *batch_args[i], **batch_kwargs[i]))

            outputs = concat_decoder_layer_outputs(batch_outputs)

            del batch_outputs, batch_args, batch_kwargs, args
            mod.to('cpu')
            torch.cuda.empty_cache()
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            print(f'{m_name}, samples: {samples}, '
                  f'max gpu memory: {max_memory:.2f} GB')
            return outputs

        for layer in self.name2layer.values():
            self._ori_forwards[layer] = layer.forward
            layer.forward = partial(_forward, layer)

    def collect_inputs_stats(self):
        """Collect statistics (min, max, absmax values) of the observed inputs.

        Returns a dictionary with these collected stats.
        """
        inputs_stats = {
            'max': {},
            'min': {},
            'mean': {},
            'absmax': {},
            'absmean': {}
        }
        obs_group = ActivationObserver.find_group(self.inp_obs_group)
        for name, obs in obs_group.items():
            inputs_stats['max'][name] = obs.max_val
            inputs_stats['min'][name] = obs.min_val
            inputs_stats['mean'][name] = obs.mean_val
            inputs_stats['absmax'][name] = obs.absmax_val
            inputs_stats['absmean'][name] = obs.absmean_val
        return inputs_stats

    def collect_outputs_stats(self):
        """Collect statistics (min, max, absmax values) of the observed
        outputs.

        Returns a dictionary with these collected stats.
        """
        outputs_stats = {
            'max': {},
            'min': {},
            'mean': {},
            'absmax': {},
            'absmean': {}
        }
        obs_group = ActivationObserver.find_group(self.out_obs_group)
        for name, obs in obs_group.items():
            outputs_stats['max'][name] = obs.max_val
            outputs_stats['min'][name] = obs.min_val
            outputs_stats['mean'][name] = obs.mean_val
            outputs_stats['absmax'][name] = obs.absmax_val
            outputs_stats['absmean'][name] = obs.absmean_val
        return outputs_stats

    def export(self, out_dir):
        """Export the calibration statistics (inputs, outputs, keys and values)
        to specified directory.

        Args:
            out_dir (Union[str, Path]): The directory path where the stats
                will be saved.
        """

        inp_stats = self.collect_inputs_stats()
        torch.save(inp_stats, out_dir / 'inputs_stats.pth')

        out_stats = self.collect_outputs_stats()
        torch.save(out_stats, out_dir / 'outputs_stats.pth')

    def calibrate(self, data):
        """Forward pass through the model in inference mode with given data."""
        if type(self.model).__name__ in ('QWenLMHeadModel',
                                         'ChatGLMForConditionalGeneration'):
            model = self.model.transformer
        else:
            model = self.model.model
        with torch.inference_mode():
            _ = model(data.to(self.device))

    def __enter__(self):
        """Prepares the Calibration object for a 'with' statement by
        registering hooks and wrapping layer forward methods."""

        self._hooks = list()

        self._ori_forwards = {}
        for layer in self.name2layer.values():
            self._ori_forwards[layer] = layer.forward

        self._insert_input_observers()
        self._insert_output_observers()
        self._wrap_decoder_layers()

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up after a 'with' statement by removing registered hooks,
        restoring original forward methods, and if no exception occurred,
        collecting all gathered statistics and saving them."""
        for h in self._hooks:
            h.remove()

        for layer in self.name2layer.values():
            layer.forward = self._ori_forwards[layer]
