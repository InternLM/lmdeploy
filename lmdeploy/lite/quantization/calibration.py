# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch import nn

from lmdeploy.lite.quantization.activation import (ActivationObserver,
                                                   KVCacheObserver)
from lmdeploy.lite.quantization.smooth import (FC_FCS_MAP, NORM_FCS_MAP,
                                               smooth_fc_fcs, smooth_ln_fcs)
from lmdeploy.lite.quantization.weight import WeightObserver, WeightQuantizer
from lmdeploy.lite.utils import (bimap_name_mod, collect_target_modules,
                                 concat_decoder_layer_outputs,
                                 split_decoder_layer_inputs)
from lmdeploy.pytorch.modules import WeightOnlyQLinear


class Calibration():
    """Calibration context manager for model quantization.

    Parameters:
      - model: The target model to be calibrated and quantized
      - layer_type: Layer type to be targeted for calibration
      - norm_type: Normalization type used for calibration
      - smooth: Flag to indicate if smoothing to be applied or not
      - w_qconfig: Config file path for weight quantization
      - work_dir: Directory path to save the calibration and quantization
                        results
      - device: Device on which model is to be calibrated ('cpu' or 'cuda')

    Note: More detailed information should be added here to explain what this
            class does exactly and how it works.
    """

    inp_obs_group = 'inputs'
    out_obs_group = 'outputs'
    key_obs_group = 'keys'
    value_obs_group = 'values'

    w_obs_group = 'weights'
    smooth_w_obs_group = 'smooth'

    def __init__(self,
                 model: nn.Module,
                 layer_type: Union[str, type],
                 norm_type: Union[str, type],
                 smooth: bool = False,
                 w_bits: int = 4,
                 w_sym: bool = False,
                 w_granularity: str = 'per_group',
                 w_group_size: int = 128,
                 work_dir: str = './work_dir',
                 device: str = 'cuda') -> None:

        self.smooth = smooth
        self.layer_type = layer_type
        self.norm_type = norm_type
        if self.smooth:
            qconfig = {}
            qconfig['bits'] = w_bits
            qconfig['symmetry'] = w_sym
            qconfig['granularity'] = w_granularity
            qconfig['group_size'] = w_group_size

            self.w_qconfig = qconfig
            self._check_smooth_supported()

        self.num_head = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.num_head
        self.model = model

        self.name2layer = collect_target_modules(self.model, layer_type)
        self.name2fc = {}
        for l_name, layer in self.name2layer.items():
            name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
            self.name2fc.update(name2fc)
        self.name2norm = collect_target_modules(self.model, norm_type)

        maps = bimap_name_mod([self.name2layer, self.name2fc, self.name2norm])
        self.name2mod, self.mod2name = maps

        self._init_input_observers(self.name2fc)
        self._init_output_observers(self.name2norm)
        self._init_output_observers(self.name2fc)
        self._init_kv_observers(self.name2layer)

        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _check_smooth_supported(self):
        """Check if the smooth function is supported by inspecting layer
        type."""
        norm_fcs_found = False
        fc_fcs_found = False
        if isinstance(self.layer_type, str):
            if self.layer_type in NORM_FCS_MAP:
                norm_fcs_found = True
            if self.layer_type in FC_FCS_MAP:
                fc_fcs_found = True

        elif isinstance(self.layer_type, type):
            if self.layer_type.__name__ in NORM_FCS_MAP:
                norm_fcs_found = True
            if self.layer_type.__name__ in FC_FCS_MAP:
                fc_fcs_found = True

        else:
            raise NotImplementedError

        if not norm_fcs_found:
            raise NotImplementedError

        if not fc_fcs_found:
            raise NotImplementedError

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

    def _init_kv_observers(self, name2mod):
        """Initialize KV observers for given modules."""
        for name in name2mod.keys():
            k_obs = KVCacheObserver(self.num_head, self.head_dim)
            v_obs = KVCacheObserver(self.num_head, self.head_dim)
            k_obs.global_available(name, group=self.key_obs_group)
            v_obs.global_available(name, group=self.value_obs_group)

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
                *args, **kwargs)
            batch_outputs = []
            samples = len(batch_args)

            m_name = self.mod2name[mod]
            k_obs = KVCacheObserver.find(m_name, group=self.key_obs_group)
            v_obs = KVCacheObserver.find(m_name, group=self.value_obs_group)

            for i in range(len(batch_args)):

                if k_obs and v_obs:
                    batch_kwargs[i]['use_cache'] = True
                    out = self._ori_forwards[mod](*batch_args[i],
                                                  **batch_kwargs[i])
                    out = list(out)
                    key, value = out.pop(-1)
                    k_obs.observe(key)
                    v_obs.observe(value)

                    del key, value
                    torch.cuda.empty_cache()
                    batch_outputs.append(tuple(out))
                else:
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

    def step(self, data):
        """Forward pass through the model in inference mode with given data."""
        with torch.inference_mode():
            _ = self.model.model(data.to(self.device))

    def collect_inputs_stats(self):
        """Collect statistics (min, max, absmax values) of the observed inputs.

        Returns a dictionary with these collected stats.
        """
        inputs_stats = {'max': {}, 'min': {}, 'absmax': {}}
        obs_group = ActivationObserver.find_group(self.inp_obs_group)
        for name, obs in obs_group.items():
            inputs_stats['max'][name] = obs.max_val
            inputs_stats['min'][name] = obs.min_val
            inputs_stats['absmax'][name] = obs.absmax_val
        return inputs_stats

    def collect_outputs_stats(self):
        """Collect statistics (min, max, absmax values) of the observed
        outputs.

        Returns a dictionary with these collected stats.
        """
        outputs_stats = {'max': {}, 'min': {}, 'absmax': {}}
        obs_group = ActivationObserver.find_group(self.out_obs_group)
        for name, obs in obs_group.items():
            outputs_stats['max'][name] = obs.max_val
            outputs_stats['min'][name] = obs.min_val
            outputs_stats['absmax'][name] = obs.absmax_val
        return outputs_stats

    def collect_kv_stats(self):
        """Collect statistics (min, max, absmax values) of the observed keys
        and values.

        Returns a tuple of two dictionaries with these collected stats.
        """
        key_stats = {'max': {}, 'min': {}, 'absmax': {}}
        obs_group = KVCacheObserver.find_group(self.key_obs_group)
        for name, obs in obs_group.items():
            key_stats['max'][name] = obs.max_val
            key_stats['min'][name] = obs.min_val
            key_stats['absmax'][name] = obs.absmax_val

        value_stats = {'max': {}, 'min': {}, 'absmax': {}}
        obs_group = KVCacheObserver.find_group(self.value_obs_group)
        for name, obs in obs_group.items():
            value_stats['max'][name] = obs.max_val
            value_stats['min'][name] = obs.min_val
            value_stats['absmax'][name] = obs.absmax_val
        return key_stats, value_stats

    def quant_weights(self):
        """Quantize the weights of the target model's linear layers."""
        for name, fc in self.name2fc.items():
            fc.to(self.device)
            quantizer = WeightQuantizer(**self.w_qconfig)
            q_linear = WeightOnlyQLinear.from_linear(fc, quantizer)

            parent_name, _, child_name = name.rpartition('.')
            parent = self.model.get_submodule(parent_name)
            fc.to('cpu')
            setattr(parent, child_name, q_linear)

            print(f'{name} weight packed.')

    def smooth_weights(self, a_scales):
        """Apply weight smoothing based on input scales."""
        if isinstance(self.layer_type, str):
            norm_fcs_map = NORM_FCS_MAP[self.layer_type]
            fc_fcs_map = FC_FCS_MAP[self.layer_type]
        else:
            layer_str = self.layer_type.__name__
            norm_fcs_map = NORM_FCS_MAP[layer_str]
            fc_fcs_map = FC_FCS_MAP[layer_str]

        for name, fc in self.name2fc.items():
            obs = WeightObserver(fc.in_features, 0)
            obs.observe(fc.weight)
            obs.global_available(name, group=self.smooth_w_obs_group)

        for l_name, layer in self.name2layer.items():
            layer.to(self.device)
            for ln_name, fc_names in norm_fcs_map.items():
                m_names = [f'{l_name}.{n}' for n in fc_names]

                observers = [
                    WeightObserver.find(n, group=self.smooth_w_obs_group)
                    for n in m_names
                ]
                WeightObserver.merge_stats(observers)

                ln = self.name2mod[f'{l_name}.{ln_name}']
                fcs = [self.name2mod[f'{l_name}.{n}'] for n in fc_names]
                smooth_ln_fcs(ln, fcs, a_scales[m_names[0]])

            for f_name, fc_names in fc_fcs_map.items():
                m_names = [f'{l_name}.{n}' for n in fc_names]

                observers = [
                    WeightObserver.find(n, group=self.smooth_w_obs_group)
                    for n in m_names
                ]
                WeightObserver.merge_stats(observers)

                fc = self.name2mod[f'{l_name}.{f_name}']
                fcs = [self.name2mod[f'{l_name}.{n}'] for n in fc_names]
                smooth_fc_fcs(fc, fcs, a_scales[m_names[0]])

            layer.to('cpu')
            print(f'{l_name} smooth weight done.')

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

        if exc_type is None and exc_value is None and traceback is None:

            inp_stats = self.collect_inputs_stats()
            torch.save(inp_stats, self.work_dir / 'inputs_stats.pth')

            out_stats = self.collect_outputs_stats()
            torch.save(out_stats, self.work_dir / 'outputs_stats.pth')

            key_stats, value_stats = self.collect_kv_stats()
            torch.save(key_stats, self.work_dir / 'key_stats.pth')
            torch.save(value_stats, self.work_dir / 'value_stats.pth')

            if self.smooth:
                self.smooth_weights(inp_stats['absmax'])
                self.quant_weights()
                self.model.save_pretrained(self.work_dir)
