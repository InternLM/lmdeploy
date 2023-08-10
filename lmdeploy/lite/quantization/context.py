# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from lmdeploy.lite.quantization.activation import (ActivationObserver,
                                                   KVCacheObserver)
from lmdeploy.lite.quantization.smooth import (FC_FCS_MAP, NORM_FCS_MAP,
                                               smooth_fc_fcs, smooth_ln_fcs)
from lmdeploy.lite.quantization.weight import WeightQuantizer
from lmdeploy.lite.utils import (bimap_name_mod, collect_target_modules,
                                 concat_decoder_layer_outputs,
                                 split_decoder_layer_inputs)
from lmdeploy.pytorch.modules import WeightOnlyQLinear


class QuantizeContext():
    """Calibration context manager for model quantization.

    Parameters:
      - model: The target model to be calibrated and quantized
      - layer_type: Layer type to be targeted for calibration
      - norm_type: Normalization type used for calibration
      - smooth: Flag to indicate if smoothing to be applied or not
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

    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 layer_type: Union[str, type],
                 norm_type: Union[str, type],
                 device: str = 'cuda') -> None:

        self.layer_type = layer_type
        self.norm_type = norm_type

        self.num_head = self._guess_num_heads(model)
        self.head_dim = model.config.hidden_size // self.num_head
        self.model = model
        self.tokenizer = tokenizer

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

    def _guess_num_heads(self, model):
        if hasattr(model.config, 'num_attention_heads'):
            return model.config.num_attention_heads
        elif hasattr(model.config, 'num_key_value_heads'):
            return model.config.num_key_value_heads
        else:
            raise KeyError

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

    def quant_weights(self, bits, symmetry, granularity, group_size=-1):
        """Quantize the weights of the target model's linear layers."""
        for name, fc in self.name2fc.items():
            fc.to(self.device)
            quantizer = WeightQuantizer(bits, symmetry, granularity,
                                        group_size)
            q_linear = WeightOnlyQLinear.from_linear(fc, quantizer)

            parent_name, _, child_name = name.rpartition('.')
            parent = self.model.get_submodule(parent_name)
            fc.to('cpu')
            setattr(parent, child_name, q_linear)

            print(f'{name} weight packed.')

    def smooth_weights(self, a_scales, group_size=-1):
        """Apply weight smoothing based on input scales."""
        if isinstance(self.layer_type, str):
            norm_fcs_map = NORM_FCS_MAP[self.layer_type]
            fc_fcs_map = FC_FCS_MAP[self.layer_type]
        else:
            layer_str = self.layer_type.__name__
            norm_fcs_map = NORM_FCS_MAP[layer_str]
            fc_fcs_map = FC_FCS_MAP[layer_str]

        for l_name, layer in self.name2layer.items():
            layer.to(self.device)
            for ln_name, fc_names in norm_fcs_map.items():
                m_names = [f'{l_name}.{n}' for n in fc_names]

                ln = self.name2mod[f'{l_name}.{ln_name}']
                fcs = [self.name2mod[f'{l_name}.{n}'] for n in fc_names]
                smooth_ln_fcs(ln, fcs, a_scales[m_names[0]], group_size)

            for f_name, fc_names in fc_fcs_map.items():
                m_names = [f'{l_name}.{n}' for n in fc_names]

                fc = self.name2mod[f'{l_name}.{f_name}']
                fcs = [self.name2mod[f'{l_name}.{n}'] for n in fc_names]
                smooth_fc_fcs(fc, fcs, a_scales[m_names[0]], group_size)

            layer.to('cpu')
            print(f'{l_name} smooth weight done.')

    def export_stats(self, out_dir):
        inp_stats = self.collect_inputs_stats()
        torch.save(inp_stats, out_dir / 'inputs_stats.pth')

        out_stats = self.collect_outputs_stats()
        torch.save(out_stats, out_dir / 'outputs_stats.pth')

        key_stats, value_stats = self.collect_kv_stats()
        torch.save(key_stats, out_dir / 'key_stats.pth')
        torch.save(value_stats, out_dir / 'value_stats.pth')

    def _export_tm_sym_kv_qparams(self,
                                  key_stats,
                                  value_stats,
                                  bits,
                                  out_dir,
                                  tp=1):
        keys_absmax = key_stats['absmax']
        values_absmax = value_stats['absmax']
        for layer_idx, name in enumerate(keys_absmax.keys()):
            k_absmax = keys_absmax[name]
            v_absmax = values_absmax[name]

            heads, dims = k_absmax.shape
            assert heads % tp == 0

            mp_k_absmax = torch.chunk(k_absmax, tp)
            mp_v_absmax = torch.chunk(v_absmax, tp)
            for i in range(tp):
                # quant: q = f / scale
                # dequant: f = q * scale
                k_s = max(mp_k_absmax[i]) / (2**(bits - 1) - 1)
                v_s = max(mp_v_absmax[i]) / (2**(bits - 1) - 1)

                kv_qparams = np.array([k_s, v_s], dtype=np.float32)
                save_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'  # noqa: E501
                kv_qparams.tofile(save_path)
                print(f'Layer {layer_idx} MP {i} KV scales done.')

    def _export_tm_asym_kv_qparams(self,
                                   key_stats,
                                   value_stats,
                                   bits,
                                   out_dir,
                                   tp=1):
        keys_min = key_stats['min']
        values_min = value_stats['min']

        keys_max = key_stats['max']
        values_max = value_stats['max']
        for layer_idx, name in enumerate(keys_min.keys()):
            k_max = keys_max[name]
            v_max = values_max[name]

            k_min = keys_min[name]
            v_min = values_min[name]

            heads, dims = keys_min.shape
            assert heads % tp == 0

            mp_k_min = torch.chunk(k_min, tp)
            mp_v_min = torch.chunk(v_min, tp)

            mp_k_max = torch.chunk(k_max, tp)
            mp_v_max = torch.chunk(v_max, tp)
            for i in range(tp):
                # quant: q = (f - zp) / scale
                # dequant: f = q * scale + zp
                k_min = min(mp_k_min[i])
                v_min = min(mp_v_min[i])

                k_max = max(mp_k_max[i])
                v_max = max(mp_v_max[i])

                k_scale = (k_max - k_min) / (2**bits - 1)
                v_scale = (v_max - v_min) / (2**bits - 1)

                kv_qparams = np.array([k_scale, k_min, v_scale, v_min],
                                      dtype=np.float32)
                save_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'  # noqa: E501
                kv_qparams.tofile(save_path)
                print(f'Layer {layer_idx} MP {i} KV scales&zeros done.')

    def export_turbomind_kv_qparams(self, bits, symmetry, out_dir, tp=1):
        out_dir = Path(out_dir)

        key_stats, value_stats = self.collect_kv_stats()
        if symmetry:

            self._export_tm_sym_kv_qparams(key_stats, value_stats, bits,
                                           out_dir, tp)
        else:
            self._export_tm_asym_kv_qparams(key_stats, value_stats, bits,
                                            out_dir, tp)

    def calibrate(self, data):
        """Forward pass through the model in inference mode with given data."""
        with torch.inference_mode():
            _ = self.model.model(data.to(self.device))

    def auto_awq(self, bits, sym, group_size, out_dir):
        self._check_smooth_supported()

        inp_stats = self.collect_inputs_stats()

        scales = inp_stats['absmean']
        self.smooth_weights(scales, group_size)
        self.quant_weights(bits, sym, 'per_group', group_size)

        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

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
