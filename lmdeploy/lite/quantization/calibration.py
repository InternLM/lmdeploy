# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Union

import torch
import transformers
from mmengine import digit_version
from torch import nn
from transformers import PreTrainedTokenizer

from lmdeploy.lite.quantization.activation import (ActivationObserver,
                                                   KVCacheObserver)
from lmdeploy.lite.quantization.awq import FC_FCS_MAP, NORM_FCS_MAP
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
    key_obs_group = 'keys'
    value_obs_group = 'values'

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

        num_kv_heads, num_attn_heads = self._guess_num_heads(model)
        self.num_kv_heads = num_kv_heads
        self.head_dim = model.config.hidden_size // num_attn_heads
        self.model = model

        self.tokenizer = tokenizer

        # Collect modules to observe
        self.name2layer = collect_target_modules(self.model, layer_type)
        self.name2fc = {}
        for l_name, layer in self.name2layer.items():
            name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
            self.name2fc.update(name2fc)
        self.name2norm = collect_target_modules(self.model, norm_type)

        maps = bimap_name_mod([self.name2layer, self.name2fc, self.name2norm])
        self.name2mod, self.mod2name = maps

        # Initialize observers
        self._init_input_observers(self.name2fc)
        self._init_output_observers(self.name2norm)
        self._init_output_observers(self.name2fc)
        self._init_kv_observers(self.name2layer)

        self.device = device

    def _guess_num_heads(self, model):

        if hasattr(model.config, 'num_key_value_heads'):
            num_kv_heads = model.config.num_key_value_heads
        else:
            num_kv_heads = model.config.num_attention_heads

        num_attn_heads = model.config.num_attention_heads

        return num_kv_heads, num_attn_heads

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
            k_obs = KVCacheObserver(self.num_kv_heads, self.head_dim)
            v_obs = KVCacheObserver(self.num_kv_heads, self.head_dim)
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
                self.batch_size, *args, **kwargs)
            batch_outputs = []
            samples = len(batch_args)

            m_name = self.mod2name[mod]
            k_obs = KVCacheObserver.find(m_name, group=self.key_obs_group)
            v_obs = KVCacheObserver.find(m_name, group=self.value_obs_group)

            for i in range(len(batch_args)):

                if k_obs and v_obs:
                    batch_kwargs[i]['use_cache'] = True
                    version = digit_version(transformers.__version__)
                    use_new_cache = type(mod).__name__ in ('LlamaDecoderLayer',
                                                           'Qwen2DecoderLayer')
                    if type(mod).__name__ == 'InternLM2DecoderLayer':
                        use_new_cache = hasattr(mod.attention, 'layer_idx')
                    if version > digit_version('4.36.0') and use_new_cache:
                        from transformers.cache_utils import DynamicCache
                        batch_kwargs[i]['past_key_value'] = DynamicCache()

                        if hasattr(mod, 'self_attn'):
                            self_attn = mod.self_attn
                        elif hasattr(mod, 'attention'):
                            self_attn = mod.attention
                        else:
                            raise RuntimeError('Attention layer not found')

                        ori_idx = self_attn.layer_idx
                        self_attn.layer_idx = 0

                        out = self._ori_forwards[mod](*batch_args[i],
                                                      **batch_kwargs[i])
                        self_attn.layer_idx = ori_idx

                        out = list(out)
                        cache = out.pop(-1)

                        key = cache.key_cache.pop(-1)
                        value = cache.value_cache.pop(-1)

                        k_obs.observe(key)
                        v_obs.observe(value)

                    else:
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

        key_stats, value_stats = self.collect_kv_stats()
        torch.save(key_stats, out_dir / 'key_stats.pth')
        torch.save(value_stats, out_dir / 'value_stats.pth')

    def calibrate(self, data):
        """Forward pass through the model in inference mode with given data."""

        if type(self.model).__name__ == 'QWenLMHeadModel':
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


@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, w_group_size, input_feat,
                     mod_name):
    if 'use_cache' in module_kwargs:
        module_kwargs.pop('use_cache')

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = x.abs().view(-1, x.shape[-1]).mean(0)

        best_error = float('inf')
        best_ratio = -1
        n_grid = 20
        history = []

        concat_w = torch.cat([_m.weight for _m in linears2scale], dim=0)
        from .awq import get_weight_scale, pseudo_quantize_tensor
        w_mean = get_weight_scale(concat_w, w_group_size)

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(0, n_grid):
            ratio = ratio * 1 / n_grid
            scales = (x_max.pow(ratio) /
                      w_mean.pow(1 - ratio)).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = pseudo_quantize_tensor(
                    fc.weight.data, w_bit, w_group_size) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # float prevents overflow
            loss = (org_out - out).float().pow(2).mean().item()
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        return best_ratio

    def _auto_get_scale(layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of
        #  this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        best_ratio = _search_module_scale(module2inspect, layers, inp.value,
                                          kwargs)
        inp.save_ratio(best_ratio)

    for i, (prev_name, layer_names) in enumerate(
            NORM_FCS_MAP[module._get_name()].items()):
        # attention input
        _auto_get_scale(
            layers=[module.get_submodule(name) for name in layer_names],
            inp=input_feat[f'{mod_name}.{layer_names[0]}'],
            module2inspect=module.get_submodule(layer_names[0].split('.')[0]),
            kwargs=module_kwargs
            if i == 0 else {},  # only attention input need
        )
    for prev_name, layer_names in FC_FCS_MAP[module._get_name()].items():
        # attention input
        _auto_get_scale(
            layers=[module.get_submodule(name) for name in layer_names],
            inp=input_feat[f'{mod_name}.{layer_names[0]}'],
        )


class CalibrationContextV2(CalibrationContext):

    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 layer_type: Union[str, type],
                 norm_type: Union[str, type],
                 batch_size: int = 1,
                 device: str = 'cuda',
                 search_scale: bool = True,
                 w_bits: int = 4,
                 w_group_size: int = 128,
                 **kwargs) -> None:
        super().__init__(model, tokenizer, layer_type, norm_type, batch_size,
                         device)
        self.w_bits = w_bits
        self.w_group_size = w_group_size
        self.search_scale = search_scale

    def _insert_input_observers(self):
        """Insert input observers into the target modules.

        This function registers a forward pre-hook on each target module to
        observe the inputs.
        """

        def _input_hook(mod: nn.Module, inp: torch.Tensor):
            m_name = self.mod2name[mod]
            obs = ActivationObserver.find(m_name, group=self.inp_obs_group)
            obs.observe(inp[0], self.search_scale)

        group = ActivationObserver.find_group(self.inp_obs_group)
        for name in group.keys():
            mod = self.name2mod[name]
            hook_fn = mod.register_forward_pre_hook(_input_hook)
            self._hooks.append(hook_fn)

    def export(self, out_dir):
        """Export the calibration statistics (inputs, outputs, keys and values)
        to specified directory.

        Args:
            out_dir (Union[str, Path]): The directory path where the stats
                will be saved.
        """
        inputs_stats = {
            'max': {},
            'min': {},
            'mean': {},
            'absmax': {},
            'absmean': {},
            'ratios': {},
        }
        obs_group = ActivationObserver.find_group(self.inp_obs_group)
        for name, obs in obs_group.items():
            inputs_stats['max'][name] = obs.max_val
            inputs_stats['min'][name] = obs.min_val
            inputs_stats['mean'][name] = obs.mean_val
            inputs_stats['absmax'][name] = obs.absmax_val
            inputs_stats['absmean'][name] = obs.absmean_val
            inputs_stats['ratios'][name] = obs.ratio
        torch.save(inputs_stats, out_dir / 'inputs_stats.pth')

    def _wrap_decoder_layers_for_search(self):
        """Method to wrap the decoder layers' forward functions for observing
        their key/value cache during batched forward passes."""

        @torch.no_grad()
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
                obs_group = ActivationObserver.find_group(self.inp_obs_group)
                mod_name = self.mod2name[mod]
                ActivationObserver.disable()
                auto_scale_block(mod, batch_kwargs[i], self.w_bits,
                                 self.w_group_size, obs_group, mod_name)
                ActivationObserver.enable()
            for key, item in obs_group.items():
                if key.startswith(f'{mod_name}.'):
                    item.value.cpu()
                    del item.value

            outputs = concat_decoder_layer_outputs(batch_outputs)

            del batch_outputs, batch_args, batch_kwargs, args
            mod.cpu()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            max_memory = torch.cuda.max_memory_allocated() / (1 << 30)
            print(f'{m_name}, samples: {samples}, '
                  f'max gpu memory: {max_memory:.2f} GB')
            return outputs

        for layer in self.name2layer.values():
            self._ori_forwards[layer] = layer.forward
            layer.forward = partial(_forward, layer)
            layer.cpu()

    def __enter__(self):
        """Prepares the Calibration object for a 'with' statement by
        registering hooks and wrapping layer forward methods."""

        self._hooks = list()

        self._insert_input_observers()
        self._ori_forwards = {}
        for layer in self.name2layer.values():
            self._ori_forwards[layer] = layer.forward

        if self.search_scale:
            self._wrap_decoder_layers_for_search()
