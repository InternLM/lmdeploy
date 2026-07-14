# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from lmdeploy.lite.quantization.activation import ActivationObserver
from lmdeploy.lite.quantization.awq import FC_FCS_MAP, NORM_FCS_MAP
from lmdeploy.lite.utils import (
    bimap_name_mod,
    collect_target_modules,
    concat_decoder_layer_outputs,
    split_decoder_layer_inputs,
)

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class CalibrationContext:
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
                 layer_type: Union[str, Type[nn.Module]],
                 norm_type: Union[str, Type[nn.Module]],
                 batch_size: int = 1,
                 device: str = 'cuda',
                 **kwargs: Any) -> None:
        """Initiate calibration context.

        Args:
            model: Model to be calibrated.
            tokenizer: Tokenizer of the given model.
            layer_type: Type of the layers to be observed.
            norm_type: Norm type used in the model.
            batch_size: The batch size for running the calib samples.
                Low GPU mem requires small batch_size. Large batch_size
                reduces the calibration time while costs more VRAM.
            device: Device where the model should run. Defaults to 'cuda'.
        """
        try:
            if not isinstance(model, nn.Module):
                raise TypeError(f'model must be nn.Module, got {type(model).__name__}')
            if not isinstance(tokenizer, PreTrainedTokenizer):
                raise TypeError(f'tokenizer must be PreTrainedTokenizer, got {type(tokenizer).__name__}')
            if batch_size <= 0:
                raise ValueError(f'batch_size must be positive, got {batch_size}')
            if device not in ('cpu', 'cuda'):
                raise ValueError(f'device must be "cpu" or "cuda", got {device}')

            self.layer_type = layer_type
            self.norm_type = norm_type
            self.batch_size = batch_size

            num_kv_heads, num_attn_heads, text_config = self._guess_num_heads(model)
            self.num_kv_heads = num_kv_heads
            self.head_dim = text_config.hidden_size // num_attn_heads
            self.model = model

            self.tokenizer = tokenizer

            self.name2layer = collect_target_modules(self.model, layer_type)
            self.name2fc: Dict[str, nn.Module] = {}
            for l_name, layer in self.name2layer.items():
                name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
                self.name2fc.update(name2fc)
            self.name2norm = collect_target_modules(self.model, norm_type)

            maps = bimap_name_mod([self.name2layer, self.name2fc, self.name2norm])
            self.name2mod, self.mod2name = maps

            self._init_input_observers(self.name2fc)
            self._init_output_observers(self.name2norm)
            self._init_output_observers(self.name2fc)

            self.device = device
        except Exception as e:
            logger.error(f'Failed to initialize CalibrationContext: {e}')
            raise

    def _guess_num_heads(self, model: nn.Module) -> Tuple[int, int, Any]:
        if hasattr(model.config, 'text_config'):
            text_config = model.config.text_config
        elif hasattr(model.config, 'llm_config'):
            text_config = model.config.llm_config
        else:
            text_config = model.config

        if hasattr(text_config, 'num_key_value_heads'):
            num_kv_heads = text_config.num_key_value_heads
        else:
            num_kv_heads = text_config.num_attention_heads

        if not hasattr(text_config, 'num_attention_heads'):
            raise ValueError('Model config must have num_attention_heads')

        num_attn_heads = text_config.num_attention_heads

        return num_kv_heads, num_attn_heads, text_config

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

        This function registers a forward pre-hook on each target module to observe the inputs.
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

        This function registers a forward hook on each target module to observe the outputs.
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
            batch_args, batch_kwargs = split_decoder_layer_inputs(self.batch_size, *args, **kwargs)
            batch_outputs = []
            samples = len(batch_args)

            m_name = self.mod2name[mod]

            for i in range(len(batch_args)):
                batch_outputs.append(self._ori_forwards[mod](*batch_args[i], **batch_kwargs[i]))

            outputs = concat_decoder_layer_outputs(batch_outputs)

            del batch_outputs, batch_args, batch_kwargs, args
            mod.to('cpu')
            torch.cuda.empty_cache()
            max_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024 / 1024
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
        inputs_stats = {'max': {}, 'min': {}, 'mean': {}, 'absmax': {}, 'absmean': {}}
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
        outputs_stats = {'max': {}, 'min': {}, 'mean': {}, 'absmax': {}, 'absmean': {}}
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
            out_dir (str | Path): The directory path where the stats
                will be saved.
        """

        inp_stats = self.collect_inputs_stats()
        torch.save(inp_stats, out_dir / 'inputs_stats.pth')
        torch.cuda.empty_cache()

        out_stats = self.collect_outputs_stats()
        torch.save(out_stats, out_dir / 'outputs_stats.pth')
        torch.cuda.empty_cache()

    def calibrate(self, data):
        """Forward pass through the model in inference mode with given data."""

        if type(self.model).__name__ in ('QWenLMHeadModel', 'ChatGLMForConditionalGeneration'):
            model = self.model.transformer
        else:
            model = self.model.model
        with torch.inference_mode():
            _ = model(data.to(self.device))
        torch.cuda.empty_cache()

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
def auto_scale_block(module: nn.Module,
                     module_kwargs: Dict[str, Any],
                     w_bit: int,
                     w_group_size: int,
                     input_feat: Dict[str, Any],
                     mod_name: str) -> None:
    try:
        module_kwargs = module_kwargs.copy()
        module_kwargs.pop('use_cache', None)

        def _search_module_scale(block: nn.Module,
                                linears2scale: List[nn.Linear],
                                x: torch.Tensor,
                                kwargs: Optional[Dict[str, Any]] = None) -> float:
            kwargs = kwargs or {}
            device = next(block.parameters(), None)
            if device is None:
                raise RuntimeError('Block has no parameters')
            x = x.to(device.device)
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

            x_max = x.abs().view(-1, x.shape[-1]).mean(0)

            best_error = float('inf')
            best_ratio = -1.0
            n_grid = 20

            concat_w = torch.cat([_m.weight for _m in linears2scale], dim=0)
            from .awq import get_weight_scale, pseudo_quantize_tensor
            w_mean = get_weight_scale(concat_w, w_group_size)

            org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
            for ratio_idx in range(n_grid):
                ratio = ratio_idx / n_grid
                w_mean_pow = w_mean.pow(1 - ratio)
                if w_mean_pow.min().item() == 0:
                    logger.warning('w_mean.pow(1 - ratio).min is zero, '
                                   'clamping w_mean.pow(1 - ratio) to 1e-4')
                    w_mean_pow = w_mean_pow.clamp(min=1e-4)
                scales = (x_max.pow(ratio) / w_mean_pow).clamp(min=1e-4).view(-1)

                scales = scales / (scales.max() * scales.min()).sqrt()
                for fc in linears2scale:
                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, w_group_size) / (scales.view(1, -1))
                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = (org_out - out).float().pow(2).mean().item()
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                block.load_state_dict(org_sd)

            if best_ratio < 0:
                raise RuntimeError(f'Failed to find best scale ratio for {mod_name}')
            return best_ratio

        def _auto_get_scale(layers: List[nn.Module],
                            inp: Any,
                            module2inspect: Optional[nn.Module] = None,
                            kwargs: Optional[Dict[str, Any]] = None) -> None:
            kwargs = kwargs or {}
            if module2inspect is None:
                if len(layers) != 1:
                    raise ValueError('module2inspect must be provided when len(layers) != 1')
                module2inspect = layers[0]

            if module2inspect._get_name() == 'InternLM2MLP':
                from inspect import signature
                if 'im_mask' in signature(module2inspect.forward).parameters:
                    kwargs['im_mask'] = None

            best_ratio = _search_module_scale(module2inspect, layers, inp.value, kwargs)
            inp.save_ratio(best_ratio)

        module_name = module._get_name()
        if module_name not in NORM_FCS_MAP:
            logger.warning(f'Module {module_name} not found in NORM_FCS_MAP, skipping auto-scale')
            return

        for i, (prev_name, layer_names) in enumerate(NORM_FCS_MAP[module_name].items()):
            _auto_get_scale(
                layers=[module.get_submodule(name) for name in layer_names],
                inp=input_feat[f'{mod_name}.{layer_names[0]}'],
                module2inspect=module.get_submodule(layer_names[0].split('.')[0]),
                kwargs=module_kwargs if i == 0 else {},
            )

        if module_name in FC_FCS_MAP:
            for prev_name, layer_names in FC_FCS_MAP[module_name].items():
                _auto_get_scale(
                    layers=[module.get_submodule(name) for name in layer_names],
                    inp=input_feat[f'{mod_name}.{layer_names[0]}'],
                )
    except Exception as e:
        logger.error(f'Failed to auto-scale block {mod_name}: {e}')
        raise


class CalibrationContextV2(CalibrationContext):

    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 layer_type: str | type,
                 norm_type: str | type,
                 batch_size: int = 1,
                 device: str = 'cuda',
                 search_scale: bool = True,
                 w_bits: int = 4,
                 w_group_size: int = 128,
                 **kwargs) -> None:
        super().__init__(model, tokenizer, layer_type, norm_type, batch_size, device)
        self.w_bits = w_bits
        self.w_group_size = w_group_size
        self.search_scale = search_scale

    def _insert_input_observers(self):
        """Insert input observers into the target modules.

        This function registers a forward pre-hook on each target module to observe the inputs.
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
            out_dir (str | Path): The directory path where the stats
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
        torch.cuda.empty_cache()

    def _wrap_decoder_layers_for_search(self):
        """Method to wrap the decoder layers' forward functions for observing
        their key/value cache during batched forward passes."""

        @torch.no_grad()
        def _forward(mod, *args, **kwargs):

            mod.to(self.device)
            batch_args, batch_kwargs = split_decoder_layer_inputs(self.batch_size, *args, **kwargs)
            batch_outputs = []
            samples = len(batch_args)

            m_name = self.mod2name[mod]
            for i in range(len(batch_args)):
                batch_outputs.append(self._ori_forwards[mod](*batch_args[i], **batch_kwargs[i]))
                obs_group = ActivationObserver.find_group(self.inp_obs_group)
                mod_name = self.mod2name[mod]
                ActivationObserver.disable()
                auto_scale_block(mod, batch_kwargs[i], self.w_bits, self.w_group_size, obs_group, mod_name)
                ActivationObserver.enable()
            for key, item in obs_group.items():
                if key.startswith(f'{mod_name}.') and item.value is not None:
                    item.value.cpu()
                    del item.value

            outputs = concat_decoder_layer_outputs(batch_outputs)

            del batch_outputs, batch_args, batch_kwargs, args
            mod.cpu()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            max_memory = torch.cuda.max_memory_allocated(device=self.device) / (1 << 30)
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
