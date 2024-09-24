# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import os.path as osp
import re
import sys
from typing import Any, Dict

import torch
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.utils import get_logger

from ..config import ModelConfig
from ..devices import get_device_manager
from .module_map import (CUSTOM_MODULE_MAP, DEVICE_SPECIAL_MODULE_MAP,
                         MODULE_MAP)

logger = get_logger('lmdeploy')


def _get_rewrite_qualname(origin_qualname: str, module_map: Dict[str,
                                                                 str]) -> str:
    """get rewrite module from origin module name.

    Args:
        origin_qualname (str): The origin qualname of the module.

    Returns:
        str: The rewrite qualname.
    """
    if origin_qualname in module_map:
        return module_map[origin_qualname]
    for key, value in module_map.items():
        if re.search(key, origin_qualname):
            return value
    return None


def _class_from_qualname(qualname: str) -> Any:
    """Import class with qualname.

    Args:
        qualname (str): Qualname of the class

    Returns:
        Any: class or builder of the class
    """
    last_dot = qualname.rfind('.')
    modname = qualname[:last_dot]
    clsname = qualname[last_dot + 1:]

    # get class at runtime
    mod = importlib.import_module(modname)
    assert mod is not None, f'failed to import module: {modname}'
    cls_type = getattr(mod, clsname)
    return cls_type


def _find_rewrite_module_qualname(model, module_map: Dict[str, str]):
    """find rewrite module."""
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__

    def _find_fullname():
        origin_qualname = f'{module_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_classname():
        origin_qualname = class_name
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_submodulename():
        # name with first module
        mod_name = module_name[module_name.rfind('.') + 1:]
        origin_qualname = f'{mod_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    rewrite_qualname = _find_fullname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_classname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_submodulename()

    origin_qualname = f'{module_name}.{class_name}'
    if rewrite_qualname is not None:
        logger.debug('Find rewrite of module\n'
                     f'{origin_qualname} <=> {rewrite_qualname}')
    return rewrite_qualname


def get_rewrite_cls(model: torch.nn.Module, module_map: Dict[str, str] = None):
    """get rewrite cls."""
    if module_map is None:
        module_map = _get_module_map()
    rewrite_qualname = _find_rewrite_module_qualname(model,
                                                     module_map=module_map)
    if rewrite_qualname is None:
        return None
    return _class_from_qualname(rewrite_qualname)


def _get_module_map():
    """get module map."""
    module_map = MODULE_MAP.copy()
    device_type = get_device_manager().current_context().device_type
    if device_type != 'cuda':
        device_map = DEVICE_SPECIAL_MODULE_MAP.get(device_type, dict())
        module_map.update(device_map)
    # add custom module map
    module_map.update(CUSTOM_MODULE_MAP)
    return module_map


def update_custom_module_map(module_map_path: str):
    """moad custom module map from file."""
    from importlib.machinery import SourceFileLoader

    from lmdeploy.pytorch.models.module_map import LMDEPLOY_PYTORCH_MODEL_PATH
    assert osp.exists(module_map_path), (
        f'custom module map path: "{module_map_path}" not exists.')

    module_map_path = osp.abspath(module_map_path)
    folder = osp.split(module_map_path)[0]
    sys.path.append(folder)
    custom_mod = SourceFileLoader('map_mod', module_map_path).load_module()
    sys.modules[f'{LMDEPLOY_PYTORCH_MODEL_PATH}._custom_mod'] = custom_mod

    new_mod_map = dict()
    has_map = False
    if hasattr(custom_mod, 'MODULE_MAP'):
        has_map = True
        mod_map = custom_mod.MODULE_MAP
        assert isinstance(mod_map, Dict)
        new_mod_map.update(mod_map)

    if hasattr(custom_mod, 'CUSTOM_MODULE_MAP'):
        has_map = True
        mod_map = custom_mod.CUSTOM_MODULE_MAP
        assert isinstance(mod_map, Dict)
        new_mod_map.update(mod_map)

    if not has_map:
        raise RuntimeError(f'Found no map in "{module_map_path}".')

    for k, v in new_mod_map.items():
        if '.' not in v:
            v = f'{LMDEPLOY_PYTORCH_MODEL_PATH}._custom_mod.{v}'
            new_mod_map[k] = v

    CUSTOM_MODULE_MAP.update(new_mod_map)


def _get_model_class(config, module_map):
    """get model class."""
    auto_map = getattr(config, 'auto_map', dict())
    if 'AutoModelForCausalLM' in auto_map:
        mapname = auto_map['AutoModelForCausalLM']
        if '.' in mapname:
            mapname = mapname.split('.')[-1]
        if mapname in module_map:
            qualname = module_map[mapname]
            module_cls = _class_from_qualname(qualname)
            return module_cls
        raise RuntimeError(f'Can not found rewrite for auto_map: {mapname}')

    architectures = getattr(config, 'architectures', [])
    for arch in architectures:
        if arch in module_map:
            qualname = module_map[arch]
            module_cls = _class_from_qualname(qualname)
            return module_cls

    raise RuntimeError(
        f'Can not found rewrite for architectures: {architectures}')


def build_model_from_hf_config(model_config: PretrainedConfig,
                               dtype: torch.dtype = None,
                               device: torch.device = None):
    """build model from hf config."""
    from lmdeploy.pytorch.model_inputs import StepContextManager
    ctx_mgr = StepContextManager()
    module_map = _get_module_map()
    if device is None:
        device = torch.device('cuda')
    model_cls = _get_model_class(model_config, module_map)
    model = model_cls(model_config, ctx_mgr, dtype=dtype, device=device)
    return model.eval()


@torch.inference_mode()
def build_patched_model(config: ModelConfig, device: torch.device = None):
    """build patched model."""
    model_config = config.hf_config
    dtype = config.dtype
    return build_model_from_hf_config(model_config, dtype=dtype, device=device)


@torch.inference_mode()
def add_adapters(model: torch.nn.Module,
                 adapters: Dict[str, str],
                 dtype: torch.dtype = torch.float16,
                 device: torch.device = None):
    """add adapters."""
    from peft import PeftConfig
    from peft.tuners.lora import LoraConfig

    from lmdeploy.pytorch.adapter.adapter import (find_all_target,
                                                  get_ranks_and_scalings,
                                                  load_lora_weights)
    from lmdeploy.pytorch.nn.linear import LoRA
    num_adapters = len(adapters)
    if num_adapters == 0:
        return

    if device is None:
        device = torch.device('cuda')

    # model could be graph runner
    if hasattr(model, 'get_model'):
        model = model.get_model()
    ctx_mgr = model.ctx_mgr

    adapter_names = list(adapters.keys())
    adapter_names = sorted(adapter_names)

    adapter_cfgs = [
        PeftConfig.from_pretrained(adapters[name]) for name in adapter_names
    ]

    # insert one for no adapter
    adapter_cfgs = [LoraConfig(r=0, target_modules=[])] + adapter_cfgs
    adapter_names = [None] + adapter_names
    adapter_id_map = dict(zip(adapter_names, range(len(adapter_names))))

    # target layer name to add adapter
    target_names = set()
    for cfg in adapter_cfgs:
        target_names = target_names.union(cfg.target_modules)
    target_names = list(target_names)
    target_names = sorted(target_names)

    target_infos = dict()
    for _, target_name in enumerate(target_names):
        # get ranks and scalings
        ranks, scalings = get_ranks_and_scalings(target_name,
                                                 adapter_cfgs,
                                                 device=device)
        found_mods, pack_idx = find_all_target(model, target_name)
        sum_rank = ranks.sum().item()

        in_features = 0
        out_features = 0
        colwise = True
        for _, mod in found_mods:
            assert hasattr(mod, 'lora_adapters')
            in_features = mod.in_features
            colwise = mod.colwise
            if pack_idx is None:
                base_slice = slice(0, mod.out_features)
                out_features = mod.out_features
                lora_b_spliter = getattr(mod, 'weight_spliter_lora_b', None)
            else:
                prev_feats = sum(mod.all_out_features[:pack_idx])
                out_features = mod.all_out_features[pack_idx]
                base_slice = slice(prev_feats, prev_feats + out_features)
                lora_b_spliter = None
            lora_a = torch.empty((sum_rank, in_features),
                                 dtype=dtype,
                                 device=device)
            lora_b = torch.empty((sum_rank, out_features),
                                 dtype=dtype,
                                 device=device)

            lora = LoRA(
                in_features,
                out_features,
                ranks=ranks,
                scalings=scalings,
                lora_a=lora_a,
                lora_b=lora_b,
                base_slice=base_slice,
                ctx_mgr=ctx_mgr,
                colwise=colwise,
                is_tp=mod.is_tp,
                lora_b_spliter=lora_b_spliter,
            )
            mod.lora_adapters[target_name] = lora

    # fill adapter data
    for name, path in adapters.items():
        adapter_id = adapter_id_map[name]
        checkpoint_path = f'{path}/adapter_model.bin'
        state_dict = torch.load(checkpoint_path, map_location=device)

        if hasattr(model, 'load_lora_weights'):
            model.load_lora_weights(state_dict.items(), adapter_id=adapter_id)
        else:
            load_lora_weights(model, state_dict.items(), adapter_id=adapter_id)

    return target_infos
