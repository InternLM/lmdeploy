# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import ABC

import torch
import tqdm
import yaml
from mmengine import Registry

from ..config import (AttentionConfig, LoraConfig, ModelConfig,
                      TurbomindModelConfig, config_from_dict, config_to_dict)
from ..source_model.base import BaseInputModel

OUTPUT_MODELS = Registry(
    'target model', locations=['lmdeploy.turbomind.deploy.target_model.base'])


def tprint(*args, **kwargs):
    to_file = kwargs.pop('to_file', False)
    if not to_file:
        return
    from io import StringIO
    s = StringIO()
    print(*args, **kwargs, file=s, end='')
    tqdm.tqdm.write(s.getvalue())


def _weight_dtype_map(weight_type: str, default=None):
    """map literal data type to torch dtype."""

    _WEIGHT_DTYPE_MAP = dict(int4=torch.float16,
                             float16=torch.float16,
                             float32=torch.float16,
                             bfloat16=torch.bfloat16)

    return _WEIGHT_DTYPE_MAP.get(weight_type, default)


def _pad_inter_size(inter_size: int, group_size: int, tp: int):
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


class BaseOutputModel(ABC):
    """Base output model."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 model_cls,
                 out_dir: str = ''):
        super().__init__()
        self.input_model = input_model
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.tensor_para_size = self.model_config.tp
        self.out_dir = out_dir
        self.to_file = True if out_dir else False
        self.tm_params = {}

        # get `model_info` and `tokenizer_info` at first, which
        # will be updated to `self.model_config` and `self.attention_config`
        self.input_model_info = self.input_model.model_info()
        self.input_model_tokenizer_info = self.input_model.tokenizer_info()
        self.permute_qk = self.input_model_info.get('permute_qk', True)

        self.update_model_config()
        self.model_config.inter_size = _pad_inter_size(
            self.model_config.inter_size, self.model_config.group_size,
            self.tensor_para_size)
        if self.model_config.expert_num:
            self.model_config.expert_inter_size = _pad_inter_size(
                self.model_config.expert_inter_size,
                self.model_config.group_size, self.tensor_para_size)
        self.model_config.verify()
        assert self.model_config.kv_head_num % self.tensor_para_size == 0

        self.update_attention_config()
        self.update_lora_config()
        # ! Dependency on `self`
        self.model = model_cls(self)

    def update_model_config(self):
        """Update `self.model_config` according to the input_model's
        `tokenizer_info` and `model_info`"""
        _, bos_id, eos_id = self.input_model_tokenizer_info

        final_cfg = config_to_dict(self.model_config)
        final_cfg.update(dict(start_id=bos_id, end_id=eos_id))
        final_cfg.update(self.input_model_info)
        if 'embedding_size' not in self.input_model_info.keys():
            final_cfg.update(
                embedding_size=self.input_model_info['vocab_size'])

        self.model_config = config_from_dict(ModelConfig, final_cfg)

    def update_attention_config(self):
        """update attention config according to input model's model info."""
        final_cfg = config_to_dict(self.attention_config)
        final_cfg.update(self.input_model_info)
        self.attention_config = config_from_dict(AttentionConfig, final_cfg)

    def update_lora_config(self):
        """update lora config according to input model's model info."""
        final_cfg = config_to_dict(self.lora_config)
        final_cfg.update(self.input_model_info)
        self.lora_config = config_from_dict(LoraConfig, final_cfg)

    def export_config(self) -> None:
        """export turbomind config."""
        if self.to_file:
            config_path = osp.join(self.out_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(self.tm_config.to_dict(), f)

    def export_weight(self, param: torch.Tensor, name: str) -> None:
        """export turbomind weight."""

        def _tofile(tensor, path):
            """to file."""
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.half)
            tensor.contiguous().cpu().numpy().tofile(path)

        if self.to_file:
            if torch.is_floating_point(param):
                torch_type = _weight_dtype_map(self.model_config.weight_type,
                                               torch.float16)
                param = param.to(torch_type)
            tprint(name, param.shape)
            _tofile(param, osp.join(self.out_dir, name))
        elif len(self.tm_params) > 0:
            tm_params = self.tm_params
            weight_type = self.model_config.weight_type
            assert weight_type in ['float16', 'bfloat16', 'int4']

            # currently, the tensor type should in
            # [torch.float, torch.half, torch.bfloat16, torch.int32]
            torch_tensor = param.cuda().contiguous()
            assert torch_tensor.dtype in [
                torch.int32, torch.float, torch.half, torch.bfloat16
            ]
            if torch_tensor.dtype != torch.int32:
                if weight_type in ['float16', 'int4']:
                    torch_tensor = torch_tensor.half()
                elif weight_type == 'bfloat16':
                    torch_tensor = torch_tensor.bfloat16()
                else:
                    torch_tensor = torch_tensor.half()
            for tm_tensor in tm_params[name]:
                tm_tensor.copy_from(torch_tensor)
            tm_params.pop(name)
        else:
            tprint('skip export', name, param.shape)

    def save_split(self,
                   tensor: torch.Tensor,
                   name: str,
                   split_dim=None,
                   copy=False) -> None:
        """save split.

        - 2D input
            shape must be (input_dims, output_dims)
        - 1D input (bias)
            shape must be (output_dims)
            split is skipped when split_dim == 0
        """

        if copy or (tensor.dim() == 1 and split_dim == 0):
            split_dim = None
            copy = True

        tp = self.tensor_para_size
        if split_dim is not None:
            tprint(
                f'*** splitting {name}, shape={tensor.shape}, '
                f'split_dim={split_dim}, tp={tp}',
                to_file=self.to_file)
            if tensor.shape[split_dim] % tp != 0:
                raise RuntimeError(
                    f'{name}: shape={list(tensor.shape)}, tp={tp}')
            split_size = tensor.shape[split_dim] // tp
            splits = torch.split(tensor, split_size, dim=split_dim)
            for i, split in enumerate(splits):
                prefix, ext = osp.splitext(name)
                self.export_weight(split, f'{prefix}.{i}{ext}')
        elif copy:
            tprint(f'### copying {name}, shape={tensor.shape}',
                   to_file=self.to_file)
            copies = [tensor] * tp
            for i, copy in enumerate(copies):
                prefix, ext = osp.splitext(name)
                self.export_weight(copy, f'{prefix}.{i}{ext}')
        else:
            self.export_weight(tensor, name)

    def export(self) -> None:
        """Export to turbomind model format."""
        num_layer = self.model_config.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer,
                    desc='Convert to turbomind format',
                    leave=self.to_file)
        self.export_config()
        for i, reader in self.input_model.readers():
            if self.model(i, reader):
                pbar.update(1)
        pbar.close()
        # manually clean up meta reader
        if hasattr(self.input_model, 'meta_reader'):
            self.input_model.meta_reader.clean_up(True)
            del self.input_model.meta_reader
            torch.cuda.empty_cache()

    @property
    def tm_config(self):
        return TurbomindModelConfig(model_config=self.model_config,
                                    attention_config=self.attention_config,
                                    lora_config=self.lora_config)
