# Copyright (c) OpenMMLab. All rights reserved.
import configparser
import inspect
import os.path as osp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import tqdm
from mmengine import Registry

from lmdeploy.model import MODELS

from ..source_model.base import BaseInputModel, BaseReader

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


@dataclass
class TurbomindModelConfig:
    """Config for turbomind model."""
    model_name: str
    tensor_para_size: int
    head_num: int
    kv_head_num: int
    vocab_size: int
    num_layer: int
    inter_size: int
    norm_eps: float
    attn_bias: int
    start_id: int
    end_id: int
    session_len: int
    weight_type: str = 'fp16'
    rotary_embedding: int = 128
    rope_theta: float = 10000.0
    size_per_head: int = 128
    group_size: int = 0
    max_batch_size: int = 64
    max_context_token_num: int = 4
    step_length: int = 1
    cache_max_entry_count: float = 0.5
    cache_block_seq_len: int = 128
    cache_chunk_size: int = 1
    use_context_fmha: int = 1
    quant_policy: int = 0
    max_position_embeddings: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: int = 0

    @classmethod
    def from_dict(cls, env, allow_none=False):
        """Construct from dict."""
        params = inspect.signature(cls).parameters
        used = {k: v for k, v in env.items() if k in params and v is not None}
        if not allow_none:
            return cls(**used)
        else:
            default = {
                k: None
                for k in params.keys() if params[k].default is inspect._empty
            }
            default.update(used)
            return cls(**default)

    @property
    def valid(self):
        """Check if cfg is valid."""
        for _, v in self.__dict__.items():
            if v is None:
                return False
        return True


class BaseOutputModel(ABC):
    """Base output model."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__()
        self.input_model = input_model
        self.cfg = cfg
        if not cfg.valid:
            self.cfg = self.get_config(cfg)
        assert self.cfg.valid
        self.to_file = to_file
        self.out_dir = out_dir
        self.tm_params = {}

    @abstractmethod
    def get_config(self, cfg: TurbomindModelConfig) -> TurbomindModelConfig:
        """Generate turbomind model config (config.ini)."""
        _, bos_id, eos_id = self.input_model.tokenizer_info()
        model = MODELS.get(cfg.model_name)()
        final_cfg = cfg.__dict__
        final_cfg.update(
            dict(start_id=bos_id,
                 end_id=eos_id,
                 session_len=model.session_len + 8))
        final_cfg.update(self.input_model.model_info())

        # head_num, vocab_size
        for bin in self.input_model.bins():
            emb = bin.tok_embeddings()
            if emb is not None:
                _vocab_size, dim = emb.shape
                head_num = dim // cfg.size_per_head
                break
        final_cfg.update(dict(head_num=head_num, vocab_size=_vocab_size))
        return TurbomindModelConfig.from_dict(final_cfg, allow_none=True)

    def export_config(self) -> None:
        """export turbomind config."""
        if self.to_file:
            config = configparser.ConfigParser()
            cfg = dict(llama=self.cfg.__dict__)
            for section, key_values in cfg.items():
                config[section] = key_values
            config_path = osp.join(self.out_dir, 'config.ini')
            with open(config_path, 'w') as f:
                config.write(f)

    def export_weight(self, param: torch.Tensor, name: str) -> None:
        """export turbomind weight."""
        if self.to_file:
            if param.dtype in [torch.float, torch.bfloat16]:
                param = param.half()
            tprint(name, param.shape)
            param.contiguous().cpu().numpy().tofile(
                osp.join(self.out_dir, name))
        elif len(self.tm_params) > 0:
            tm_params = self.tm_params
            weight_type = self.cfg.weight_type
            assert weight_type in ['fp16', 'fp32', 'int4']

            # currently, the tensor type should in
            # [torch.float, torch.half, torch.int32]
            torch_tensor = param.cuda().contiguous()
            assert torch_tensor.dtype in [
                torch.int32, torch.float, torch.half, torch.bfloat16
            ]
            if torch_tensor.dtype != torch.int32:
                if weight_type in ['fp16', 'int4']:
                    torch_tensor = torch_tensor.half()
                else:
                    torch_tensor = torch_tensor.float()
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
        """save split."""
        tp = self.cfg.tensor_para_size
        if split_dim is not None:
            tprint(
                f'*** splitting {name}, shape={tensor.shape}, '
                f'split_dim={split_dim}, tp={tp}',
                to_file=self.to_file)
            assert tensor.shape[split_dim] % tp == 0
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
        num_layer = self.cfg.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer,
                    desc='Convert to turbomind format',
                    leave=self.to_file)
        self.export_config()
        for bin in self.input_model.bins():
            self.export_misc(bin)
            for i in range(bin.start_layer_id, bin.end_layer_id):
                self.export_transformer_block(bin, i)
                pbar.update(1)
        pbar.close()
        # manually clean up meta reader
        if hasattr(self.input_model, 'meta_reader'):
            self.input_model.meta_reader.clean_up(True)
            del self.input_model.meta_reader
            torch.cuda.empty_cache()

    def export_misc(self, bin: BaseReader) -> None:
        """Export embedding, norm, output weight."""
        emb = bin.tok_embeddings()
        norm_weight = bin.norm_weight()
        output_weight = bin.output_weight()

        def pad_weight(tensor):
            pad_size = None
            vocab_size = self.cfg.vocab_size
            tp = self.cfg.tensor_para_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size

            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size),
                                           'constant', 0)

        if emb is not None:
            emb = pad_weight(emb)
            self.export_weight(emb, 'tok_embeddings.weight')
        if norm_weight is not None:
            self.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight)
            self.export_weight(output_weight, 'output.weight')

    @abstractmethod
    def export_transformer_block(self, bin: BaseReader, i: int) -> None:
        """Export transformer block."""
        pass


def permute(x: torch.Tensor, size_per_head: int = 128):
    if x.shape[-1] > 1:
        dim = x.shape[-1]
        n_heads = dim // size_per_head
        return x.view(-1, n_heads, 2,
                      dim // n_heads // 2).transpose(2, 3).reshape(-1, dim)
    else:  # scales, zeros
        dim = x.shape[0]
        n_heads = dim // size_per_head
        return x.view(n_heads, 2, dim // n_heads // 2,
                      1).transpose(1, 2).reshape(dim, 1)


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int,
              dim: int):

    def reshape(x):
        return x.view(x.size(0), tp, -1) if dim == 2 else x.view(tp, -1)

    qkv = torch.cat((reshape(q), reshape(k), reshape(v)), dim=-1)
    # (input_dim, head_num + 2 * kv_head_num)
    return qkv.view(q.size(0), -1)
