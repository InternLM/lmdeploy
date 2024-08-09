# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import numpy as np
import torch

from ..source_model.base import BaseInputModel, BaseReader
from .base import (OUTPUT_MODELS, BaseOutputModel, TurbomindModelConfig,
                   merge_qkv, permute, tprint)


def get_cuda_tensor(tensors):
    """Get cuda tensor."""
    result = map(lambda x: x.cuda() if x is not None else x, tensors)
    return (*result, )


def get_qqq_perms(group_size: int):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    4 * (i % 4), 4 * (i % 4) + 1, 4 * (i % 4) + 2,
                    4 * (i % 4) + 3
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    if group_size == -1:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    scale_perm = torch.from_numpy(np.array(scale_perm))
    scale_perm_single = torch.from_numpy(np.array(scale_perm_single))
    return perm, scale_perm, scale_perm_single


def pack(w: torch.Tensor,
         s_channel: torch.Tensor,
         s_group: torch.Tensor,
         group_size: int,
         tile: int = 16):
    assert w.dim() == 2
    infeatures, outfeatures = w.shape[0], w.shape[1]
    _perm, _scale_perm, _scale_perm_single = get_qqq_perms(group_size)
    org_device = w.device
    # permute scales
    if group_size != -1 and group_size < infeatures:
        s_group = s_group.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        s_group = s_group.reshape((-1, outfeatures)).contiguous()
    s_channel = s_channel.reshape(
        (-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s_channel = s_channel.reshape((-1, outfeatures)).contiguous()
    # permute and pack weight
    w = w.reshape((
        infeatures // tile,
        tile,
        outfeatures // tile,
        tile,
    ))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((infeatures // tile, outfeatures * tile))
    res = w
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    if group_size != -1 and group_size < infeatures:
        for i in range(8):
            q |= res[:, i::8] << 4 * i
    else:
        for i in range(8):
            q |= (res[:, i::8] & 0xF) << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(org_device)
    return q, s_channel, s_group


def unpack(w: torch.Tensor,
           s_channel: torch.Tensor,
           s_group: torch.Tensor,
           group_size: int,
           tile: int = 16,
           wbits: int = 4):
    assert w.dim() == 2
    pack_factor = 32 // wbits
    infeatures = w.shape[0] * tile
    outfeatures = w.shape[1] * pack_factor // tile
    org_device = w.device
    _perm, _scale_perm, _scale_perm_single = get_qqq_perms(group_size)
    wf = torch.tensor(list(range(0, 32, 4)),
                      dtype=torch.int32).unsqueeze(0).to(org_device)
    # unpack weight
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(w, 2).expand(-1, -1, 32 // wbits),
        wf.unsqueeze(0),
    )
    weight = torch.bitwise_and(weight, (2**wbits) - 1)
    weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])

    # reshape weight and scale
    _perm_inv = torch.argsort(_perm)
    _scale_perm_inv = torch.argsort(_scale_perm)
    _scale_perm_single_inv = torch.argsort(_scale_perm_single)

    weight = weight.reshape(-1, _perm.numel())[:, _perm_inv]
    weight = weight.reshape((
        infeatures // tile,
        outfeatures // tile,
        tile,
        tile,
    ))
    weight = weight.permute((0, 2, 1, 3))
    weight = weight.reshape((infeatures, outfeatures))
    s_channel = s_channel.reshape(
        -1, len(_scale_perm_single))[:, _scale_perm_single_inv].reshape(
            -1, outfeatures)
    if group_size != -1 and group_size < infeatures:
        s_group = s_group.reshape(
            -1, len(_scale_perm))[:, _scale_perm_inv].reshape(-1, outfeatures)

    return weight, s_channel, s_group


def permute_qk(w: torch.Tensor,
               s_channel: torch.Tensor,
               s_group: torch.Tensor,
               group_size: int,
               size_per_head: int = 128):
    unp_w, unp_s_channel, unp_s_group = unpack(w, s_channel, s_group,
                                               group_size)
    dim = unp_w.shape[-1]
    n_heads = dim // size_per_head
    perm_w = unp_w.view(-1, n_heads, 2,
                        dim // n_heads // 2).transpose(2, 3).reshape(-1, dim)
    perm_s_channel = unp_s_channel.view(-1, n_heads,
                                        2, dim // n_heads // 2).transpose(
                                            2, 3).reshape(-1, dim)
    perm_s_group = unp_s_group
    if group_size != -1 and group_size < unp_w.shape[0]:
        perm_s_group = unp_s_group.view(-1, n_heads,
                                        2, dim // n_heads // 2).transpose(
                                            2, 3).reshape(-1, dim)
    p_w, p_s_channel, p_s_group = pack(perm_w, perm_s_channel, perm_s_group,
                                       group_size)
    return p_w, p_s_channel, p_s_group


@OUTPUT_MODELS.register_module(name='qqq-w4')
class TurbomindQQQW4Model(BaseOutputModel):
    """Export to turbomind QQQ w4a8 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        self.weight_bits = 4
        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // self.weight_bits
        self.tile_size = 16
        # supported group size
        self.supported_group_size = [-1, 128]
        # Min out_features dim
        self.min_n_threads = 64
        # Min in_features dim
        self.min_k_threads = 128
        # Permutation length used by the QQQ kernels.
        self.perm_len = 1024
        super().__init__(input_model, cfg, to_file, out_dir)

    def get_config(self, cfg: TurbomindModelConfig):
        """Get turbomind config."""
        final_cfg = super().get_config(cfg).__dict__

        # attn_bias, inter_size
        visit = False
        attn_bias = 0
        for bin in self.input_model.bins():
            for i in range(bin.start_layer_id, bin.end_layer_id):
                visit = True
                w1s, _, _ = bin.ffn_scale_channel(i)
                inter_size = w1s.shape[-1]
                qb, _, _, _ = bin.attn_bias(i)
                if qb is not None:
                    attn_bias = 1
                break
            if visit:
                break
        final_cfg.update(dict(attn_bias=attn_bias, inter_size=inter_size))
        final_cfg = TurbomindModelConfig.from_dict(final_cfg)

        if final_cfg.group_size not in self.supported_group_size:
            raise ValueError(f'The group_size of QQQ should be in'
                             f'{self.supported_group_size}')
        # check weight size
        hidden_size = final_cfg.head_num * final_cfg.size_per_head
        merge_qkv_size = (final_cfg.head_num +
                          2 * final_cfg.kv_head_num) * final_cfg.size_per_head
        tp = final_cfg.tensor_para_size
        weight_info = {
            'mgrge_qkv': {
                'weight_size': [hidden_size, merge_qkv_size],
                'split_dim': -1
            },
            'o': {
                'weight_size': [hidden_size, hidden_size],
                'split_dim': 0
            },
            'w1': {
                'weight_size': [hidden_size, inter_size],
                'split_dim': -1
            },
            'w2': {
                'weight_size': [hidden_size, inter_size],
                'split_dim': -1
            },
            'w3': {
                'weight_size': [inter_size, hidden_size],
                'split_dim': 0
            },
        }
        for weight_name, split_info in weight_info.items():
            self.check_weight_size(weight_name, split_info['weight_size'], tp,
                                   final_cfg.group_size,
                                   split_info['split_dim'])
        return final_cfg

    def check_weight_size(self, weight_name: str, weight_size: List[int],
                          tp: int, group_size: int, split_dim: int):
        assert weight_size[
            split_dim] % tp == 0, 'The split size must be divisible by tp size'
        input_size_per_partition = weight_size[
            0] // tp if split_dim == 0 else weight_size[0]
        output_size_per_partition = weight_size[
            -1] // tp if split_dim == -1 else weight_size[-1]
        # Validate output_size_per_partition
        if output_size_per_partition % self.min_n_threads != 0:
            raise ValueError(
                f'{weight_name} weight output_size_per_partition = '
                f'{output_size_per_partition} is not divisible by '
                f'min_n_threads = {self.min_n_threads}.')
        if output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f'{weight_name} weight output_size_per_partition = '
                f'{output_size_per_partition} is not divisible by '
                f'pack_factor = {self.pack_factor}.')

        # Validate input_size_per_partition
        if input_size_per_partition % self.min_k_threads != 0:
            raise ValueError(
                f'{weight_name} weight input_size_per_partition = '
                f'{input_size_per_partition} is not divisible by '
                f'min_k_threads = {self.min_k_threads}.')
        if (group_size != -1 and input_size_per_partition % group_size != 0):
            raise ValueError(
                f'{weight_name} weight input_size_per_partition = '
                f'{input_size_per_partition} is not divisible by '
                f'group_size = {group_size}.')

        # Check that we have at least 4 tiles horizontally in the shard
        num_tiles_per_perm = self.perm_len // (self.tile_size**2)
        if output_size_per_partition % num_tiles_per_perm != 0:
            raise ValueError(
                'Each permutation group must reside on the same gpu')

    def export_transformer_block(self, bin: BaseReader, i: int):
        """Export transformer layer i."""
        group_size = self.cfg.group_size
        tp = self.cfg.tensor_para_size
        size_per_head = self.cfg.size_per_head
        # attn
        q_qw, k_qw, v_qw, o_qw = get_cuda_tensor(bin.attn(i))
        q_sc, k_sc, v_sc, o_sc = get_cuda_tensor(bin.attn_scale_channel(i))
        q_sg, k_sg = None, None
        if group_size != -1:
            q_sg, k_sg, v_sg, o_sg = get_cuda_tensor(bin.attn_scale_group(i))

        # TODO(HandH1998): verify correctness
        q_qw, q_sc, q_sg = permute_qk(q_qw, q_sc, q_sg, group_size,
                                      size_per_head)
        k_qw, k_sc, k_sg = permute_qk(k_qw, k_sc, k_sg, group_size,
                                      size_per_head)

        qkv_qw = merge_qkv(q_qw, k_qw, v_qw, tp, dim=2)
        qkv_sc = merge_qkv(q_sc, k_sc, v_sc, tp, dim=2)

        self.save_split(qkv_qw, f'layers.{i}.attention.w_qkv.qweight', -1)
        self.save_split(qkv_sc, f'layers.{i}.attention.w_qkv.scales_channel',
                        -1)

        self.save_split(o_qw, f'layers.{i}.attention.wo.qweight', 0)
        # TODO(HandH1998): verify tp > 1
        self.save_split(o_sc,
                        f'layers.{i}.attention.wo.scales_channel',
                        copy=True)

        if group_size != -1:
            qkv_sg = merge_qkv(q_sg, k_sg, v_sg, tp, dim=2)
            self.save_split(qkv_sg, f'layers.{i}.attention.w_qkv.scales_zeros',
                            -1)
            self.save_split(o_sg, f'layers.{i}.attention.wo.scales_zeros', 0)

        q_b, k_b, v_b, o_b = get_cuda_tensor(bin.attn_bias(i))
        if q_b is not None:
            q_b = permute(q_b, size_per_head)
            k_b = permute(k_b, size_per_head)
            qkv_b = merge_qkv(q_b, k_b, v_b, tp, dim=1)
            self.save_split(qkv_b, f'layers.{i}.attention.w_qkv.bias', -1)
            self.save_split(o_b, f'layers.{i}.attention.wo.bias', copy=True)

        # ffn weights
        w1_qw, w2_qw, w3_qw = get_cuda_tensor(bin.ffn(i))
        w1_sc, w2_sc, w3_sc = get_cuda_tensor(bin.ffn_scale_channel(i))

        self.save_split(w1_qw, f'layers.{i}.feed_forward.w1.qweight', -1)
        self.save_split(w1_sc, f'layers.{i}.feed_forward.w1.scales_channel',
                        -1)
        self.save_split(w3_qw, f'layers.{i}.feed_forward.w3.qweight', -1)
        self.save_split(w3_sc, f'layers.{i}.feed_forward.w3.scales_channel',
                        -1)
        self.save_split(w2_qw, f'layers.{i}.feed_forward.w2.qweight', 0)
        # TODO(HandH1998): verify tp > 1
        self.save_split(w2_sc,
                        f'layers.{i}.feed_forward.w2.scales_channel',
                        copy=True)

        if group_size != -1:
            w1_sg, w2_sg, w3_sg = get_cuda_tensor(bin.ffn_scale_group(i))
            self.save_split(w1_sg, f'layers.{i}.feed_forward.w1.scales_zeros',
                            -1)
            self.save_split(w3_sg, f'layers.{i}.feed_forward.w3.scales_zeros',
                            -1)
            self.save_split(w2_sg, f'layers.{i}.feed_forward.w2.scales_zeros',
                            0)

        # norm
        attn_norm = bin.attn_norm(i)
        ffn_norm = bin.ffn_norm(i)
        self.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')

    def export_weight(self, param: torch.Tensor, name: str) -> None:
        """export turbomind weight."""

        def _tofile(tensor, path):
            """to file."""
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.half)
            tensor.contiguous().cpu().numpy().tofile(path)

        if self.to_file:
            tprint(name, param.shape)
            _tofile(param, osp.join(self.out_dir, name))
        elif len(self.tm_params) > 0:
            tm_params = self.tm_params
            # currently, the tensor type should in
            # [torch.float, torch.half, torch.bfloat16, torch.int32]
            torch_tensor = param.cuda().contiguous()
            assert torch_tensor.dtype in [
                torch.int32, torch.float, torch.half, torch.bfloat16
            ]
            for tm_tensor in tm_params[name]:
                tm_tensor.copy_from(torch_tensor)
            tm_params.pop(name)
        else:
            tprint('skip export', name, param.shape)
