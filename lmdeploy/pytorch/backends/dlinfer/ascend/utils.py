# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29


def nd_to_nz_spec(tensor: torch.Tensor) -> torch.Tensor:
    '''
    This function is copied from vllm-ascend commit hash: 420e794c35fe887db2be81cf9db0461f5b71da0b
    It converts a tensor in ACL_FORMAT_ND format to ACL_FORMAT_FRACTAL_NZ format for Ascend 310P devices.
    It behaves similarly to the TransdataOperation and it requires the input tensor to be 2D.
    '''
    num_tokens = tensor.shape[0]
    max_seq_len = tensor.shape[1]

    tokens_pad = (num_tokens + 15) // 16 * 16
    max_seq_len_pad = (max_seq_len + 15) // 16 * 16

    tensor_pad = \
        torch.zeros((1, tokens_pad, max_seq_len_pad), dtype=tensor.dtype, device=tensor.device)

    tensor_pad[0][:num_tokens, :max_seq_len] = tensor
    tensor_nz = tensor_pad.reshape((1, tokens_pad, max_seq_len_pad // 16, 16)).permute(0, 2, 1, 3)

    tensor_nz = torch_npu.npu_format_cast(tensor_nz.contiguous(), ACL_FORMAT_FRACTAL_NZ)
    return tensor_nz
