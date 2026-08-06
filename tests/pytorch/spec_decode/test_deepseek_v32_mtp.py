# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lmdeploy.pytorch.models.deepseek_v32 import DeepseekV32DecoderLayer
from lmdeploy.pytorch.models.deepseek_v32_mtp import DeepseekV32MTPModel


def test_deepseek_v32_mtp_builds_sparse_decoder_layer():
    config = SimpleNamespace(quantization_config=None)

    with patch('lmdeploy.pytorch.models.deepseek_mtp.DeepSeekMultiTokenPredictor',
               return_value=nn.Module()) as build_predictor:
        DeepseekV32MTPModel(config,
                           ctx_mgr=None,
                           dtype=torch.bfloat16,
                           device=torch.device('cpu'))

    assert build_predictor.call_args.kwargs[
        'decoder_layer_cls'] is DeepseekV32DecoderLayer
