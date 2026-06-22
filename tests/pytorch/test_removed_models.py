import pytest
import torch
from transformers import PretrainedConfig

from lmdeploy.pytorch.models.patch import build_model_from_hf_config


@pytest.mark.parametrize('arch', [
    'InternLMForCausalLM',
    'QWenLMHeadModel',
    'BaiChuanForCausalLM',
    'BaichuanForCausalLM',
    'Starcoder2ForCausalLM',
    'InternLM2VEForCausalLM',
    'MllamaForConditionalGeneration',
])
def test_removed_model_error(arch):
    config = PretrainedConfig(architectures=[arch])

    with pytest.raises(RuntimeError, match='support has been removed from LMDeploy'):
        build_model_from_hf_config(config, device=torch.device('cpu'))
