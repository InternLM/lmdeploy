# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder
from .utils import flash_mla_available


def _check_env_v32(device: str = 'cuda'):
    """Environment check."""
    if device != 'cuda':
        return

    # check cuda
    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError:
        raise ImportError('Deepseek V3.2 requires <fast_hadamard_transform>.')

    try:
        import flash_mla  # noqa: F401
    except ImportError:
        raise ImportError('Deepseek V3.2 requires <flash_mla>.')

    if not hasattr(flash_mla, 'flash_mla_sparse_fwd'):
        raise RuntimeError('Latest flash_mla is required: https://github.com/deepseek-ai/FlashMLA.')


class DeepseekV2ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['deepseek_v3', 'deepseek_v2', 'deepseek_v32', 'kimi_k2']

    @classmethod
    def build(cls, hf_config, model_path: str = None, is_draft_model: bool = False, spec_method: str = None, **kwargs):
        """build."""
        head_dim = (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)
        k_head_dim = head_dim
        v_head_dim = 0
        num_attention_heads = hf_config.num_attention_heads
        # multi query attn
        num_key_value_heads = 1
        tp = kwargs.get('tp', 1)
        # update num_kv_heads for tp mode
        num_key_value_heads = cls.update_num_kv_heads(hf_config, tp, num_key_value_heads)
        hf_config.use_flash_mla = flash_mla_available()
        num_layers = hf_config.num_hidden_layers
        model_paradigm = 'ar'

        if spec_method is not None:
            assert spec_method == 'deepseek_mtp'

        # draft model cfg
        if is_draft_model:
            num_layers = hf_config.num_nextn_predict_layers
            hf_config.architectures[0] = 'DeepseekMTPModel'
            # remove for correct mapping when building the patched model
            del hf_config.auto_map

        if is_draft_model or spec_method is not None:
            model_paradigm = 'ar_spec'

        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=head_dim,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            vocab_size=hf_config.vocab_size,
            use_flash_mla=hf_config.use_flash_mla,
            model_paradigm=model_paradigm,
        )

        if hf_config.model_type == 'deepseek_v32':
            assert hf_config.use_flash_mla, 'DeepSeek-V3.2 requires flash_mla to be available.'
            index_k_shape = ([hf_config.index_head_dim], torch.float8_e4m3fn)
            index_k_scale_shape = ([1], torch.float32)
            config.cache_shapes = [index_k_shape, index_k_scale_shape]
            config.use_mla_fp8_cache = True
            config.mla_index_topk = hf_config.index_topk
            config.check_env_func = _check_env_v32
        return config
