# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


def _check_env_v4(device: str = 'cuda'):
    """Environment check for DeepSeek-V4."""
    if device != 'cuda':
        return

    try:
        import deep_gemm  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <deep_gemm> to be installed.') from e

    try:
        import tile_kernels  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <tile_kernels> to be installed.') from e

    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <fast_hadamard_transform> to be installed.') from e

    if not hasattr(torch, 'float4_e2m1fn_x2'):
        raise RuntimeError('DeepSeek-V4 requires PyTorch with float4_e2m1fn_x2 support.')


class DeepseekV4ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'deepseek_v4'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, tp: int = 1, **kwargs):
        """Build model config.

        DeepSeek-V4 manages its own compressed/window caches instead of using
        lmdeploy's generic paged KV layout. We still keep a tiny dummy cache
        shape so the scheduler / block manager remain operational, while stable
        per-sequence cache slots are provided by `state_offsets`.
        """
        dummy_heads = max(tp, 1)
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=1,
            num_attention_heads=dummy_heads,
            num_key_value_heads=dummy_heads,
            bos_token_id=bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=1,
            k_head_dim=1,
            v_head_dim=1,
            sliding_window=hf_config.sliding_window,
            vocab_size=hf_config.vocab_size,
            model_paradigm='ar',
        )

        # Enable stable logical-state allocation so V4 can index its own
        # compressed cache slots without depending on paged KV blocks.
        config.states_shapes = [((1,), torch.uint8)]
        config.check_env_func = _check_env_v4
        return config
