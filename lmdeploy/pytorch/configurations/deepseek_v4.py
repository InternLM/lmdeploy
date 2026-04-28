# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import BlockCacheSpec, ModelConfig, StateCacheSpec

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

        Declares real V4 cache footprint via block_cache_specs / state_cache_specs and removes the dummy KV cache shapes
        used during bring-up.
        """
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        head_dim = getattr(hf_config, 'head_dim', 512)
        num_layers = hf_config.num_hidden_layers
        compress_ratios = getattr(hf_config, 'compress_ratios', None) or [0] * num_layers

        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=num_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', 1),
            bos_token_id=bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=head_dim,
            sliding_window=hf_config.sliding_window,
            vocab_size=hf_config.vocab_size,
            model_paradigm='ar',
            use_standard_kv_cache=False,
        )

        # ---- block cache specs ----
        all_layers = list(range(num_layers))
        block_specs = [
            BlockCacheSpec('v4_raw_kv', all_layers, (head_dim,), torch.bfloat16),
        ]

        ratio4_layers = [i for i, r in enumerate(compress_ratios) if r == 4]
        ratio128_layers = [i for i, r in enumerate(compress_ratios) if r == 128]

        if ratio4_layers:
            block_specs.append(
                BlockCacheSpec('v4_compressed_kv_r4', ratio4_layers, (head_dim,), torch.bfloat16))
            index_head_dim = getattr(hf_config, 'index_head_dim', 128)
            block_specs.append(
                BlockCacheSpec('v4_index_kv_r4', ratio4_layers, (index_head_dim,), torch.bfloat16))

        if ratio128_layers:
            block_specs.append(
                BlockCacheSpec('v4_compressed_kv_r128', ratio128_layers, (head_dim,), torch.bfloat16))

        config.block_cache_specs = block_specs

        # ---- state cache specs ----
        state_specs = []
        if ratio4_layers:
            # overlap compressor scratch for Attention (kv_state + score_state)
            # rows = 2 * ratio = 8, state_dim = 2 * head_dim
            # compress_state shape = (2 * rows, state_dim) = (16, 2 * head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r4', (16, 2 * head_dim), torch.float32, layer_ids=ratio4_layers))
            index_head_dim = getattr(hf_config, 'index_head_dim', 128)
            # Indexer also has its own compressor (overlap=True because ratio==4)
            # rows = 2 * ratio = 8, state_dim = 2 * index_head_dim
            # compress_state shape = (2 * rows, state_dim) = (16, 2 * index_head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r4_idx',
                               (16, 2 * index_head_dim),
                               torch.float32,
                               layer_ids=ratio4_layers))

        if ratio128_layers:
            # rows = ratio = 128, state_dim = head_dim
            # compress_state shape = (2 * rows, state_dim) = (256, head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r128', (256, head_dim), torch.float32, layer_ids=ratio128_layers))

        config.state_cache_specs = state_specs
        # backward-compat bridge to keep scheduler.is_ssm working
        config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in state_specs]

        config.check_env_func = _check_env_v4
        return config
