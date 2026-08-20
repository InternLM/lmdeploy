# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from lmdeploy.pytorch.config import BlockCacheSpec, ModelConfig, StateCacheSpec

from .builder import AutoModelConfigBuilder


def _get_mimo_cache_layers(hf_config):
    """Validate the hybrid pattern and return full/SWA layer partitions."""
    num_layers = hf_config.num_hidden_layers
    pattern = list(hf_config.hybrid_layer_pattern)
    if len(pattern) != num_layers:
        raise ValueError(
            "MiMo-V2-Flash hybrid_layer_pattern must contain one entry per layer, "
            f"but got {len(pattern)} entries for num_hidden_layers={num_layers}."
        )

    invalid = sorted(set(pattern) - {0, 1})
    if invalid:
        raise ValueError(
            f"MiMo-V2-Flash hybrid_layer_pattern only supports 0 (full attention) and 1 (SWA), but got {invalid}."
        )

    full_layers = [i for i, layer_type in enumerate(pattern) if layer_type == 0]
    swa_layers = [i for i, layer_type in enumerate(pattern) if layer_type == 1]
    if not full_layers or not swa_layers:
        raise ValueError("MiMo-V2-Flash requires both full-attention and SWA layers.")
    return full_layers, swa_layers


def _num_kv_heads_per_rank(num_heads: int, tp: int, attention_type: str) -> int:
    """Return local KV heads, including replication when TP exceeds KV heads."""
    if num_heads >= tp:
        if num_heads % tp != 0:
            raise ValueError(f"MiMo-V2-Flash {attention_type} KV heads ({num_heads}) must be divisible by TP ({tp}).")
        return num_heads // tp
    if tp % num_heads != 0:
        raise ValueError(f"MiMo-V2-Flash TP ({tp}) must be divisible by {attention_type} KV heads ({num_heads}).")
    return 1


def _update_kv_heads_for_tp(hf_config, *, heads_attr: str, replicate_attr: str, tp: int, attention_type: str) -> int:
    """Record independent Full/SWA KV replication metadata on HF config."""
    original_heads_attr = f"mimo_original_{heads_attr}"
    num_heads = getattr(hf_config, original_heads_attr, getattr(hf_config, heads_attr))
    setattr(hf_config, original_heads_attr, num_heads)
    _num_kv_heads_per_rank(num_heads, tp, attention_type)
    if tp > num_heads:
        setattr(hf_config, replicate_attr, tp // num_heads)
        num_heads = tp
        setattr(hf_config, heads_attr, num_heads)
    else:
        setattr(hf_config, replicate_attr, 1)
    return num_heads


def _finalize_mimo_cache_specs(model_config: ModelConfig, block_size: int, *, tp: int):
    """Materialize P1 Full block caches and the fixed-size SWA state ring."""
    hf_config = model_config.hf_config
    full_layers, swa_layers = _get_mimo_cache_layers(hf_config)
    full_heads = _num_kv_heads_per_rank(hf_config.num_key_value_heads, tp, "full-attention")
    swa_heads = _num_kv_heads_per_rank(hf_config.swa_num_key_value_heads, tp, "SWA")
    window_size = getattr(hf_config, "sliding_window_size", getattr(hf_config, "sliding_window", None))
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"MiMo-V2-Flash requires a positive integer SWA window size, but got {window_size!r}.")

    model_config.block_cache_specs = [
        BlockCacheSpec("mimo_full_k", full_layers, (block_size, full_heads, hf_config.head_dim), model_config.dtype),
        BlockCacheSpec("mimo_full_v", full_layers, (block_size, full_heads, hf_config.v_head_dim), model_config.dtype),
    ]
    state_specs = [
        StateCacheSpec(
            "mimo_swa_ring_k",
            (window_size, swa_heads, hf_config.swa_head_dim),
            model_config.dtype,
            layer_ids=swa_layers,
        ),
        StateCacheSpec(
            "mimo_swa_ring_v",
            (window_size, swa_heads, hf_config.swa_v_head_dim),
            model_config.dtype,
            layer_ids=swa_layers,
        ),
    ]
    model_config.state_cache_specs = state_specs
    # StateCacheSpec owns the named/layered physical allocation.  This legacy
    # bridge only tells Scheduler/Executor that the model has sequence state;
    # StateCacheEngine must not allocate a second anonymous copy.
    model_config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in state_specs]


def update_cache_config(cache_config):
    """Keep all logical blocks and align named-cache allocation granularity."""
    # The named cache shapes are finalized with ModelConfig.block_size. Keep
    # CacheEngine's kernel-block granularity identical so each physical block
    # has exactly the shape declared by the specs.
    cache_config.kernel_block_size = cache_config.block_size
    # SWA masking is handled by the model backend. Using DefaultBlockManager
    # preserves the full-attention history and keeps BlockTrie available.
    cache_config.window_size = -1


class MiMoV2FlashModelConfigBuilder(AutoModelConfigBuilder):
    """Build the Full-block + SWA-state configuration for MiMo-V2-Flash."""

    @classmethod
    def condition(cls, hf_config):
        """Match MiMo-V2-Flash Hugging Face configurations."""
        return hf_config.model_type == "mimo_v2_flash"

    @classmethod
    def build(
        cls,
        hf_config,
        model_path: str | None = None,
        tp: int = 1,
        is_draft_model: bool = False,
        spec_method: str | None = None,
        **kwargs,
    ):
        """Build the target model configuration and its hybrid cache specs."""
        _get_mimo_cache_layers(hf_config)
        if is_draft_model:
            raise ValueError("MiMo-V2-Flash draft-model support is not available yet.")
        if spec_method is not None:
            raise ValueError("MiMo-V2-Flash speculative decoding support is not available yet.")

        if getattr(hf_config, "routed_scaling_factor", None) is None:
            # The official eager implementation interprets null as 1.0;
            # LMDeploy's fused noaux_tc router expects a numeric multiplier.
            hf_config.routed_scaling_factor = 1.0
        num_key_value_heads = _update_kv_heads_for_tp(
            hf_config,
            heads_attr="num_key_value_heads",
            replicate_attr="num_replicate_key_value_heads",
            tp=tp,
            attention_type="full-attention",
        )
        _update_kv_heads_for_tp(
            hf_config,
            heads_attr="swa_num_key_value_heads",
            replicate_attr="swa_num_replicate_key_value_heads",
            tp=tp,
            attention_type="SWA",
        )

        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=getattr(hf_config, "bos_token_id", None),
            eos_token_id=getattr(hf_config, "eos_token_id", None),
            head_dim=hf_config.head_dim,
            k_head_dim=hf_config.head_dim,
            v_head_dim=hf_config.v_head_dim,
            sliding_window=-1,
            vocab_size=hf_config.vocab_size,
            model_paradigm="ar",
            use_standard_kv_cache=False,
        )
        config.block_cache_specs = []
        config.post_build_func = partial(_finalize_mimo_cache_specs, tp=tp)
        config.update_cache_config_func = update_cache_config
        return config
