# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import ModelConfig, StateCacheSpec

from .builder import AutoModelConfigBuilder


def _get_mimo_cache_layers(hf_config):
    """Validate the hybrid pattern and return full/SWA layer partitions."""
    num_layers = hf_config.num_hidden_layers
    pattern = list(hf_config.hybrid_layer_pattern)
    if len(pattern) != num_layers:
        raise ValueError(
            'MiMo-V2-Flash hybrid_layer_pattern must contain one entry per layer, '
            f'but got {len(pattern)} entries for num_hidden_layers={num_layers}.'
        )

    invalid = sorted(set(pattern) - {0, 1})
    if invalid:
        raise ValueError(
            f'MiMo-V2-Flash hybrid_layer_pattern only supports 0 (full attention) and 1 (SWA), but got {invalid}.'
        )

    full_layers = [i for i, layer_type in enumerate(pattern) if layer_type == 0]
    swa_layers = [i for i, layer_type in enumerate(pattern) if layer_type == 1]
    if not full_layers or not swa_layers:
        raise ValueError('MiMo-V2-Flash requires both full-attention and SWA layers.')
    return full_layers, swa_layers


def _num_kv_heads_per_rank(num_heads: int, tp: int, attention_type: str) -> int:
    """Return local KV heads, including replication when TP exceeds KV
    heads."""
    if num_heads >= tp:
        if num_heads % tp != 0:
            raise ValueError(f'MiMo-V2-Flash {attention_type} KV heads ({num_heads}) must be divisible by TP ({tp}).')
        return num_heads // tp
    if tp % num_heads != 0:
        raise ValueError(f'MiMo-V2-Flash TP ({tp}) must be divisible by {attention_type} KV heads ({num_heads}).')
    return 1


def _update_kv_heads_for_tp(hf_config, *, heads_attr: str, replicate_attr: str, tp: int, attention_type: str) -> int:
    """Expand KV heads for runtime replication while retaining idempotence."""
    current_heads = getattr(hf_config, heads_attr)
    current_replicas = getattr(hf_config, replicate_attr, 1)
    if current_replicas < 1 or current_heads % current_replicas:
        raise ValueError(
            f'Invalid {attention_type} KV replication metadata: '
            f'heads={current_heads}, replicas={current_replicas}.'
        )
    original_heads = current_heads // current_replicas
    _num_kv_heads_per_rank(original_heads, tp, attention_type)
    replicas = tp // original_heads if tp > original_heads else 1
    runtime_heads = original_heads * replicas
    # Always overwrite both runtime fields. This keeps repeated config builds
    # idempotent if the same HF config object is reused with a different TP.
    setattr(hf_config, heads_attr, runtime_heads)
    setattr(hf_config, replicate_attr, replicas)
    return runtime_heads


def _update_mimo_cache_config(cache_config):
    """Keep all logical blocks and align named-cache allocation granularity."""
    # Operator cache requests use the finalized kernel-block granularity.
    cache_config.kernel_block_size = cache_config.block_size
    # SWA masking is handled by the model backend. Using DefaultBlockManager
    # preserves the full-attention history and keeps BlockTrie available.
    cache_config.window_size = -1


class MiMoV2FlashModelConfigBuilder(AutoModelConfigBuilder):
    """Build MiMo-V2-Flash target or MTP configuration and cache policy."""

    @classmethod
    def condition(cls, hf_config):
        """Match MiMo-V2-Flash Hugging Face configurations."""
        return hf_config.model_type == 'mimo_v2_flash'

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
        """Build the target or three-depth draft model configuration."""
        _, swa_layers = _get_mimo_cache_layers(hf_config)
        supported_spec_methods = {'mimo_mtp'}
        if spec_method is not None and spec_method not in supported_spec_methods:
            raise ValueError(
                f'MiMo-V2-Flash only supports speculative method mimo_mtp, got {spec_method!r}.'
            )

        if is_draft_model:
            # The official config does not declare the MTP depth, while the
            # checkpoint index contains model.mtp.layers.{0,1,2}. Keep this
            # explicit until depth discovery is moved into the weight-index
            # loader; silently falling back to one layer would load an
            # incomplete draft model.
            num_mtp_layers = getattr(hf_config, 'num_nextn_predict_layers', 3)
            if num_mtp_layers != 3:
                raise ValueError(f'MiMo-V2-Flash requires 3 MTP prediction-depth layers, got {num_mtp_layers}.')
            hf_config.num_nextn_predict_layers = num_mtp_layers
            hf_config.architectures = ['MiMoV2FlashMTPModel']
            # Remote-code AutoModelForCausalLM still points at the target
            # model and takes precedence over architectures in the patcher.
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map

            num_key_value_heads = _update_kv_heads_for_tp(
                hf_config,
                heads_attr='swa_num_key_value_heads',
                replicate_attr='swa_num_replicate_key_value_heads',
                tp=tp,
                attention_type='MTP SWA',
            )
            window_size = getattr(hf_config, 'sliding_window_size', getattr(hf_config, 'sliding_window', None))
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(f'MiMo-V2-Flash MTP requires a positive integer SWA window size, got {window_size!r}.')

            # MTP owns a separate, rollback-safe paged KV cache. Do not attach
            # the target model's Full block caches or in-place SWA state ring:
            # speculative rejection cannot restore overwritten ring slots.
            return ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=num_mtp_layers,
                num_attention_heads=hf_config.swa_num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                bos_token_id=getattr(hf_config, 'bos_token_id', None),
                eos_token_id=getattr(hf_config, 'eos_token_id', None),
                head_dim=hf_config.swa_head_dim,
                k_head_dim=hf_config.swa_head_dim,
                v_head_dim=hf_config.swa_v_head_dim,
                sliding_window=window_size,
                vocab_size=hf_config.vocab_size,
                model_paradigm='ar_spec',
                use_standard_kv_cache=True,
            )

        if getattr(hf_config, 'routed_scaling_factor', None) is None:
            # The official eager implementation interprets null as 1.0;
            # LMDeploy's fused noaux_tc router expects a numeric multiplier.
            hf_config.routed_scaling_factor = 1.0
        # Target verification writes tentative candidates and may reject a
        # suffix.  A fixed in-place ring cannot restore history slots that a
        # candidate overwrote across the modulo boundary, whereas paged KV is
        # reclaimed by the normal speculative rollback path.
        hf_config._lmdeploy_use_paged_swa = spec_method in supported_spec_methods
        num_key_value_heads = _update_kv_heads_for_tp(
            hf_config,
            heads_attr='num_key_value_heads',
            replicate_attr='num_replicate_key_value_heads',
            tp=tp,
            attention_type='full-attention',
        )
        _update_kv_heads_for_tp(
            hf_config,
            heads_attr='swa_num_key_value_heads',
            replicate_attr='swa_num_replicate_key_value_heads',
            tp=tp,
            attention_type='SWA',
        )

        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=getattr(hf_config, 'bos_token_id', None),
            eos_token_id=getattr(hf_config, 'eos_token_id', None),
            head_dim=hf_config.head_dim,
            k_head_dim=hf_config.head_dim,
            v_head_dim=hf_config.v_head_dim,
            sliding_window=-1,
            vocab_size=hf_config.vocab_size,
            model_paradigm='ar_spec' if spec_method is not None else 'ar',
            use_standard_kv_cache=False,
        )
        if not hf_config._lmdeploy_use_paged_swa:
            window_size = getattr(hf_config, 'sliding_window_size', getattr(hf_config, 'sliding_window', None))
            swa_heads = _num_kv_heads_per_rank(hf_config.swa_num_key_value_heads, tp, 'SWA')
            state_specs = [
                StateCacheSpec(
                    'mimo_swa_ring_k',
                    (window_size, swa_heads, hf_config.swa_head_dim),
                    torch.bfloat16,
                    layer_ids=swa_layers,
                ),
                StateCacheSpec(
                    'mimo_swa_ring_v',
                    (window_size, swa_heads, hf_config.swa_v_head_dim),
                    torch.bfloat16,
                    layer_ids=swa_layers,
                ),
            ]
            config.state_cache_specs = state_specs
            # Keep scheduler state-slot accounting until it consumes named
            # StateCacheSpec directly.
            config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in state_specs]
        config.update_cache_config_func = _update_mimo_cache_config
        return config
