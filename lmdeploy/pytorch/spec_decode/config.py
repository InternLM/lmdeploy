# Copyright (c) OpenMMLab. All rights reserved.
"""Resolve speculative model configuration and distributed capabilities."""
import copy
import os

from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.config import CacheConfig, DistConfig, SpecDecodeConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.transformers import config_from_pretrained
from lmdeploy.utils import get_model

_EAGLE3_DEEPSEEK_ARCH = 'Eagle3DeepseekV2ForCausalLM'


def build_specdecode_config(target_model,
                            speculative_config: SpeculativeConfig,
                            engine_config: PytorchEngineConfig,
                            cache_config: CacheConfig,
                            dist_config: DistConfig,
                            trust_remote_code: bool = False,
                            ):
    """Build spec decode config."""
    def _build_draft_dist_ctx(dist_config, draft_arch):
        # TODO support tp > 1, ep > 1 for other methods
        if speculative_config.method in ('deepseek_mtp', 'qwen3_5_mtp', 'hy3_mtp'):
            draft_dist_config = dist_config
        elif speculative_config.method in ('dflash', 'dspark'):
            if speculative_config.method == 'dflash':
                from lmdeploy.pytorch.spec_decode.dflash_utils import (
                    validate_dflash_dist_config as validate_dist_config,
                )
                from lmdeploy.pytorch.spec_decode.dflash_utils import (
                    validate_dflash_runtime_config as validate_runtime_config,
                )
            else:
                from lmdeploy.pytorch.spec_decode.dspark_utils import (
                    validate_dspark_dist_config as validate_dist_config,
                )
                from lmdeploy.pytorch.spec_decode.dspark_utils import (
                    validate_dspark_runtime_config as validate_runtime_config,
                )
            validate_dist_config(dist_config)
            validate_runtime_config(cache_config=cache_config, backend_config=engine_config)
            draft_dist_config = copy.deepcopy(dist_config)
        elif speculative_config.method == 'eagle3' and draft_arch == _EAGLE3_DEEPSEEK_ARCH:
            draft_dist_config = dist_config
        else:
            draft_dist_config = DistConfig()
        return draft_dist_config

    specdecode_config = None
    if speculative_config is not None:
        draft_model = speculative_config.model
        if draft_model and not os.path.exists(speculative_config.model):
            draft_model = get_model(draft_model, engine_config.download_dir, engine_config.revision)
        draft_arch = None
        if speculative_config.method == 'eagle3' and draft_model is not None:
            draft_hf_config = config_from_pretrained(
                draft_model, trust_remote_code=trust_remote_code)
            draft_architectures = getattr(draft_hf_config, 'architectures', None) or []
            if draft_architectures:
                draft_arch = draft_architectures[0]
        draft_dist_config = _build_draft_dist_ctx(dist_config, draft_arch)
        draft_model_format = (
            None if draft_arch == _EAGLE3_DEEPSEEK_ARCH else engine_config.model_format)

        specdecode_config = SpecDecodeConfig.from_config(
            method=speculative_config.method,
            num_speculative_tokens=speculative_config.num_speculative_tokens,
            model=draft_model,
            target_model=target_model,
            target_cache_cfg=cache_config,
            dtype=engine_config.dtype,
            trust_remote_code=trust_remote_code,
            model_format=draft_model_format,
            hf_overrides=engine_config.hf_overrides,
            dist_config=draft_dist_config,
        )
    if specdecode_config is not None and speculative_config.method in ('dflash', 'dspark'):
        if dist_config.dp > 1 or dist_config.ep > 1:
            if engine_config.enable_microbatch:
                raise ValueError('DFlash-family DP/EP does not support microbatch overlap.')
            if cache_config.kv_transfer_config is not None or cache_config.role != EngineRole.Hybrid:
                raise ValueError('DFlash-family DP/EP does not support KV transfer or PD.')
            arch = specdecode_config.model_config.hf_config.architectures[0]
            if arch not in ('DFlashDraftModel', 'Qwen3DSparkModel', 'DeepseekV4ForCausalLMDSpark'):
                raise ValueError(f'DFlash-family DP/EP draft architecture is not supported: {arch}')
    return specdecode_config
