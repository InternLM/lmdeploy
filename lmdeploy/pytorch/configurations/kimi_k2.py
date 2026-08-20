# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_v2 import DeepseekV2ModelConfigBuilder

_EAGLE3_DEEPSEEK_ARCH = 'Eagle3DeepseekV2ForCausalLM'


def _enable_kimi_fused_qkv_a_proj(hf_config) -> bool:
    """Return whether Kimi can merge its replicated MLA A projections."""
    dtype = str(getattr(hf_config, 'dtype', '')).removeprefix('torch.')
    return (getattr(hf_config, 'model_type', None) == 'kimi_k2'
            and getattr(hf_config, 'q_lora_rank', None) is not None
            and getattr(hf_config, 'hidden_size', None) == 7168
            and hf_config.q_lora_rank == 1536
            and getattr(hf_config, 'kv_lora_rank', None) == 512
            and getattr(hf_config, 'qk_rope_head_dim', None) == 64
            and dtype in {'bfloat16', 'float16'})


class KimiK2ModelConfigBuilder(DeepseekV2ModelConfigBuilder):
    """Build the PyTorch engine config for Kimi-K2 text models."""

    @classmethod
    def condition(cls, hf_config):
        """Match Kimi-K2 text configurations."""
        return getattr(hf_config, 'model_type', None) == 'kimi_k2'

    @classmethod
    def build(cls,
              hf_config,
              model_path: str = None,
              is_draft_model: bool = False,
              spec_method: str = None,
              **kwargs):
        """Build Kimi-K2 target and standalone EAGLE3 draft configs."""
        supported_spec_methods = {'deepseek_mtp', 'eagle3'}
        if spec_method is not None and spec_method not in supported_spec_methods:
            raise ValueError(
                f'Unsupported speculative method for Kimi-K2: {spec_method}')
        if is_draft_model and spec_method is None:
            raise ValueError(
                'A speculative method is required when building a Kimi-K2 draft model.'
            )

        hf_config.fuse_qkv_a_proj = _enable_kimi_fused_qkv_a_proj(hf_config)

        if spec_method != 'eagle3':
            return super().build(
                hf_config,
                model_path,
                is_draft_model=is_draft_model,
                spec_method=spec_method,
                **kwargs,
            )

        cfg = super().build(
            hf_config,
            model_path,
            is_draft_model=False,
            spec_method=None,
            **kwargs,
        )
        cfg.model_paradigm = 'ar_spec'

        if is_draft_model:
            architectures = getattr(hf_config, 'architectures', None) or []
            if not architectures or architectures[0] != _EAGLE3_DEEPSEEK_ARCH:
                raise ValueError(
                    'Kimi-K2 EAGLE3 draft config must use architecture '
                    f'{_EAGLE3_DEEPSEEK_ARCH}.')
            if hf_config.num_hidden_layers != 1:
                raise ValueError(
                    'Kimi-K2 EAGLE3 draft model must contain exactly one layer.'
                )
            cfg.num_layers = 1
            cfg.vocab_size = getattr(hf_config, 'draft_vocab_size',
                                     None) or hf_config.vocab_size
        else:
            if cfg.num_layers < 5:
                raise ValueError(
                    'Kimi-K2 EAGLE3 target model must contain at least five layers.'
                )
            hf_config.aux_hidden_state_layers = (2, cfg.num_layers // 2,
                                                 cfg.num_layers - 3)

        return cfg
