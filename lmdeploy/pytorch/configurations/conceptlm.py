# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import torch

from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.utils import get_logger

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder

logger = get_logger('lmdeploy')

CONCEPT_STATE_CHUNK_SOURCE = 0
CONCEPT_STATE_LAST = 1
CONCEPT_STATE_NAMES = (
    'concept_chunk_source_state',
    'concept_last_state',
)

_TRAINING_CONFIG_NAME = 'training_config.yaml'
_CONCEPT_TRAINING_CONFIG_KEYS = {
    'concept_chunk_size': 'conceptlm_chunk_size',
    'concept_shift_feature': 'conceptlm_shift_feature',
    'concept_chunk_merge_method': 'conceptlm_chunk_merge_method',
    'concept_layer_norm_option': 'conceptlm_layer_norm_option',
    'concept_fusion_norm_alpha_init': 'conceptlm_fusion_norm_alpha_init',
    'concept_fusion_alpha_init': 'conceptlm_fusion_alpha_init',
    'concept_hlm_ffn_hidden_size': 'conceptlm_hlm_ffn_hidden_size',
    'concept_hlm_attention_mode': 'conceptlm_hlm_attention_mode',
    'concept_v22_vq_codebook_size': 'conceptlm_v22_vq_codebook_size',
    'concept_v22_vq_num_codebooks': 'conceptlm_v22_vq_num_codebooks',
    'concept_v22_vq_commitment_cost': 'conceptlm_v22_vq_commitment_cost',
    'concept_v22_vq_merge_mode': 'conceptlm_v22_vq_merge_mode',
    'concept_v22_vq_hlm_loss_type': 'conceptlm_v22_vq_hlm_loss_type',
    'concept_v22_vq_detach_hlm_target': 'conceptlm_v22_vq_detach_hlm_target',
    'concept_dd_two_route_add': 'conceptlm_v21_dd_two_route_add',
    'concept_dd_two_route_add_concept_source': 'conceptlm_v21_dd_two_route_add_concept_source',
    'concept_dd_two_route_add_enable_raw_concept_route': (
        'conceptlm_v21_dd_two_route_add_enable_raw_concept_route'
    ),
    'concept_dd_two_route_add_enable_final_concept_route': (
        'conceptlm_v21_dd_two_route_add_enable_final_concept_route'
    ),
    'concept_dd_two_route_add_beta_init': 'conceptlm_v21_dd_two_route_add_beta_init',
    'concept_dd_two_route_add_every_n_layers': 'conceptlm_v21_dd_two_route_add_every_n_layers',
    'concept_dd_two_route_add_concept_route_first_n': 'conceptlm_v21_dd_two_route_add_concept_route_first_n',
    'concept_dd_two_route_add_decoder_hidden_size': 'conceptlm_v21_dd_two_route_add_decoder_hidden_size',
    'concept_dd_two_route_add_concept_hidden_size': 'conceptlm_v21_dd_two_route_add_concept_hidden_size',
    'concept_dd_two_route_add_use_softmax': 'conceptlm_v21_dd_two_route_add_use_softmax',
    'concept_dd_two_route_add_disable_decoder_dd': 'conceptlm_v21_dd_two_route_add_disable_decoder_dd',
    'concept_dd_two_route_add_decoder_use_layernorm': 'conceptlm_v21_dd_two_route_add_decoder_use_layernorm',
    'concept_dd_two_route_add_decoder_use_softmax': 'conceptlm_v21_dd_two_route_add_decoder_use_softmax',
    'concept_dd_two_route_add_concept_use_layernorm': 'conceptlm_v21_dd_two_route_add_concept_use_layernorm',
    'concept_dd_encoder_self_dd': 'conceptlm_v21_dd_encoder_self_dd',
    'concept_dd_encoder_self_dd_every_n_layers': 'conceptlm_v21_dd_encoder_self_dd_every_n_layers',
    'concept_dd_encoder_self_dd_hidden_size': 'conceptlm_v21_dd_encoder_self_dd_hidden_size',
    'concept_dd_encoder_self_dd_use_layernorm': 'conceptlm_v21_dd_encoder_self_dd_use_layernorm',
    'concept_dd_concept_self_dd': 'conceptlm_v21_dd_concept_self_dd',
    'concept_dd_concept_self_dd_every_n_layers': 'conceptlm_v21_dd_concept_self_dd_every_n_layers',
    'concept_dd_concept_self_dd_hidden_size': 'conceptlm_v21_dd_concept_self_dd_hidden_size',
    'concept_dd_concept_self_dd_use_layernorm': 'conceptlm_v21_dd_concept_self_dd_use_layernorm',
    'concept_enable_concept_read_encoder': 'conceptlm_v21_enable_concept_read_encoder',
    'concept_enable_decoder_read_encoder': 'conceptlm_v21_enable_decoder_read_encoder',
    'concept_enable_decoder_read_concept': 'conceptlm_v21_enable_decoder_read_concept',
    'concept_read_encoder_first_n': 'conceptlm_v21_concept_read_encoder_first_n',
    'concept_decoder_read_encoder_first_n': 'conceptlm_v21_decoder_read_encoder_first_n',
    'concept_residual_flow_beta_init': 'conceptlm_v21_residual_flow_beta_init',
    'concept_residual_flow_route_hidden_size': 'conceptlm_v21_residual_flow_route_hidden_size',
    'concept_residual_flow_route_use_softmax': 'conceptlm_v21_residual_flow_route_use_softmax',
    'concept_residual_flow_source_use_layernorm': 'conceptlm_v21_residual_flow_source_use_layernorm',
    'concept_residual_flow_shared_source_norm': 'conceptlm_v21_residual_flow_shared_source_norm',
    'concept_final_read_concept_gate': 'conceptlm_v21_final_read_concept_gate',
    'concept_final_read_concept_gate_init_final': 'conceptlm_v21_final_read_concept_gate_init_final',
    'concept_final_read_concept_gate_target_final': 'conceptlm_v21_final_read_concept_gate_target_final',
    'concept_final_read_concept_gate_reg_weight': 'conceptlm_v21_final_read_concept_gate_reg_weight',
    'concept_dd_self_dd_mode': 'conceptlm_v21_dd_self_dd_mode',
    'concept_enable_full_residual_flow': 'conceptlm_v21_enable_full_residual_flow',
    'concept_encoder_layers': 'conceptlm_encoder_layers',
    'concept_special_layers': 'conceptlm_special_layers',
    'concept_decoder_layers': 'conceptlm_decoder_layers',
}
_REQUIRED_CONCEPT_CONFIG_KEYS = (
    'concept_encoder_layers',
    'concept_special_layers',
    'concept_decoder_layers',
    'concept_v22_vq_num_codebooks',
    'concept_v22_vq_codebook_size',
)


def _get_concept_state_dtype(hf_config):
    """Return the dtype used by ConceptLM sequence-state caches."""
    torch_dtype = getattr(hf_config, 'torch_dtype', None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    torch_dtype = str(torch_dtype).lower()
    if 'bfloat16' in torch_dtype or 'bf16' in torch_dtype:
        return torch.bfloat16
    if 'float32' in torch_dtype or 'fp32' in torch_dtype:
        return torch.float32
    return torch.float16


def _training_config_value(value):
    """Unwrap wandb-style ``{desc, value}`` entries."""
    if isinstance(value, dict) and 'value' in value:
        return value['value']
    return value


def _load_training_config(model_path: str = None):
    """Load ConceptLM's exported training config when the HF config is
    sparse."""
    if not model_path:
        return None

    path = Path(model_path) / _TRAINING_CONFIG_NAME
    if not path.is_file():
        return None

    try:
        import yaml
        with path.open(encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning(f'ConceptLM: failed to load {path}: {e}')
        return None

    return data if isinstance(data, dict) else None


def _fill_concept_runtime_config(hf_config, model_path: str = None):
    """Fill old LMDeploy ConceptLM fields from the new exported training
    config.

    Newer ConceptLM HF exports keep only generic model metadata in
    ``config.json`` and leave ConceptLM runtime structure in
    ``training_config.yaml`` using the original training argument names.  The
    LMDeploy model uses the older ``concept_*`` names internally, so normalize
    the config once at the configuration boundary.
    """
    training_config = _load_training_config(model_path)
    if training_config is None:
        return

    for attr_name, training_name in _CONCEPT_TRAINING_CONFIG_KEYS.items():
        if hasattr(hf_config, attr_name) or training_name not in training_config:
            continue
        setattr(hf_config, attr_name, _training_config_value(training_config[training_name]))


def _check_concept_runtime_config(hf_config):
    """Raise a readable error for sparse ConceptLM exports."""
    missing = [name for name in _REQUIRED_CONCEPT_CONFIG_KEYS if not hasattr(hf_config, name)]
    if missing:
        raise AttributeError(
            'ConceptLM config is missing runtime fields '
            f'{missing}. If this is a new HF export, keep {_TRAINING_CONFIG_NAME} beside config.json '
            'so LMDeploy can backfill the ConceptLM architecture fields.')


def _normalize_concept_rotary_config(hf_config):
    """Normalize ConceptLM rotary fields for the shared rotary builder."""
    head_dim = getattr(hf_config, 'head_dim', None)
    if head_dim is None:
        head_dim = getattr(hf_config, 'kv_channels', None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    hf_config.head_dim = int(head_dim)

    if not hasattr(hf_config, 'rope_theta') and hasattr(hf_config, 'rotary_base'):
        hf_config.rope_theta = hf_config.rotary_base

    rotary_percent = getattr(hf_config, 'rotary_percent', None)
    if rotary_percent is not None and not hasattr(hf_config, 'partial_rotary_factor'):
        rotary_dim = int(hf_config.head_dim * float(rotary_percent))
        rotary_dim -= rotary_dim % 2
        if rotary_dim <= 0:
            raise ValueError(
                f'Invalid ConceptLM rotary dimension: head_dim={hf_config.head_dim}, rotary_percent={rotary_percent}')
        hf_config.partial_rotary_factor = rotary_dim / hf_config.head_dim

    position_embedding_type = getattr(hf_config, 'position_embedding_type', 'rope')
    if position_embedding_type != 'yarn':
        return

    scaling_factor = getattr(hf_config, 'yarn_rotary_scaling_factor', 1.0)
    if scaling_factor is None:
        scaling_factor = 1.0
    rope_scaling = {
        'rope_type': 'yarn',
        'rope_theta': getattr(hf_config, 'rope_theta', getattr(hf_config, 'rotary_base', 10000)),
        'factor': float(scaling_factor),
        'beta_fast': getattr(hf_config, 'yarn_beta_fast', 32.0),
        'beta_slow': getattr(hf_config, 'yarn_beta_slow', 1.0),
        'mscale': getattr(hf_config, 'yarn_mscale', 1.0),
        'mscale_all_dim': getattr(hf_config, 'yarn_mscale_all_dim', 0.0),
        'truncate': getattr(hf_config, 'yarn_correction_range_round_to_int', True),
    }
    original_max_position_embeddings = getattr(hf_config, 'yarn_original_max_position_embeddings',
                                               getattr(hf_config, 'max_position_embeddings', None))
    if original_max_position_embeddings is not None:
        rope_scaling['original_max_position_embeddings'] = original_max_position_embeddings
    if getattr(hf_config, 'rope_parameters', None) is None:
        hf_config.rope_parameters = dict(rope_scaling)
    if getattr(hf_config, 'rope_scaling', None) is None:
        hf_config.rope_scaling = dict(rope_scaling)


class ConceptLMModelConfigBuilder(AutoModelConfigBuilder):
    """Config builder for ConceptLM V2.2-VQ.

    The upstream checkpoint config does not declare bos/eos/pad token ids, so derive them from the tokenizer when
    missing before handing the config to the default builder (which asserts they exist).
    """

    @classmethod
    def condition(cls, hf_config):
        """config."""
        archs = getattr(hf_config, 'architectures', None) or []
        return 'ConceptLMV22VQForCausalLM' in archs

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        _fill_concept_runtime_config(hf_config, model_path)
        _check_concept_runtime_config(hf_config)
        _normalize_concept_rotary_config(hf_config)

        # fill missing special token ids from the tokenizer
        if getattr(hf_config, 'bos_token_id', None) is None:
            cls._fill_special_tokens(hf_config, model_path)

        model_config = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)

        enc_layers = int(hf_config.concept_encoder_layers)
        concept_layers = int(hf_config.concept_special_layers)
        dec_layers = int(hf_config.concept_decoder_layers)
        model_config.num_layers = enc_layers + concept_layers + dec_layers

        # TODO: ConceptLM's concept predictor attends over a compressed
        # chunk-level timeline, so these concept KV layers do not need full
        # token-length block capacity. Keep the standard KV cache layout for
        # now to avoid adding another cache-engine abstraction; optimize this
        # later if concept KV memory becomes a real bottleneck.
        model_config.llm_config.concept_kv_encoder_offset = 0
        model_config.llm_config.concept_kv_concept_offset = enc_layers
        model_config.llm_config.concept_kv_decoder_offset = enc_layers + concept_layers
        model_config.llm_config.concept_kv_total_layers = model_config.num_layers

        hidden_size = int(hf_config.hidden_size)
        last_state_dtype = _get_concept_state_dtype(hf_config)
        concept_encoder_read_sources = max(enc_layers - 1, 0)
        # Decode accumulates the current chunk for every state needed when a
        # chunk boundary emits one concept. Keep this accumulator in fp32 to
        # match the reference chunk merge: reduce the whole chunk, then cast the
        # emitted concept input back to model dtype. Using bf16 here would round
        # the partial sum after every decode token and drift from full-forward
        # semantics.
        #
        # Row 0 is the final encoder hidden used as concept-predictor input;
        # following rows are encoder raw states consumed by concept-read-encoder
        # residual routes.
        concept_chunk_state_sources = 1 + concept_encoder_read_sources
        # Last emitted concept snapshot. Row 0 is the final concept vector,
        # rows 1: are raw concept-layer states. Keep this packed so decode can
        # gather the visible last-concept state once per batch row.
        concept_last_state_sources = 1 + concept_layers
        state_specs = [
            StateCacheSpec(CONCEPT_STATE_NAMES[CONCEPT_STATE_CHUNK_SOURCE],
                           (concept_chunk_state_sources, hidden_size), torch.float32),
            StateCacheSpec(CONCEPT_STATE_NAMES[CONCEPT_STATE_LAST], (concept_last_state_sources, hidden_size),
                           last_state_dtype),
        ]
        model_config.state_cache_specs = state_specs
        # Backward-compat bridge used by scheduler/state-cache sizing. The
        # actual runtime access should use state_cache_specs/named_state_caches
        # like DSV4, not anonymous order-dependent indices.
        model_config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in state_specs]
        model_config.llm_config.concept_state_names = CONCEPT_STATE_NAMES
        model_config.llm_config.concept_state_chunk_source_idx = CONCEPT_STATE_CHUNK_SOURCE
        model_config.llm_config.concept_state_last_idx = CONCEPT_STATE_LAST
        return model_config

    @staticmethod
    def _fill_special_tokens(hf_config, model_path: str = None):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            hf_config.bos_token_id = tok.bos_token_id
            hf_config.eos_token_id = tok.eos_token_id
            if getattr(hf_config, 'pad_token_id', None) is None:
                hf_config.pad_token_id = tok.pad_token_id
        except Exception as e:  # noqa: BLE001
            logger.warning(f'ConceptLM: failed to derive special token ids: {e}')
