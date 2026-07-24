# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder

logger = get_logger('lmdeploy')

CONCEPT_STATE_CHUNK_SOURCE = 0
CONCEPT_STATE_LAST_RAW = 1
CONCEPT_STATE_LAST_FINAL = 2
CONCEPT_STATE_NAMES = (
    'concept_chunk_source_state',
    'concept_last_raw_states',
    'concept_last_final_state',
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


class ConceptLMModelConfigBuilder(AutoModelConfigBuilder):
    """Config builder for ConceptLM V2.2-VQ.

    The upstream checkpoint config does not declare bos/eos/pad token ids,
    so derive them from the tokenizer when missing before handing the config
    to the default builder (which asserts they exist).
    """

    @classmethod
    def condition(cls, hf_config):
        """config."""
        archs = getattr(hf_config, 'architectures', None) or []
        return 'ConceptLMV22VQForCausalLM' in archs

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
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
        state_dtype = _get_concept_state_dtype(hf_config)
        concept_encoder_read_sources = max(enc_layers - 1, 0)
        # Decode accumulates the current chunk for every state needed when a
        # chunk boundary emits one concept. Row 0 is the final encoder hidden
        # used as concept-predictor input; following rows are encoder raw states
        # consumed by concept-read-encoder residual routes.
        concept_chunk_state_sources = 1 + concept_encoder_read_sources
        model_config.states_shapes = [
            ((concept_chunk_state_sources, hidden_size), state_dtype),
            ((concept_layers, hidden_size), state_dtype),
            ((hidden_size, ), state_dtype),
        ]
        # The current branch only supports anonymous states_shapes. Keep stable
        # indices on the HF config so model code has one semantic source of
        # truth; if DSV4 StateCacheSpec lands here later, these become the
        # names of ConceptLM's state-cache specs.
        model_config.llm_config.concept_state_names = CONCEPT_STATE_NAMES
        model_config.llm_config.concept_state_chunk_source_idx = CONCEPT_STATE_CHUNK_SOURCE
        model_config.llm_config.concept_state_last_raw_idx = CONCEPT_STATE_LAST_RAW
        model_config.llm_config.concept_state_last_final_idx = CONCEPT_STATE_LAST_FINAL
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
