import argparse
import inspect
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.cli.utils import ArgumentHelper, get_speculative_config
from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.model_inputs import BuildModelContext, ModelInputs
from lmdeploy.pytorch.models.module_map import MODULE_MAP
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.qwen3 import Qwen3ForCausalLM
from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model
from lmdeploy.pytorch.models.qwen3_dflash import (
    DFlashDraftModel,
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _normalize_dflash_weight_name,
    _resolve_dflash_layer_attention,
)
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin
from lmdeploy.pytorch.spec_decode.dflash_utils import (
    build_target_layer_ids,
    parse_dflash_config,
    validate_dflash_cache_config,
    validate_dflash_dist_config,
    validate_dflash_runtime_config,
)
from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash


def _draft_config(**kwargs):
    values = dict(
        architectures=['DFlashDraftModel'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=128,
        model_type='qwen3',
        num_attention_heads=4,
        num_hidden_layers=4,
        num_target_layers=16,
        num_key_value_heads=2,
        vocab_size=32000,
        dflash_config=dict(
            block_size=4,
            mask_token_id=32001,
            target_layer_ids=[1, 5, 9, 13],
        ),
        layer_types=['full_attention'] * 4,
        use_sliding_window=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _parse_dflash(draft_config, num_speculative_tokens, target_num_layers=None):
    if target_num_layers is None:
        target_num_layers = draft_config.num_target_layers
    return parse_dflash_config(draft_config,
                               num_speculative_tokens=num_speculative_tokens,
                               target_num_layers=target_num_layers)


def test_parse_dflash_config_valid():
    target_layer_ids, mask_token_id = _parse_dflash(_draft_config(), num_speculative_tokens=3)

    assert target_layer_ids == (1, 5, 9, 13)
    assert mask_token_id == 32001


def test_specdecode_config_stores_resolved_dflash_fields_directly():
    target_layer_ids, mask_token_id = _parse_dflash(_draft_config(), num_speculative_tokens=3)
    cfg = SpecDecodeConfig(model='draft-model',
                           method='dflash',
                           num_speculative_tokens=3,
                           target_layer_ids=target_layer_ids,
                           mask_token_id=mask_token_id)

    assert cfg.target_layer_ids == (1, 5, 9, 13)
    assert cfg.mask_token_id == 32001
    assert not hasattr(cfg, 'dflash_config')


def test_dflash_block_size_overrides_draft_token_count():
    cfg = SpeculativeConfig(method='dflash', num_speculative_tokens=15, dflash_block_size=8)

    assert cfg.dflash_block_size == 8
    assert cfg.num_speculative_tokens == 7


@pytest.mark.parametrize('block_size', [True, 1, 0, -1])
def test_dflash_block_size_rejects_invalid_values(block_size):
    with pytest.raises(ValueError, match='integer greater than or equal to 2'):
        SpeculativeConfig(method='dflash', dflash_block_size=block_size)


def test_dflash_block_size_rejects_invalid_method():
    with pytest.raises(ValueError, match='only when method="dflash"'):
        SpeculativeConfig(method='eagle3', dflash_block_size=8)


def test_dflash_block_size_cli_override_and_algorithm_validation():
    parser = argparse.ArgumentParser()
    ArgumentHelper.add_spec_group(parser)
    args = parser.parse_args([
        '--speculative-algorithm',
        'dflash',
        '--speculative-num-draft-tokens',
        '15',
        '--speculative-dflash-block-size',
        '8',
    ])
    cfg = get_speculative_config(args)
    assert cfg.dflash_block_size == 8
    assert cfg.num_speculative_tokens == 7

    args = parser.parse_args(['--speculative-dflash-block-size', '8'])
    with pytest.raises(ValueError, match='requires --speculative-algorithm dflash'):
        get_speculative_config(args)


def test_specdecode_config_checks_matching_dflash_target_depth(monkeypatch):
    draft_model_config = SimpleNamespace(hf_config=_draft_config(num_target_layers=16))
    target_model_config = SimpleNamespace(num_layers=4, llm_config=SimpleNamespace(num_hidden_layers=16))
    calls = []

    def from_pretrained(model, **kwargs):
        calls.append((model, kwargs.get('is_draft_model')))
        return draft_model_config if model == 'draft-model' else target_model_config

    monkeypatch.setattr(ModelConfig, 'from_pretrained', from_pretrained)
    cfg = SpecDecodeConfig.from_config(method='dflash',
                                       num_speculative_tokens=3,
                                       model='draft-model',
                                       target_model='target-model',
                                       target_cache_cfg=CacheConfig(max_batches=1,
                                                                   block_size=64,
                                                                   num_cpu_blocks=0,
                                                                   num_gpu_blocks=1))

    assert calls == [('draft-model', True), ('target-model', False)]
    assert cfg.target_layer_ids == (1, 5, 9, 13)


def test_specdecode_config_rejects_mismatched_dflash_target_depth(monkeypatch):
    draft_model_config = SimpleNamespace(hf_config=_draft_config(num_target_layers=16))
    target_model_config = SimpleNamespace(num_layers=4, llm_config=SimpleNamespace(num_hidden_layers=15))
    monkeypatch.setattr(ModelConfig, 'from_pretrained',
                        lambda model, **kwargs: draft_model_config if model == 'draft-model' else target_model_config)

    with pytest.raises(ValueError, match='draft/target depth mismatch.*16.*15'):
        SpecDecodeConfig.from_config(method='dflash',
                                     num_speculative_tokens=3,
                                     model='draft-model',
                                     target_model='target-model',
                                     target_cache_cfg=CacheConfig(max_batches=1,
                                                                 block_size=64,
                                                                 num_cpu_blocks=0,
                                                                 num_gpu_blocks=1))


def test_parse_dflash_config_requires_mask_token_id():
    draft_config = _draft_config(dflash_config=dict(block_size=4, target_layer_ids=[1, 5]))

    with pytest.raises(ValueError, match='mask_token_id'):
        _parse_dflash(draft_config, num_speculative_tokens=3)


@pytest.mark.parametrize('num_speculative_tokens', [3, 7, 15])
def test_parse_dflash_config_allows_runtime_query_up_to_checkpoint_block_size(num_speculative_tokens):
    draft_config = _draft_config(dflash_config=dict(block_size=16, mask_token_id=32001))

    target_layer_ids, mask_token_id = _parse_dflash(draft_config,
                                                    num_speculative_tokens=num_speculative_tokens)

    assert target_layer_ids == build_target_layer_ids(16, 4)
    assert mask_token_id == 32001


def test_parse_dflash_config_rejects_query_above_checkpoint_block_size():
    draft_config = _draft_config(dflash_config=dict(block_size=16, mask_token_id=32001))

    with pytest.raises(ValueError, match='must not exceed.*block_size'):
        _parse_dflash(draft_config, num_speculative_tokens=16)


def test_parse_dflash_config_resolves_target_layers_from_num_target_layers():
    draft_config = _draft_config(dflash_config=dict(block_size=4, mask_token_id=32001))

    target_layer_ids, mask_token_id = _parse_dflash(draft_config, num_speculative_tokens=3)

    assert target_layer_ids == build_target_layer_ids(16, 4)
    assert mask_token_id == 32001


def test_parse_dflash_config_rejects_non_increasing_target_layers():
    draft_config = _draft_config(
        dflash_config=dict(
            block_size=4,
            mask_token_id=32001,
            target_layer_ids=[1, 9, 5, 13],
        ))

    with pytest.raises(ValueError, match='strictly increasing'):
        _parse_dflash(draft_config, num_speculative_tokens=3)


def test_parse_dflash_config_rejects_duplicate_target_layers():
    draft_config = _draft_config(
        dflash_config=dict(
            block_size=4,
            mask_token_id=32001,
            target_layer_ids=[1, 5, 5, 13],
        ))

    with pytest.raises(ValueError, match='duplicate-free'):
        _parse_dflash(draft_config, num_speculative_tokens=3)


def test_parse_dflash_config_rejects_out_of_range_target_layers():
    draft_config = _draft_config(
        num_target_layers=8,
        dflash_config=dict(
            block_size=4,
            mask_token_id=32001,
            target_layer_ids=[1, 5, 8],
        ))

    with pytest.raises(ValueError, match='out-of-range'):
        _parse_dflash(draft_config, num_speculative_tokens=3)


def test_parse_dflash_config_rejects_mismatched_target_depth():
    draft_config = _draft_config(num_target_layers=40)

    with pytest.raises(ValueError, match='draft/target depth mismatch.*40.*39'):
        _parse_dflash(draft_config, num_speculative_tokens=3, target_num_layers=39)


def test_parse_dflash_config_matches_real_checkpoint_schema():
    draft_config = _draft_config(
        num_hidden_layers=6,
        num_target_layers=40,
        dflash_config={
            'block_size': 16,
            'mask_token_id': 248077,
            'target_layer_ids': [1, 6, 11, 16, 22, 27, 32, 37],
        },
        layer_types=['sliding_attention'] * 5 + ['full_attention'],
        use_sliding_window=True,
        sliding_window=4096,
    )

    target_layer_ids, mask_token_id = _parse_dflash(draft_config, num_speculative_tokens=15)

    assert target_layer_ids == (1, 6, 11, 16, 22, 27, 32, 37)
    assert mask_token_id == 248077


def test_validate_dflash_allows_hybrid_sliding_attention():
    draft_config = _draft_config(
        layer_types=['sliding_attention', 'sliding_attention', 'full_attention', 'full_attention'],
        sliding_window=4096,
        use_sliding_window=True,
    )
    target_layer_ids, mask_token_id = _parse_dflash(draft_config, num_speculative_tokens=3)

    assert target_layer_ids == (1, 5, 9, 13)
    assert mask_token_id == 32001


def test_validate_dflash_rejects_unknown_layer_type():
    draft_config = _draft_config(layer_types=['linear_attention'] * 4)

    with pytest.raises(ValueError, match='full_attention and sliding_attention'):
        _parse_dflash(draft_config, num_speculative_tokens=3)


def test_parse_dflash_config_rejects_invalid_schema_before_model_construction():
    with pytest.raises(ValueError, match='dflash_config dictionary'):
        _parse_dflash(_draft_config(dflash_config=SimpleNamespace()), num_speculative_tokens=3)

    with pytest.raises(ValueError, match='layer_types length'):
        _parse_dflash(_draft_config(layer_types=['full_attention']), num_speculative_tokens=3)

    with pytest.raises(ValueError, match='layer_types must contain only strings'):
        _parse_dflash(_draft_config(layer_types=['full_attention', 1, 'full_attention', 'full_attention']),
                      num_speculative_tokens=3)

    sink_config = _draft_config(
        dflash_config=dict(block_size=4, mask_token_id=32001, attention_sink_bias=True))
    with pytest.raises(ValueError, match='attention-sink bias'):
        _parse_dflash(sink_config, num_speculative_tokens=3)


def test_parse_dflash_config_rejects_non_default_scheduler_patterns():
    forced_swa = _draft_config(
        dflash_config=dict(block_size=4,
                           mask_token_id=32001,
                           use_swa=True,
                           swa_window_size=2048),
        sliding_window=2048,
        use_sliding_window=True,
    )
    with pytest.raises(ValueError, match='does not support use_swa forcing'):
        _parse_dflash(forced_swa, num_speculative_tokens=3)

    alternate_window = _draft_config(
        dflash_config=dict(block_size=4, mask_token_id=32001, swa_window_size=2048),
        layer_types=['sliding_attention'] * 4,
        sliding_window=4096,
        use_sliding_window=True,
    )
    with pytest.raises(ValueError, match='SWA window.*ModelConfig.sliding_window'):
        _parse_dflash(alternate_window, num_speculative_tokens=3)

    noncausal_swa = _draft_config(
        dflash_config=dict(block_size=4, mask_token_id=32001, causal=False),
        layer_types=['sliding_attention'] * 4,
        sliding_window=4096,
        use_sliding_window=True,
    )
    with pytest.raises(ValueError, match='unsupported pattern=.*4096, False'):
        _parse_dflash(noncausal_swa, num_speculative_tokens=3)


def test_validate_dflash_dist_rejects_dp_and_ep_allows_tp():
    validate_dflash_dist_config(DistConfig(tp=2))

    with pytest.raises(ValueError, match='dp=1'):
        validate_dflash_dist_config(DistConfig(dp=2, tp=2))

    with pytest.raises(ValueError, match='ep=1'):
        validate_dflash_dist_config(DistConfig(ep=2))


def test_validate_dflash_cache_rejects_prefix_cache_and_kv_quant():
    validate_dflash_cache_config(SimpleNamespace(enable_prefix_caching=False, quant_policy=0))

    with pytest.raises(ValueError, match='prefix-cache'):
        validate_dflash_cache_config(SimpleNamespace(enable_prefix_caching=True, quant_policy=0))

    with pytest.raises(ValueError, match='KV-cache quantization'):
        validate_dflash_cache_config(SimpleNamespace(enable_prefix_caching=False, quant_policy=8))


def test_validate_dflash_runtime_rejects_non_cuda_and_allows_cudagraph():
    validate_dflash_runtime_config(cache_config=SimpleNamespace(device_type='cuda'),
                                   backend_config=SimpleNamespace(device_type='cuda', eager_mode=True))
    validate_dflash_runtime_config(cache_config=SimpleNamespace(device_type='cuda'),
                                   backend_config=SimpleNamespace(device_type='cuda', eager_mode=False))

    with pytest.raises(ValueError, match='requires CUDA'):
        validate_dflash_runtime_config(cache_config=SimpleNamespace(device_type='ascend'))


@pytest.mark.parametrize('model_cls', [Qwen3ForCausalLM, Qwen3_5ForConditionalGeneration])
def test_qwen_cudagraph_outputs_preserve_only_target_aux_hidden_states(model_cls):
    output_buffers = {
        'hidden_states': torch.arange(24).view(1, 6, 4),
        'aux_hidden_states': torch.arange(60).view(1, 6, 10),
        'target_inputs_embeds': torch.arange(30).view(1, 6, 5),
        'all_routed_experts': torch.arange(36).view(6, 2, 3),
    }
    input_ids = torch.zeros((1, 3), dtype=torch.long)

    model = object.__new__(model_cls)
    outputs = model.get_outputs_cudagraph(output_buffers, input_ids)

    assert outputs['hidden_states'].shape == (1, 3, 4)
    assert outputs['aux_hidden_states'].shape == (1, 3, 10)
    assert 'target_inputs_embeds' not in outputs
    assert outputs['aux_hidden_states'].data_ptr() == output_buffers['aux_hidden_states'].data_ptr()
    torch.testing.assert_close(outputs['aux_hidden_states'], output_buffers['aux_hidden_states'][:, :3])
    torch.testing.assert_close(outputs['all_routed_experts'], output_buffers['all_routed_experts'][:3])


def test_dflash_draft_graph_inputs_exclude_target_hidden_and_target_embeds():
    assert 'target_hidden_states' not in inspect.getsource(DFlashDraftModel.make_buffers_cudagraph)
    assert 'target_hidden_states' not in inspect.getsource(DFlashDraftModel.fill_buffers_cudagraph)

    model = object.__new__(DFlashDraftModel)
    explicit_embeds = torch.randn(1, 2, 4)
    context = SimpleNamespace(
        input_ids=torch.tensor([[1, 2]]),
        position_ids=torch.tensor([[0, 1]]),
        attn_metadata=object(),
        target_inputs_embeds=torch.full((1, 2, 4), 42.0),
    )
    outputs = model.prepare_inputs_for_generation([], inputs_embeds=explicit_embeds, context=context)

    assert outputs['inputs_embeds'] is explicit_embeds
    assert 'target_hidden_states' not in outputs


def test_dflash_graph_scheduler_metadata_reuses_generic_swa_and_separates_full(monkeypatch):
    model = object.__new__(DFlashDraftModel)
    torch.nn.Module.__init__(model)

    generic = torch.tensor([1, 2, 3], dtype=torch.int32)
    monkeypatch.setattr(CudaGraphMixin, 'make_buffers_cudagraph',
                        lambda self, graph_meta, **kwargs: {
                            'kv_seqlens': torch.zeros(2, dtype=torch.int32),
                            'scheduler_metadata': generic.clone(),
                        })
    patterns = []
    lengths = iter((4, 4, 2, 3))

    def build_fa3_scheduler_metadata(*args, sliding_window, causal, **kwargs):
        patterns.append((sliding_window, causal))
        return torch.arange(next(lengths), dtype=torch.int32)

    monkeypatch.setattr(model, 'build_fa3_scheduler_metadata', build_fa3_scheduler_metadata)
    graph_meta1 = SimpleNamespace(use_fa3_decoding=True,
                                  num_blocks=2,
                                  block_size=64,
                                  max_batchs=2,
                                  decode_query_len=16)
    graph_meta2 = SimpleNamespace(**vars(graph_meta1))

    buffers1 = model.make_buffers_cudagraph(graph_meta1)
    buffers2 = model.make_buffers_cudagraph(graph_meta2)
    graph_meta1.input_buffers = buffers1
    graph_meta2.input_buffers = buffers2
    assert buffers1['scheduler_metadata'].data_ptr() != buffers1['dflash_full_scheduler_metadata'].data_ptr()
    assert buffers1['dflash_full_scheduler_metadata'].data_ptr() != buffers2[
        'dflash_full_scheduler_metadata'].data_ptr()

    attn_metadata = TritonAttentionMetadata(is_decoding=True,
                                            block_offsets=torch.zeros(2, 1, dtype=torch.int32),
                                            scheduler_metadata=buffers1['scheduler_metadata'])
    monkeypatch.setattr(CudaGraphMixin, 'fill_buffers_cudagraph',
                        lambda self, graph_meta, **kwargs: {'attn_metadata': attn_metadata})
    inputs1 = model.fill_buffers_cudagraph(graph_meta1)
    inputs2 = model.fill_buffers_cudagraph(graph_meta2)

    assert patterns == [(None, False)] * 4
    full1 = inputs1['dflash_full_scheduler_metadata']
    full2 = inputs2['dflash_full_scheduler_metadata']
    assert full1.shape == (2, )
    assert full2.shape == (3, )
    assert full1.data_ptr() == buffers1['dflash_full_scheduler_metadata'].data_ptr()
    assert full2.data_ptr() == buffers2['dflash_full_scheduler_metadata'].data_ptr()
    assert not hasattr(attn_metadata, 'scheduler_metadata_overrides')
    assert attn_metadata.scheduler_metadata is buffers1['scheduler_metadata']


def test_dflash_layer_scheduler_metadata_is_local_and_never_mutates_shared_input():
    draft_config = _draft_config(
        layer_types=['sliding_attention'] * 3 + ['full_attention'],
        sliding_window=4096,
        use_sliding_window=True,
    )
    _parse_dflash(draft_config, num_speculative_tokens=3)
    layer_patterns = [
        _resolve_dflash_layer_attention(draft_config, layer_idx)
        for layer_idx in range(draft_config.num_hidden_layers)
    ]
    assert layer_patterns == [(4096, True)] * 3 + [(None, False)]

    generic = torch.tensor([1, 2, 3], dtype=torch.int32)
    attn_metadata = TritonAttentionMetadata(is_decoding=True,
                                            block_offsets=torch.zeros(1, 1, dtype=torch.int32),
                                            scheduler_metadata=generic)
    swa_layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=4096, causal=True))
    full_layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=None, causal=False))

    swa_metadata = DFlashQwen3DecoderLayer._get_layer_attn_metadata(swa_layer, attn_metadata, None)
    eager_full_metadata = DFlashQwen3DecoderLayer._get_layer_attn_metadata(full_layer, attn_metadata, None)
    full_buffer = torch.tensor([4, 5, 6], dtype=torch.int32)
    graph_full_metadata = DFlashQwen3DecoderLayer._get_layer_attn_metadata(full_layer, attn_metadata, full_buffer)
    prefill_metadata = TritonAttentionMetadata(is_decoding=False,
                                               block_offsets=attn_metadata.block_offsets,
                                               scheduler_metadata=generic)

    assert swa_metadata is attn_metadata
    assert eager_full_metadata is not attn_metadata
    assert eager_full_metadata.scheduler_metadata is None
    assert graph_full_metadata is not attn_metadata
    assert graph_full_metadata.scheduler_metadata is full_buffer
    assert DFlashQwen3DecoderLayer._get_layer_attn_metadata(full_layer, prefill_metadata,
                                                            full_buffer) is prefill_metadata
    assert attn_metadata.scheduler_metadata is generic

    scheduler_policy_by_ptr = {
        generic.data_ptr(): (4096, True),
        full_buffer.data_ptr(): (None, False),
    }
    for sliding_window, causal in layer_patterns:
        layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=sliding_window, causal=causal))
        layer_metadata = DFlashQwen3DecoderLayer._get_layer_attn_metadata(layer, attn_metadata, full_buffer)
        assert scheduler_policy_by_ptr[layer_metadata.scheduler_metadata.data_ptr()] == (sliding_window, causal)


def test_dflash_production_forward_requires_attention_metadata():
    assert inspect.signature(DFlashDraftModel.forward).parameters['attn_metadata'].default is inspect.Parameter.empty
    assert inspect.signature(DFlashQwen3DecoderLayer.forward).parameters[
        'attn_metadata'].default is inspect.Parameter.empty
    assert inspect.signature(DFlashQwen3Attention.forward).parameters[
        'attn_metadata'].default is inspect.Parameter.empty


def test_qwen35_dflash_target_embed_policy_covers_image_and_chunked_multimodal():
    model = object.__new__(Qwen3_5ForConditionalGeneration)
    model.is_spec_decoding = True
    model.requires_target_inputs_embeds = False

    image_context = SimpleNamespace(is_chunk_multimodal=False)
    chunk_context = SimpleNamespace(is_chunk_multimodal=True)
    assert model._should_return_target_inputs_embeds(torch.empty(1), image_context) is False
    assert model._should_return_target_inputs_embeds(None, chunk_context) is False

    model.requires_target_inputs_embeds = True
    assert model._should_return_target_inputs_embeds(torch.empty(1), image_context) is True
    assert model._should_return_target_inputs_embeds(None, chunk_context) is True


def test_qwen35_visual_processing_and_aux_output_do_not_require_returned_input_embeds():

    class FakeEmbeddings(torch.nn.Module):

        def forward(self, input_ids):
            return torch.zeros((*input_ids.shape, 2), dtype=torch.float32)

    class FakeLanguageModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.embeddings = FakeEmbeddings()

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, **kwargs):
            inputs_embeds = kwargs['inputs_embeds']
            return dict(hidden_states=inputs_embeds, aux_hidden_states=inputs_embeds + 1)

    class FakeVisual(torch.nn.Module):
        spatial_merge_size = 1

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, pixel_values, **kwargs):
            self.calls += 1
            return pixel_values

    model = object.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.language_model = FakeLanguageModel()
    model.visual = FakeVisual()
    image_embeds = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

    hidden_states, target_inputs_embeds, aux_hidden_states = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        past_key_values=[],
        attn_metadata=None,
        state_ids=torch.tensor([0]),
        pixel_values=image_embeds,
        vis_cu_seqlens=torch.tensor([2], dtype=torch.int32),
        vis_pos_emb=(torch.ones(2, 1), torch.ones(2, 1)),
        multimodal_mask=torch.tensor([[True, True, False]]),
        grid_thw=torch.tensor([[1, 1, 2]]),
        return_input_embeds=False,
    )

    assert model.visual.calls == 1
    assert target_inputs_embeds is None
    torch.testing.assert_close(hidden_states[0, :2], image_embeds)
    torch.testing.assert_close(aux_hidden_states[0, :2], image_embeds + 1)


def test_dflash_specdecode_builder_preserves_tp(monkeypatch):
    captured = {}

    def _fake_from_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dist_config=kwargs['dist_config'])

    monkeypatch.setattr('lmdeploy.pytorch.engine.config_builder.SpecDecodeConfig.from_config', _fake_from_config)

    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        device_type='cuda',
    )
    dist_config = DistConfig(tp=2)
    speculative_config = SpeculativeConfig(method='dflash',
                                           model=None,
                                           num_speculative_tokens=15,
                                           dflash_block_size=8)
    specdecode_config = ConfigBuilder.build_specdecode_config(
        target_model='target-model',
        speculative_config=speculative_config,
        engine_config=PytorchEngineConfig(tp=2, eager_mode=True),
        cache_config=cache_config,
        dist_config=dist_config,
        trust_remote_code=True,
    )

    assert specdecode_config.dist_config.tp == 2
    assert specdecode_config.dist_config.attn_tp == 2
    assert captured['dist_config'] is not dist_config
    assert captured['num_speculative_tokens'] == 7


def test_dflash_target_model_config_uses_ar_spec_not_dllm():
    hf_config = _draft_config(architectures=['Qwen3ForCausalLM'])

    model_config = ModelConfig.from_hf_config(hf_config, spec_method='dflash')

    assert model_config.model_paradigm == 'ar_spec'
    assert not hasattr(model_config, 'dflash_config')


def test_qwen3_dflash_uses_only_resolved_build_context_metadata(monkeypatch):
    import lmdeploy.pytorch.models.qwen3_dflash as dflash_model_mod

    class FakeAttention(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.causal = True
            self.sliding_window = None

    class FakeLayer(torch.nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.self_attn = FakeAttention()

    monkeypatch.setattr(dflash_model_mod, 'DFlashQwen3DecoderLayer', FakeLayer)
    monkeypatch.setattr(dflash_model_mod, 'build_colwise_linear',
                        lambda in_features, out_features, **kwargs: torch.nn.Linear(in_features, out_features,
                                                                                  bias=False))
    monkeypatch.setattr(dflash_model_mod, 'RMSNorm', lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(dflash_model_mod, 'build_rotary_embedding_from_config',
                        lambda *args, **kwargs: torch.nn.Identity())

    # Raw checkpoint values deliberately disagree with the resolved metadata.
    hf_config = _draft_config(
        target_hidden_size=8,
        rms_norm_eps=1e-6,
        dflash_config=dict(mask_token_id=32001, target_layer_ids=[0]),
    )
    resolved_ctx = BuildModelContext(target_aux_hidden_state_layers=(1, 6, 11),
                                     speculative_mask_token_id=99)
    with build_model_context(resolved_ctx):
        model = DFlashDraftModel(hf_config, ctx_mgr=SimpleNamespace(), device=torch.device('cpu'))

    assert model.target_layer_ids == (1, 6, 11)
    assert model.mask_token_id == 99
    assert model.num_context_features == 3
    assert model.fc.in_features == 24


def test_dflash_module_map_keeps_only_supported_architecture_name():
    assert MODULE_MAP['DFlashDraftModel'].endswith('.qwen3_dflash.DFlashDraftModel')
    assert 'DFlashQwen3ForCausalLM' not in MODULE_MAP
    assert 'Qwen3DFlashForCausalLM' not in MODULE_MAP


def test_qwen3_dflash_weight_name_normalization():
    assert _normalize_dflash_weight_name('model.fc.weight') == 'fc.weight'
    assert _normalize_dflash_weight_name('midlayer.self_attn.q_proj.weight') == 'layers.0.self_attn.q_proj.weight'
    assert _normalize_dflash_weight_name('layers.0.self_attn.rotary_emb.inv_freq') is None


def test_qwen3_dflash_attention_resolver_consumes_trusted_checkpoint_fields():
    sliding_config = _draft_config(
        layer_types=['sliding_attention', 'full_attention', 'full_attention', 'full_attention'],
        sliding_window=4096,
        use_sliding_window=True,
    )
    _parse_dflash(sliding_config, num_speculative_tokens=3)
    assert _resolve_dflash_layer_attention(sliding_config, 0) == (4096, True)
    assert _resolve_dflash_layer_attention(sliding_config, 1) == (None, False)

    resolver_source = inspect.getsource(_resolve_dflash_layer_attention)
    assert '_get_dflash_cfg' not in resolver_source
    assert 'raise ValueError' not in resolver_source


def test_qwen3_dflash_materialization_requires_cpu_max_q_seqlen():
    model = object.__new__(DFlashDraftModel)
    model.precompute_context_kv = lambda target_hidden, position_ids: []

    with pytest.raises(ValueError, match='CPU max_q_seqlen'):
        model.precompute_and_store_context_kv(
            torch.empty(0, 1),
            torch.empty(0, dtype=torch.long),
            past_key_values=[],
            attn_metadata=SimpleNamespace(),
        )


def test_dflash_block_proposer_builds_full_context_and_query_inputs():
    proposer = DFlash(
        SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 20, 21]]),
        seq_length=torch.tensor([3, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=7,
        sum_kv_seqlen=10,
        target_inputs_embeds=torch.randn(1, 5, 4),
    )
    extra_inputs = SimpleNamespace(
        num_rejected_tokens=torch.tensor([1, 0]),
        target_hidden_states=torch.arange(5 * 4, dtype=torch.float32).view(5, 4),
    )

    context_inputs, target_hidden, context_lengths, query_start_positions = proposer._prepare_context_materialization(
        model_inputs, extra_inputs)
    query_inputs = proposer._build_query_inputs(model_inputs, context_lengths, torch.tensor([7, 8]),
                                                query_start_positions=query_start_positions)

    assert not hasattr(DFlash, '_slice_by_lengths')
    assert context_inputs.input_ids.tolist() == [[10, 11, 12, 20, 21]]
    assert context_inputs.seq_length.tolist() == [3, 2]
    assert context_inputs.max_q_seqlen == 3
    assert context_inputs.max_kv_seqlen == 7
    assert context_inputs.sum_kv_seqlen == 10
    assert context_inputs.target_inputs_embeds is None
    assert target_hidden.tolist() == extra_inputs.target_hidden_states.tolist()
    assert context_lengths.tolist() == [2, 2]
    assert query_start_positions.tolist() == [2, 7]
    assert query_inputs.input_ids.tolist() == [[7, 99, 99, 99, 8, 99, 99, 99]]
    assert query_inputs.history_lengths.tolist() == [2, 7]
    assert query_inputs.target_position_ids.tolist() == [[2, 3, 4, 5, 7, 8, 9, 10]]
    assert query_inputs.is_decoding is True
    assert query_inputs.max_q_seqlen == 4
    assert query_inputs.max_kv_seqlen == 11
    assert query_inputs.sum_kv_seqlen == 18
    assert query_inputs.target_inputs_embeds is None


def test_dflash_block_proposer_metadata_helpers_do_not_call_tensor_item(monkeypatch):
    proposer = DFlash(
        SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 20, 21]]),
        seq_length=torch.tensor([3, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=7,
        sum_kv_seqlen=10,
        target_position_ids=torch.tensor([[100, 101, 102, 200, 201]]),
    )
    extra_inputs = SimpleNamespace(
        num_rejected_tokens=torch.tensor([1, 0]),
        target_hidden_states=torch.arange(5 * 4, dtype=torch.float32).view(5, 4),
    )

    def _fail_item(self):
        raise AssertionError('DFlash proposer helper path should not call Tensor.item().')

    monkeypatch.setattr(torch.Tensor, 'item', _fail_item)

    context_inputs, target_hidden, context_lengths, query_start_positions = proposer._prepare_context_materialization(
        model_inputs, extra_inputs)
    query_inputs = proposer._build_query_inputs(model_inputs, context_lengths, torch.tensor([7, 8]),
                                                query_start_positions=query_start_positions)

    assert context_inputs.max_q_seqlen == 3
    assert context_inputs.max_kv_seqlen == 7
    assert context_inputs.sum_kv_seqlen == 10
    assert context_inputs.seq_length.tolist() == [3, 2]
    assert target_hidden.shape == (5, 4)
    assert query_start_positions.tolist() == [102, 202]
    assert query_inputs.max_q_seqlen == 4
    assert query_inputs.max_kv_seqlen == 11
    assert query_inputs.sum_kv_seqlen == 18


def test_dflash_block_proposer_uses_explicit_target_positions():
    proposer = DFlash(
        SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 20, 21]]),
        seq_length=torch.tensor([3, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=7,
        sum_kv_seqlen=10,
        target_position_ids=torch.tensor([[100, 101, 102, 200, 201]]),
    )
    extra_inputs = SimpleNamespace(
        num_rejected_tokens=torch.tensor([1, 1]),
        target_hidden_states=torch.arange(5 * 4, dtype=torch.float32).view(5, 4),
    )

    context_inputs, target_hidden, context_lengths, query_start_positions = proposer._prepare_context_materialization(
        model_inputs, extra_inputs)
    query_inputs = proposer._build_query_inputs(model_inputs, context_lengths, torch.tensor([7, 8]),
                                                query_start_positions=query_start_positions)

    assert context_inputs.input_ids.tolist() == [[10, 11, 12, 20, 21]]
    assert context_inputs.target_position_ids.tolist() == [[100, 101, 102, 200, 201]]
    assert target_hidden.tolist() == extra_inputs.target_hidden_states.tolist()
    assert context_lengths.tolist() == [2, 1]
    assert query_start_positions.tolist() == [102, 201]
    assert query_inputs.target_position_ids.tolist() == [[102, 103, 104, 105, 201, 202, 203, 204]]


def test_dflash_decode_materialization_uses_full_block_without_ragged_slice(monkeypatch):
    proposer = DFlash(
        SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]]),
        seq_length=torch.tensor([4, 4]),
        history_lengths=torch.tensor([100, 200]),
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=4,
        max_kv_seqlen=204,
        sum_kv_seqlen=308,
        target_position_ids=torch.tensor([[100, 101, 102, 103, 200, 201, 202, 203]]),
    )
    extra_inputs = SimpleNamespace(
        num_rejected_tokens=torch.tensor([2, 0]),
        target_hidden_states=torch.arange(8 * 4, dtype=torch.float32).view(8, 4),
    )

    assert not hasattr(DFlash, '_slice_by_lengths')

    context_inputs, target_hidden, context_lengths, query_start_positions = \
        proposer._prepare_context_materialization(model_inputs, extra_inputs)
    query_inputs = proposer._build_query_inputs(model_inputs, context_lengths, torch.tensor([7, 8]),
                                                query_start_positions=query_start_positions)

    assert context_inputs.is_decoding is False
    assert context_inputs.input_ids.tolist() == model_inputs.input_ids.tolist()
    assert context_inputs.seq_length.tolist() == [4, 4]
    assert context_inputs.target_position_ids.tolist() == model_inputs.target_position_ids.tolist()
    assert target_hidden.tolist() == extra_inputs.target_hidden_states.tolist()
    assert context_lengths.tolist() == [2, 4]
    assert query_start_positions.tolist() == [102, 204]
    assert query_inputs.history_lengths.tolist() == [102, 204]
    assert query_inputs.target_position_ids.tolist() == [[102, 103, 104, 105, 204, 205, 206, 207]]


def test_dflash_prefill_materialize_context_uses_full_block_without_ragged_slice(monkeypatch):
    proposer = DFlash(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=8),
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 20, 21]]),
        seq_length=torch.tensor([3, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=7,
        sum_kv_seqlen=10,
    )
    extra_inputs = SimpleNamespace(
        num_rejected_tokens=torch.tensor([1, 0]),
        target_hidden_states=torch.arange(5 * 4, dtype=torch.float32).view(5, 4),
    )
    captured = {}

    def _materialize_context(context_inputs, target_hidden, cache_engine):
        captured['context_inputs'] = context_inputs
        captured['target_hidden'] = target_hidden
        captured['cache_engine'] = cache_engine

    assert not hasattr(DFlash, '_slice_by_lengths')

    monkeypatch.setattr(proposer, '_materialize_context', _materialize_context)
    cache_engine = object()

    proposer.materialize_context(model_inputs, extra_inputs, cache_engine)

    assert captured['cache_engine'] is cache_engine
    assert captured['context_inputs'].input_ids.tolist() == [[10, 11, 12, 20, 21]]
    assert captured['context_inputs'].seq_length.tolist() == [3, 2]
    assert captured['target_hidden'].tolist() == extra_inputs.target_hidden_states.tolist()
