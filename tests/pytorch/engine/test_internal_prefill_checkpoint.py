from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs, PrefillStateCapture
from lmdeploy.pytorch.engine.inputs_maker import _ForwardInputsResult, _ForwardInputsTask
from lmdeploy.pytorch.messages import MessageStatus


def make_task(start=0, end=8256, step=8192, budget=16384, reserve=True, mtp=False):
    seq = SimpleNamespace(
        num_history_ids=start, input_end_pos=end, num_token_ids=end - start,
        logprob_start_pos=-1, return_logits=False, return_ce_loss=False, return_routed_experts=False,
        history_embeddings=[], history_multimodals=SimpleNamespace(multimodals={}),
        _seq_meta=SimpleNamespace(prefill_save_alignment=8192, prefix_cache_token_lookahead=int(mtp)),
        kv_token_limit=step,
        session_id=1, seq_id=2)

    def reserve_chunk(seq, chunk_size, prealloc_size, is_last_chunk):
        assert chunk_size == end - start and is_last_chunk and prealloc_size == 1
        if reserve:
            seq.kv_token_limit = None
        return reserve

    maker = SimpleNamespace(
        config=SimpleNamespace(enable_internal_prefill_checkpoint=True, max_prefill_token_num=budget,
                               prefill_graph_max_tokens=2048, role=EngineRole.Hybrid),
        spec_decoding=mtp, running_seqs=[],
        scheduler=SimpleNamespace(has_kv_connector=lambda: False,
                                  reserve_long_context_chunk=Mock(side_effect=reserve_chunk)),
        engine_strategy=SimpleNamespace(get_prealloc_size=lambda prefill: 1),
        long_context_chunker=SimpleNamespace(clear=Mock()),
        model_agent_strategy=SimpleNamespace(make_extra_inputs=lambda seqs, inputs: None),
        create_model_inputs=Mock(return_value=SimpleNamespace()),
        create_model_inputs_delta_valid_only=lambda: (None, [], []),
        _prepare_prefill_cache_inputs=Mock(side_effect=AssertionError('Do not reserve H a second time')))
    task = _ForwardInputsTask(maker, prefill=True)
    result = _ForwardInputsResult(
        running=[seq], inputs=SimpleNamespace(max_q_seqlen=step - start, is_first_chunk=start == 0),
        cache_inputs=CacheCheckpointInputs(state_restore_plan=((2,), (1,)), state_save_plan=((1,), (3,))))
    return task, result


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize(('start', 'end', 'step'), [(0, 8193, 8192), (0, 8256, 8192), (0, 10000, 8192),
                                                   (8192, 20000, 16384)])
def test_internal_capture_preserves_one_reservation_and_restore(start, end, step, mtp):
    task, result = make_task(start, end, step, mtp=mtp)
    assert task._try_internal_prefill(result)
    assert result.cache_inputs.state_capture == PrefillStateCapture(start, end, step, 1, 3)
    assert result.cache_inputs.state_save_plan is None
    assert result.cache_inputs.state_restore_plan == ((2,), (1,))
    assert result.inputs.is_chunk and result.inputs.is_last_chunk
    assert result.inputs.is_first_chunk == (start == 0)
    task.maker._prepare_prefill_cache_inputs.assert_not_called()
    task.maker.long_context_chunker.clear.assert_called_once()
    assert result.cache_inputs.to_device('cpu').state_capture == result.cache_inputs.state_capture


@pytest.mark.parametrize('condition', ['disabled', 'unsupported_spec', 'mixed', 'connector', 'logprob', 'logits',
                                      'routes', 'embeddings',
                                      'multimodal', 'end_checkpoint', 'budget', 'pcg', 'no_slot', 'kv_copy',
                                      'unaligned_start', 'no_expansion_capacity'])
def test_internal_capture_falls_back_without_losing_original_plan(condition):
    task, result = make_task(reserve=condition != 'no_expansion_capacity')
    maker, seq = task.maker, result.running[0]
    if condition == 'disabled':
        maker.config.enable_internal_prefill_checkpoint = False
    elif condition == 'unsupported_spec':
        maker.spec_decoding = True
    elif condition == 'mixed':
        maker.running_seqs = [SimpleNamespace(status=MessageStatus.RUNNING)]
    elif condition == 'connector':
        maker.scheduler.has_kv_connector = lambda: True
    elif condition == 'logprob':
        seq.logprob_start_pos = 0
    elif condition == 'logits':
        seq.return_logits = True
    elif condition == 'routes':
        seq.return_routed_experts = True
    elif condition == 'embeddings':
        seq.history_embeddings = [object()]
    elif condition == 'multimodal':
        seq.history_multimodals.multimodals = {'image': [object()]}
    elif condition == 'end_checkpoint':
        seq.input_end_pos = seq.num_token_ids = 16384
    elif condition == 'budget':
        maker.config.max_prefill_token_num = 8192
    elif condition == 'pcg':
        maker.config.prefill_graph_max_tokens = 16384
    elif condition == 'no_slot':
        result.cache_inputs = replace(result.cache_inputs, state_save_plan=None)
    elif condition == 'kv_copy':
        result.cache_inputs = replace(result.cache_inputs, kv_save_plan=torch.tensor([[1], [2]]))
    elif condition == 'unaligned_start':
        seq.num_history_ids = 1
        seq.num_token_ids -= 1
        result.inputs.max_q_seqlen -= 1
    original_cache, original_inputs = result.cache_inputs, result.inputs
    assert not task._try_internal_prefill(result)
    assert result.cache_inputs is original_cache and result.inputs is original_inputs
    assert seq.kv_token_limit == 8192
    maker._prepare_prefill_cache_inputs.assert_not_called()
    maker.long_context_chunker.clear.assert_not_called()
    maker.create_model_inputs.assert_not_called()
    assert maker.scheduler.reserve_long_context_chunk.call_count == (condition == 'no_expansion_capacity')


@pytest.mark.parametrize('arguments', [(0, 64, 64, 1, 2), (64, 128, 64, 1, 2), (0, 128, 64, 1, 1),
                                      (0, 128, 64, -1, 2)])
def test_invalid_internal_capture_is_rejected(arguments):
    with pytest.raises(ValueError):
        PrefillStateCapture(*arguments)


def test_capture_cannot_be_overwritten_by_final_copy():
    with pytest.raises(ValueError):
        CacheCheckpointInputs(state_capture=PrefillStateCapture(0, 8256, 8192, 1, 3),
                              state_save_plan=((1,), (3,)))


def test_completed_tracked_sequence_does_not_block_capture_or_drop_cleanup_delta():
    task, result = make_task(start=8192, end=20000, step=16384)
    finished = SimpleNamespace(status=MessageStatus.STOPPED)
    task.maker.running_seqs = [finished]
    cleanup_delta = SimpleNamespace(indice_cpu=[])
    task.maker.create_model_inputs_delta_valid_only = lambda: (cleanup_delta, [], [finished])
    assert task._try_internal_prefill(result)
    assert result.delta is cleanup_delta
    assert task.maker.running_seqs == []
    assert result.cache_inputs.state_capture == PrefillStateCapture(8192, 20000, 16384, 1, 3)


def test_mtp_can_merge_aligned_end_without_losing_an_endpoint_checkpoint():
    task, result = make_task(end=16384, mtp=True)
    assert task._try_internal_prefill(result)
    assert result.cache_inputs.state_capture == PrefillStateCapture(0, 16384, 8192, 1, 3)
    assert result.cache_inputs.state_save_plan is None


def test_mtp_capacity_failure_preserves_pending_checkpoint():
    task, result = make_task(end=16384, mtp=True, reserve=False)
    original = result.cache_inputs
    assert not task._try_internal_prefill(result)
    assert result.cache_inputs is original
    task.maker.long_context_chunker.clear.assert_not_called()


def test_mtp_ic_config_accepts_lookahead_without_enabling_decode_saves(monkeypatch):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda import flash_kda
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.configurations.glm5_next import update_cache_config

    monkeypatch.setattr(envs, 'internal_prefill_checkpoint', True)
    monkeypatch.setattr(envs, 'kda_prefill_backend', 'flash_kda')
    monkeypatch.setattr(flash_kda, 'check_flash_kda', Mock())
    config = CacheConfig(max_batches=32, block_size=64, num_cpu_blocks=0, num_gpu_blocks=64,
                         enable_prefix_caching=True, prefill_save_alignment=8192)
    update_cache_config(config, token_lookahead=1)
    assert config.enable_internal_prefill_checkpoint
    assert config.prefix_cache_token_lookahead == 1
    config.prefix_cache_decode_state_interval = 64
    with pytest.raises(ValueError, match='only publishes prefill'):
        update_cache_config(config, token_lookahead=1)


@pytest.mark.parametrize(('internal', 'routes'), [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('return_embeds', [False, True])
def test_wrapper_preserves_capture_routes_and_mtp_embeddings(internal, routes, return_embeds):
    from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 4)
            self.seen_capture = None
            self.seen_routes = None

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, input_ids, inputs_embeds, all_routed_experts, state_capture, **kwargs):
            self.seen_capture = state_capture
            self.seen_routes = all_routed_experts
            if all_routed_experts is not None:
                all_routed_experts[:, 1].fill_(7)
            return self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.enable_return_routed_experts = routes
    model.config = SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=2)
    model.model = Backbone()
    input_ids = torch.arange(65).remainder(16).unsqueeze(0)
    capture = PrefillStateCapture(0, 65, 64, 1, 2) if internal else None
    output = model(input_ids, input_ids, [], state_capture=capture, return_input_embeds=return_embeds)
    expected = model.get_input_embeddings()(input_ids)
    assert model.model.seen_capture is capture
    if not routes and not return_embeds:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert model.model.seen_routes is None
        return
    torch.testing.assert_close(output['hidden_states'], expected, rtol=0, atol=0)
    assert ('target_inputs_embeds' in output) == return_embeds
    assert ('all_routed_experts' in output) == routes
    if return_embeds:
        torch.testing.assert_close(output['target_inputs_embeds'], expected, rtol=0, atol=0)
    if routes:
        assert output['all_routed_experts'] is model.model.seen_routes
        assert output['all_routed_experts'].shape == (65, 3, 2)
        assert torch.all(output['all_routed_experts'][:, 1].int() == 7)
