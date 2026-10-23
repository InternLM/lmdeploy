# Copyright (c) OpenMMLab. All rights reserved.
"""Model-agent/real full-graph integration with a small recurrent test model.

No attention, weights or scheduler: this does not establish Qwen/GLM E2E gates.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.compact_state import CompactStateInputs, CompactStatePrefill, CompactStateRuntime
from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.model_agent.agent import model_forward
from lmdeploy.pytorch.model_inputs import BuildModelContext, ModelInputs, StepContextManager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin


def _inputs(batch=3, width=4, prefill=False, device='cuda', base=10):
    length = 72 if prefill else width
    bank = torch.arange(base, base + batch * (width + 1), device=device).reshape(batch, width + 1)
    if device == 'cuda':
        compact = CompactStateInputs.plan(bank, torch.ones(batch, device=device, dtype=torch.long),
                                           1 if prefill else width)
    else:
        compact = CompactStateInputs(bank, torch.ones(batch, dtype=torch.long), bank[:, :1].contiguous())
    return ModelInputs(input_ids=torch.arange(batch * length, device=device)[None] % 16,
                       seq_length=torch.full((batch,), length, device=device),
                       history_lengths=torch.full((batch,), 2048, device=device),
                       block_offsets=torch.zeros(batch, 1, device=device, dtype=torch.long),
                       num_ignored_history=torch.zeros(batch, device=device, dtype=torch.long),
                       is_decoding=not prefill, max_q_seqlen=length, max_kv_seqlen=2048 + length,
                       sum_kv_seqlen=batch * (2048 + length),
                       state_offsets=torch.arange(batch, device=device) + 1000, compact_state=compact)


@pytest.mark.parametrize('case', [
    'legacy_allocation', 'legacy_inputs', 'legacy_copy', 'missing_prefill', 'stale_prefill', 'unconsumed_source'
])
def test_model_forward_rejects_mode_mismatch_before_any_mutation(case):
    inputs = _inputs(batch=1, width=1, prefill=case == 'missing_prefill', device='cpu')
    state_cache = SimpleNamespace(compact_plan=SimpleNamespace(unit_specs=(object(),)))
    carrier = None
    if case == 'legacy_allocation':
        state_cache.compact_plan = None
    elif case == 'legacy_inputs':
        inputs.compact_state = None
    elif case == 'legacy_copy':
        carrier = CacheCheckpointInputs(state_save_plan=((1,), (2,)))
    elif case == 'stale_prefill':
        carrier = CacheCheckpointInputs(compact_prefill=CompactStatePrefill.from_lengths([1], [2048]))
    elif case == 'unconsumed_source':
        carrier = CacheCheckpointInputs(compact_source_ids=torch.ones(1, dtype=torch.long))
    # No context, stream or cache-copy API exists on these objects. Validation
    # must raise before attempting CUDA work or partially applying KV copies.
    with pytest.raises(ValueError, match='Compact|compact|One-forward'):
        model_forward(object(), inputs, None, object(), state_cache, cache_inputs=carrier)


class _RecurrentModel(torch.nn.Module, CudaGraphMixin):
    """One genuine CUDA conv/GDN block behind normal model-agent methods."""

    def __init__(self, conv, recurrent, width):
        super().__init__()
        from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
        from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaMetaImpl, CudaGatedDeltaRuleImpl
        self.ctx_mgr = StepContextManager(BuildModelContext(num_spec_tokens=width - 1))
        self.conv, self.recurrent = conv, recurrent
        self.meta = CudaGatedDeltaMetaImpl()
        self.conv_op, self.rule = CausalConv1dTilelangImpl(), CudaGatedDeltaRuleImpl()
        self.embedding = torch.randn(16, 384, device='cuda', dtype=torch.bfloat16) / 8
        self.weight = torch.randn(384, 4, device='cuda', dtype=torch.bfloat16) / 8
        self.gate_bias = torch.zeros(1, device='cuda')
        self.log_exp = -torch.ones(1, device='cuda')
        self.calls = 0

    def update_model_metas(self, **kwargs):
        return None

    def prepare_inputs_for_generation(self, past_key_values, context):
        return dict(input_ids=context.input_ids, position_ids=context.position_ids, past_key_values=past_key_values,
                    attn_metadata=context.attn_metadata, state_ids=context.state_offsets, inputs_embeds=None)

    def forward(self, input_ids, position_ids, past_key_values, attn_metadata, state_ids, inputs_embeds=None):
        self.calls += 1
        x = torch.nn.functional.embedding(input_ids, self.embedding)
        meta = self.meta.forward(x.shape[1], 4, state_ids, attn_metadata)
        mixed = self.conv_op.forward(x, self.weight, None, self.conv, meta, 'silu')
        q, k, v = [t.unflatten(-1, (1, 128)) for t in mixed.split(128, -1)]
        gate = mixed[..., :1].contiguous()
        return self.rule.forward(q, k, v, gate, gate, self.gate_bias, self.log_exp,
                                 self.recurrent, meta, 1, True).flatten(-2)


def _environment(monkeypatch, width):
    from lmdeploy.pytorch import model_inputs as module

    def update(context):
        # Supply only canonical attention metadata; no attention operator runs.
        context.attn_metadata = SimpleNamespace(
            is_decoding=context.is_decoding, q_seqlens=context.q_seqlens, kv_seqlens=context.kv_seqlens,
            q_start_loc=context.q_start_loc, block_offsets=context.block_offsets,
            cu_seqlens_q=torch.nn.functional.pad(context.q_seqlens.cumsum(0), (1, 0)))
        return context

    monkeypatch.setattr(module, 'get_backend', lambda: SimpleNamespace(update_step_context=update))
    conv = torch.randn(128, 384, 3, device='cuda', dtype=torch.bfloat16) / 8
    recurrent = torch.randn(128, 1, 128, 128, device='cuda', dtype=torch.bfloat16) / 16
    conv[0].zero_()
    recurrent[0].zero_()
    model = _RecurrentModel(conv, recurrent, width)
    model_config = SimpleNamespace(vocab_size=16, use_mla_fp8_cache=False, use_flash_mla=False, mla_index_topk=None,
                                   model_paradigm='ar', states_shapes=[(1,)], use_mrope=False, block_size=64)
    cache = SimpleNamespace(cache_config=SimpleNamespace(window_size=0, quant_policy=0), gpu_cache=[], block_caches={})
    state_cache = SimpleNamespace(compact_plan=SimpleNamespace(unit_specs=(object(),)),
                                  state_caches=[conv, recurrent], named_state_caches={})
    return model, model_config, cache, state_cache


def test_model_agent_consumes_one_forward_prefill_carrier(monkeypatch):
    model, config, cache, state_cache = _environment(monkeypatch, 4)
    inputs = _inputs(batch=1, width=4, prefill=True)
    plan = CompactStatePrefill.from_lengths([72], [2048], checkpoint_ids=[100], checkpoint_offsets=[64])
    carrier = CacheCheckpointInputs(compact_prefill=plan).to_device('cuda', non_blocking=True)
    before = model.conv.clone(), model.recurrent.clone()
    output = model_forward(model, inputs, config, cache, state_cache, cache_inputs=carrier)
    assert model.calls == 1
    assert output['hidden_states'].shape == (1, 72, 128)
    assert output['seq_length'].tolist() == [72]
    assert inputs.compact_state.output_ids.shape == (1, 1)
    assert not hasattr(inputs, 'compact_prefill')
    untouched = torch.ones(len(model.conv), device='cuda', dtype=torch.bool)
    untouched[100] = False
    untouched[inputs.compact_state.output_ids.flatten()] = False
    for pool, initial in zip((model.conv, model.recurrent), before):
        torch.testing.assert_close(pool[untouched], initial[untouched], rtol=0, atol=0)
    # Independent shorter model-agent call verifies the exact exported boundary.
    prefix = inputs.clone(input_ids=inputs.input_ids[:, :64], seq_length=inputs.seq_length.new_tensor([64]),
                          max_q_seqlen=64)
    short = CacheCheckpointInputs(compact_prefill=CompactStatePrefill.from_lengths([64], [2048]).to_device('cuda'))
    model_forward(model, prefix, config, cache, state_cache, cache_inputs=short)
    for pool in (model.conv, model.recurrent):
        torch.testing.assert_close(pool[100], pool[int(inputs.compact_state.output_ids[0, 0])], rtol=0, atol=0)


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('allow_input_alias', [False, True])
def test_real_full_graph_runner_stages_versions_and_masks_retired_rows(monkeypatch, width, allow_input_alias):
    from lmdeploy.pytorch.backends.cuda.graph_runner.full_graph import CUDASingleGraphRunner
    model, config, cache, state_cache = _environment(monkeypatch, width)
    runner = CUDASingleGraphRunner(model, max_batches=8, max_tokens=8 * width, num_blocks=1,
                                   is_decoding=True, decode_query_len=width, pool=torch.cuda.graph_pool_handle(),
                                   model_config=config, device=torch.device('cuda'))

    class Captured:
        ctx_mgr = model.ctx_mgr
        update_model_metas = model.update_model_metas
        prepare_inputs_for_generation = model.prepare_inputs_for_generation
        captured = False

        def __call__(self, **kwargs):
            if not self.captured:
                output = runner.capture(**kwargs)
                self.captured = True
                return output
            return runner.forward(**kwargs)

    wrapper = Captured()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Serving captures with dummy rows, never live private state: capture warmups
    # may execute repeatedly and therefore cannot consume a real alias input.
    from lmdeploy.pytorch.strategies.base.model_inputs import MakeDummyMeta, make_dummy_inputs
    dummy = make_dummy_inputs(3, width, True, device='cuda', vocab_size=16,
                              meta=MakeDummyMeta(use_ssm=True,
                                                 compact_bank_width=width + int(not allow_input_alias),
                                                 compact_allow_input_alias=allow_input_alias))
    model_forward(wrapper, dummy, config, cache, state_cache, stream=stream)
    stream.synchronize()
    saved_pointers = None
    protected = None
    for iteration, batch in enumerate([3, 1, 2, 3]):
        inputs = _inputs(batch=batch, width=width, base=10 + iteration * 20)
        if allow_input_alias:
            banks = inputs.compact_state.bank_ids[:, :width].contiguous()
            inputs.compact_state = CompactStateInputs.plan(banks, banks[:, 0].contiguous(), width,
                                                           allow_input_alias=True)
        if protected is not None:
            # A frozen unit outside the replacement bank is now a shared input.
            inputs.compact_state.input_ids[0] = protected[0]
            inputs.compact_state = CompactStateInputs.plan(
                inputs.compact_state.bank_ids, inputs.compact_state.input_ids, width,
                allow_input_alias=allow_input_alias)
        snapshots = [t.clone() for t in (inputs.compact_state.bank_ids,
                                         inputs.compact_state.input_ids, inputs.compact_state.output_ids)]
        before = model.conv.clone(), model.recurrent.clone()
        stream.wait_stream(torch.cuda.current_stream())
        actual = model_forward(wrapper, inputs, config, cache, state_cache, stream=stream)['hidden_states']
        stream.synchronize()
        for pool, initial in zip((model.conv, model.recurrent), before):
            pool.copy_(initial)
        expected = model_forward(model, inputs, config, cache, state_cache)['hidden_states']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        static = runner.meta.compact_state_buffers
        assert static.allow_input_alias == allow_input_alias
        pointers = [t.data_ptr() for t in (static.bank_ids, static.input_ids, static.output_ids)]
        assert saved_pointers is None or saved_pointers == pointers
        torch.testing.assert_close(runner.meta.input_buffers['state_ids'][:batch], inputs.state_offsets, rtol=0, atol=0)
        assert (runner.meta.input_buffers['state_ids'][batch:] == -1).all()
        saved_pointers = pointers
        for original, snapshot, staged in zip((inputs.compact_state.bank_ids, inputs.compact_state.input_ids,
                                               inputs.compact_state.output_ids), snapshots,
                                              (static.bank_ids, static.input_ids, static.output_ids)):
            torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
            torch.testing.assert_close(staged[:batch], snapshot, rtol=0, atol=0)
            assert (staged[batch:] == -1).all()
        untouched = torch.ones(len(model.conv), device='cuda', dtype=torch.bool)
        untouched[inputs.compact_state.output_ids.flatten()] = False
        for pool, initial in zip((model.conv, model.recurrent), before):
            torch.testing.assert_close(pool[untouched], initial[untouched], rtol=0, atol=0)
        if protected is not None:
            for pool, saved in zip((model.conv, model.recurrent), protected[1:]):
                torch.testing.assert_close(pool[protected[0]], saved, rtol=0, atol=0)
        else:
            unit = int(inputs.compact_state.output_ids[0, -1])
            protected = unit, model.conv[unit].clone(), model.recurrent[unit].clone()


@pytest.mark.parametrize('prefill', [False, True])
def test_dummy_model_forward_never_writes_units(monkeypatch, prefill):
    from lmdeploy.pytorch.strategies.base.model_inputs import MakeDummyMeta, make_dummy_inputs
    model, config, cache, state_cache = _environment(monkeypatch, 4)
    inputs = make_dummy_inputs(3, 72 if prefill else 4, not prefill, device='cuda', vocab_size=16,
                               meta=MakeDummyMeta(use_ssm=True, compact_bank_width=5))
    before = model.conv.clone(), model.recurrent.clone()
    model_forward(model, inputs, config, cache, state_cache)
    for pool, initial in zip((model.conv, model.recurrent), before):
        torch.testing.assert_close(pool, initial, rtol=0, atol=0)


def test_piecewise_manager_replays_live_prefill_checkpoint_ids(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import PiecewiseGraphManager
    from lmdeploy.pytorch.backends.cuda.graph_runner.standard import StandardDecoderPiecewiseGraphRuntime
    from lmdeploy.pytorch.strategies.base.model_inputs import MakeDummyMeta, make_dummy_inputs
    model, config, cache, state_cache = _environment(monkeypatch, 4)
    for impl in (model.meta, model.conv_op, model.rule):
        impl.enable_piecewise_cuda_graph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    manager = PiecewiseGraphManager(StandardDecoderPiecewiseGraphRuntime(model, max_capture_tokens=128,
                                                                        token_stride=128), stream=stream)

    class Piecewise:
        ctx_mgr = model.ctx_mgr
        update_model_metas = model.update_model_metas
        prepare_inputs_for_generation = model.prepare_inputs_for_generation

        def __call__(self, **kwargs):
            context = self.ctx_mgr.current_context()
            descriptor = manager.get_piecewise_graph_descriptor(context, kwargs)
            assert descriptor is not None
            if not manager.has_plan(descriptor):
                assert context.is_dummy, 'Only startup dummy calls may prepare this plan.'
                return manager.prepare(descriptor, kwargs)
            return manager.replay(descriptor, kwargs)

    wrapper = Piecewise()
    dummy = make_dummy_inputs(1, 128, False, device='cuda', vocab_size=16,
                              meta=MakeDummyMeta(use_ssm=True, compact_bank_width=5))
    original = model.conv.clone(), model.recurrent.clone()
    stream.wait_stream(torch.cuda.current_stream())
    model_forward(wrapper, dummy, config, cache, state_cache, stream=stream)
    stream.synchronize()
    for pool, initial in zip((model.conv, model.recurrent), original):
        torch.testing.assert_close(pool, initial, rtol=0, atol=0)
    frozen = None
    for iteration, length in enumerate([72, 65, 64]):
        inputs = _inputs(batch=1, width=4, prefill=True, base=10 + iteration * 20)
        inputs = inputs.clone(input_ids=inputs.input_ids[:, :length], seq_length=inputs.seq_length.new_tensor([length]),
                              max_q_seqlen=length)
        if frozen is not None:
            inputs.compact_state.input_ids.fill_(100)
        plan = CompactStatePrefill.from_lengths([length], [2048], checkpoint_ids=[100 + iteration],
                                                checkpoint_offsets=[64])
        carrier = CacheCheckpointInputs(compact_prefill=plan).to_device('cuda')
        before = model.conv.clone(), model.recurrent.clone()
        stream.wait_stream(torch.cuda.current_stream())
        actual = model_forward(wrapper, inputs, config, cache, state_cache, stream=stream,
                                cache_inputs=carrier)['hidden_states']
        stream.synchronize()
        # Clone graph outputs in the oracle only; eager cannot overwrite the
        # shared output bridge and thereby mask an incorrect replay result.
        actual = actual.clone()
        written = torch.cat((inputs.compact_state.output_ids.flatten(), carrier.compact_prefill.checkpoint_ids))
        saved = [pool[written].clone() for pool in (model.conv, model.recurrent)]
        expected = model_forward(model, inputs, config, cache, state_cache, cache_inputs=carrier)['hidden_states']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        untouched = torch.ones(len(model.conv), device='cuda', dtype=torch.bool)
        untouched[written] = False
        for pool, values, initial in zip((model.conv, model.recurrent), saved, before):
            torch.testing.assert_close(pool[written], values, rtol=0, atol=0)
            torch.testing.assert_close(pool[untouched], initial[untouched], rtol=0, atol=0)
        if frozen is None:
            frozen = model.conv[100].clone(), model.recurrent[100].clone()
        else:
            for pool, saved in zip((model.conv, model.recurrent), frozen):
                torch.testing.assert_close(pool[100], saved, rtol=0, atol=0)
    assert len(manager._plans) == 1
    from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import EagerStep, GraphStep
    steps = next(iter(manager._plans.values())).graph.steps
    assert sum(isinstance(step, EagerStep) for step in steps) == 3  # metadata, convolution, recurrence
    assert any(isinstance(step, GraphStep) for step in steps)


def test_zero_resource_plan_keeps_stateless_model_forward_without_ids(monkeypatch):
    from contextlib import nullcontext

    from lmdeploy.pytorch.engine.model_agent import agent as module
    inputs = _inputs(batch=1, width=1, device='cpu').clone(compact_state=None, state_offsets=None)
    context = SimpleNamespace(q_seqlens=inputs.seq_length, position_ids=torch.zeros(1, 1, dtype=torch.long))
    manager = SimpleNamespace(build_context=lambda **kwargs: context, context=lambda ctx: nullcontext())

    class Stateless:
        ctx_mgr = manager

        def update_model_metas(self, **kwargs):
            return None

        def prepare_inputs_for_generation(self, **kwargs):
            return {}

        def __call__(self, **kwargs):
            return torch.ones(1, 1, 2)

    monkeypatch.setattr(module, 'step_ctx_manager', lambda mgr: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    cache = SimpleNamespace(cache_config=SimpleNamespace(quant_policy=0), gpu_cache=[], block_caches={})
    state_cache = SimpleNamespace(compact_plan=SimpleNamespace(unit_specs=()), state_caches=[], named_state_caches={})
    output = model_forward(Stateless(), inputs, None, cache, state_cache, stream=object())
    assert output['hidden_states'].shape == (1, 1, 2)
    assert inputs.compact_state is None and inputs.state_offsets is None


def test_graph_mode_change_rejected_before_buffer_fill(monkeypatch):
    from lmdeploy.pytorch.models.utils import cudagraph as module
    context = SimpleNamespace(compact_state=object())
    monkeypatch.setattr(module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    # No attention metadata or tensors: guard must execute before any fill.
    with pytest.raises(ValueError, match='resetting'):
        CudaGraphMixin().fill_buffers_cudagraph(SimpleNamespace(compact_state_buffers=None), None, None, [], None, None)


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('path', ['sampling', 'scoring', 'non_output_rank'])
@pytest.mark.parametrize('retain_runtime', [False, True])
def test_async_step_continues_chunks_from_final_with_current_bank(monkeypatch, width, path, retain_runtime):
    """Real chunk prepare/store transitions and recurrent forwards, fake
    sampler.

    Explicit-source independent pools are the oracle. A fresh host packet has the initial source again; blindly
    consuming it makes this test fail.
    """
    import asyncio

    from lmdeploy.pytorch.engine.model_agent import agent as module
    from lmdeploy.pytorch.strategies.ar.model_inputs import get_model_inputs_next_decoding

    model, config, cache, state_cache = _environment(monkeypatch, width)
    reference = _RecurrentModel(model.conv.clone(), model.recurrent.clone(), width)
    for name in ('embedding', 'weight', 'gate_bias', 'log_exp'):
        setattr(reference, name, getattr(model, name))
    ref_cache = SimpleNamespace(compact_plan=state_cache.compact_plan,
                                state_caches=[reference.conv, reference.recurrent], named_state_caches={})
    agent = module.BaseModelAgent.__new__(module.BaseModelAgent)
    agent.rank = 1 if path == 'non_output_rank' else 0
    agent.need_output = path != 'non_output_rank'
    agent.kv_connector = agent.memdecode_agent = None
    agent.cache_engine = cache
    agent.cache_config = SimpleNamespace(role=module.EngineRole.Hybrid)
    agent.spec_agent = SimpleNamespace(cache_engine=None, is_enabled=lambda: False)
    agent._prev_chunk_output = None
    agent.compact_runtime = CompactStateRuntime(1100) if retain_runtime else None
    agent._out_que = asyncio.Queue()
    agent.agent_strategy = SimpleNamespace(slice_extra_inputs=lambda extra, *_: extra)
    agent._slice_outs = lambda logits, lengths: logits[-1:]
    agent._get_outputs_with_logprobs = lambda *_: SimpleNamespace(kv_connector_output=None)
    actual = []
    merged = []
    if retain_runtime:
        updates = []
        advance = agent.compact_runtime.advance
        push = agent._push_output

        def record(*args, **kwargs):
            result = advance(*args, **kwargs)
            updates.append(result)
            return result

        def push_after_record(output):
            assert len(updates) == len(actual), 'Completion event must follow the current-ID write'
            push(output)

        agent.compact_runtime.advance = record
        agent._push_output = push_after_record
    agent.step_inputs = SimpleNamespace(merge_prefill=lambda inputs, *_: merged.append(
        get_model_inputs_next_decoding(inputs, torch.zeros((1, width), device='cuda', dtype=torch.long), width, None,
                                       compact_runtime=agent.compact_runtime)))
    monkeypatch.setattr(module, 'get_dist_manager', lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
        dist_config=SimpleNamespace(attn_tp=1, dp=1))))

    async def forward(inputs, return_logits, cache_inputs=None):
        output = model_forward(model, inputs, config, cache, state_cache, cache_inputs=cache_inputs)
        actual.append(output['hidden_states'].clone())
        return dict(logits=output['hidden_states'], model_metas=[{'boundary': inputs.max_kv_seqlen}],
                    seq_length=inputs.seq_length)

    async def sample(logits, all_logits, inputs, sampling, stopping, metas, broadcast, **kwargs):
        return inputs, None, None, None, torch.zeros(1, device='cuda', dtype=torch.long), SimpleNamespace()

    async def without_output(inputs, *args):
        return inputs, torch.zeros(1, device='cuda', dtype=torch.long), None, None

    agent._async_model_forward = forward
    agent._step_postprocess_with_output = sample
    agent._step_postprocess_without_output = without_output
    sampling = SimpleNamespace(max_num_logprobs=0, get_delta=lambda: None)
    previous_final = torch.ones(1, device='cuda', dtype=torch.long)
    history = 2048
    frozen = None
    for index, length in enumerate((72, 65, 8)):
        # Keep the bank for the middle chunk; replace it for the final chunk.
        # Source may then be outside the bank, e.g. a frozen former final unit.
        inputs = _inputs(batch=1, width=width, prefill=True, base=10 if index < 2 else 40)
        inputs = inputs.clone(input_ids=torch.arange(length, device='cuda')[None] % 16,
                              seq_length=torch.tensor([length], device='cuda'),
                              history_lengths=torch.tensor([history], device='cuda'),
                              max_q_seqlen=length, max_kv_seqlen=history + length,
                              sum_kv_seqlen=history + length, is_chunk=True,
                              is_first_chunk=index == 0, is_last_chunk=index == 2)
        if path == 'scoring':
            inputs.logits_indices = torch.arange(length, device='cuda')
            inputs.seq_logit_length = inputs.seq_length
        expected_ids = CompactStateInputs.plan(inputs.compact_state.bank_ids, previous_final, 1)
        expected_inputs = inputs.clone(compact_state=expected_ids)
        plan = CompactStatePrefill.from_lengths([length], [history],
                                                checkpoint_ids=[100 + index] if index < 2 else None,
                                                checkpoint_offsets=[64] if index < 2 else None)
        carrier = CacheCheckpointInputs(
            compact_prefill=plan.to_device('cuda'), compact_access_id=index + 1,
            compact_source_ids=(torch.ones(1, device='cuda', dtype=torch.long)
                                if retain_runtime and index == 0 else None))
        snapshot = inputs.compact_state
        saved_ids = snapshot.input_ids.clone(), snapshot.output_ids.clone()
        asyncio.run(agent._async_step(inputs, sampling_inputs=sampling, cache_inputs=carrier))
        acknowledgment, event = agent._out_que.get_nowait()
        assert acknowledgment.compact_access_id == index + 1
        if path == 'non_output_rank':
            assert acknowledgment.next_token_ids is None
        assert agent._out_que.empty()
        expected = model_forward(reference, expected_inputs, config, cache, ref_cache,
                                 cache_inputs=replace(carrier, compact_source_ids=None))
        torch.testing.assert_close(actual[-1], expected['hidden_states'], rtol=0, atol=0)
        for pool, ref in ((model.conv, reference.conv), (model.recurrent, reference.recurrent)):
            torch.testing.assert_close(pool, ref, rtol=0, atol=0)
        torch.testing.assert_close(inputs.compact_state.input_ids, previous_final, rtol=0, atol=0)
        for original, saved in zip((snapshot.input_ids, snapshot.output_ids), saved_ids):
            torch.testing.assert_close(original, saved, rtol=0, atol=0)
        previous_final = expected_ids.output_ids[:, 0].contiguous()
        if retain_runtime:
            torch.testing.assert_close(agent.compact_runtime.current_ids[inputs.state_offsets], previous_final,
                                       rtol=0, atol=0)
        if index < 2:
            carry = agent._prev_chunk_output
            assert set(carry) == {'model_metas', 'compact_state', 'seq_length'}
            assert carry['compact_state'] is inputs.compact_state
            # Interleaved ordinary prefill must not consume or clear the carry.
            interleaved = _inputs(batch=1, width=width, prefill=True, base=70)
            interleaved.state_offsets += 1
            interleaved.history_lengths.zero_()
            agent._prepare_inputs_prefill(interleaved, None)
            assert agent._prev_chunk_output is carry
        else:
            assert agent._prev_chunk_output is None
            if path != 'scoring':
                assert len(merged) == 1
                torch.testing.assert_close(merged[0].compact_state.input_ids, previous_final, rtol=0, atol=0)
                assert merged[0].compact_state.output_ids.shape == (1, width)
        if index == 0:
            frozen = model.conv[100].clone(), model.recurrent[100].clone()
        else:
            for pool, saved in zip((model.conv, model.recurrent), frozen):
                torch.testing.assert_close(pool[100], saved, rtol=0, atol=0)
        history += length


@pytest.mark.parametrize('case', ['missing_previous', 'lost_compact', 'unexpected_compact', 'new_first'])
def test_chunk_carry_rejects_mode_loss_and_clears_on_new_first(case):
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    agent = BaseModelAgent.__new__(BaseModelAgent)
    inputs = _inputs(batch=1, width=1, prefill=True, device='cpu')
    inputs.is_chunk = True
    previous = dict(compact_state=inputs.compact_state, seq_length=inputs.seq_length, model_metas=['old'])
    agent._prev_chunk_output = None if case == 'missing_previous' else previous
    if case == 'lost_compact':
        inputs.compact_state = None
    elif case == 'unexpected_compact':
        previous['compact_state'] = None
    elif case == 'new_first':
        inputs.is_first_chunk = True
        original = inputs.compact_state
        agent._prepare_inputs_prefill(inputs, None)
        assert inputs.compact_state is original and agent._prev_chunk_output is None
        assert inputs.model_metas is None
        return
    with pytest.raises(ValueError, match='Compact'):
        agent._prepare_inputs_prefill(inputs, None)
    assert agent._prev_chunk_output is (None if case == 'missing_previous' else previous)


@pytest.mark.parametrize('width', [1, 4])
def test_resumed_and_recycled_requests_match_explicit_source_model_after_batch_drop(monkeypatch, width):
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
    from lmdeploy.pytorch.strategies.ar.model_inputs import get_model_inputs_next_decoding, index_select_model_inputs

    model, config, cache, state_cache = _environment(monkeypatch, width)
    reference = _RecurrentModel(model.conv.clone(), model.recurrent.clone(), width)
    for name in ('embedding', 'weight', 'gate_bias', 'log_exp'):
        setattr(reference, name, getattr(model, name))
    ref_cache = SimpleNamespace(compact_plan=state_cache.compact_plan,
                                state_caches=[reference.conv, reference.recurrent], named_state_caches={})
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.compact_runtime = CompactStateRuntime(4)
    agent._prev_chunk_output = None
    current = {}
    # A restores boundary2048, B is cold, A resumes after dropping every active
    # row, then A's auxiliary slot is recycled for a new cold request C.
    for slot, base, length, history in ((1, 10, 72, 2048), (2, 40, 65, 0),
                                       (1, 10, 8, 2120), (1, 70, 72, 0)):
        inputs = _inputs(batch=1, width=width, prefill=True, base=base).clone(
            input_ids=torch.arange(length, device='cuda')[None] % 16,
            seq_length=torch.tensor([length], device='cuda'), state_offsets=torch.tensor([slot], device='cuda'),
            history_lengths=torch.tensor([history], device='cuda'), max_q_seqlen=length,
            max_kv_seqlen=history + length, sum_kv_seqlen=history + length)
        source = (torch.ones(1, device='cuda', dtype=torch.long) if history == 2048 else
                  torch.zeros(1, device='cuda', dtype=torch.long) if history == 0 else current[slot])
        expected_ids = CompactStateInputs.plan(inputs.compact_state.bank_ids, source, 1)
        expected_inputs = inputs.clone(compact_state=expected_ids)
        original = inputs.compact_state
        agent._prepare_inputs_prefill(inputs, None, compact_source_ids=source if history == 2048 else None)
        torch.testing.assert_close(inputs.compact_state.input_ids, source, rtol=0, atol=0)
        assert inputs.compact_state is not original
        carrier = CacheCheckpointInputs(compact_prefill=CompactStatePrefill.from_lengths(
            [length], [history]).to_device('cuda'))
        actual = model_forward(model, inputs, config, cache, state_cache, cache_inputs=carrier)
        expected = model_forward(reference, expected_inputs, config, cache, ref_cache, cache_inputs=carrier)
        torch.testing.assert_close(actual['hidden_states'], expected['hidden_states'], rtol=0, atol=0)
        for pool, ref in ((model.conv, reference.conv), (model.recurrent, reference.recurrent)):
            torch.testing.assert_close(pool, ref, rtol=0, atol=0)
        decoding = get_model_inputs_next_decoding(inputs, torch.zeros(1, device='cuda', dtype=torch.long), 1, None,
                                                  compact_runtime=agent.compact_runtime)
        current[slot] = expected_ids.output_ids[:, 0].contiguous()
        dropped = index_select_model_inputs(decoding, torch.empty(0, device='cuda', dtype=torch.long))
        assert dropped.compact_state.input_ids.numel() == 0
        agent.step_inputs = SimpleNamespace(model_inputs=dropped)
        for request, unit_id in current.items():
            torch.testing.assert_close(agent.compact_runtime.current_ids[request:request + 1], unit_id, rtol=0, atol=0)
