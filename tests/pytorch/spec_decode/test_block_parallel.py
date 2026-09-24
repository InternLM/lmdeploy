# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.model_agent.dp_utils import DPForwardMeta, GatheredDPForwardMeta
from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs
from lmdeploy.pytorch.spec_decode.base import BaseSpecModelAgent
from lmdeploy.pytorch.spec_decode.block_parallel import BlockDraftStepPlan, context_inputs, prepare_query


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('width', [17, 18])
@pytest.mark.parametrize('inactive', [False, True])
@pytest.mark.parametrize('graph', [False, True])
def test_query_padding_real_cache_write_isolation(monkeypatch, width, inactive, graph):
    """Exercise the actual pageable fill kernel, not just scratch table
    values."""
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache

    monkeypatch.setattr(
        DPMeta, 'build',
        staticmethod(lambda n, counts: DPMeta(tp_sizes=list(counts), moe_tp_sizes=list(counts))))
    device = 'cuda'
    meta = DPMeta(tp_sizes=[width, width * 3], moe_tp_sizes=[width, width * 3],
                  dp_batches=[1, 3], dp_is_decoding=True,
                  block_plan=BlockDraftStepPlan((1, 3), (not inactive, True), True))
    inp = ModelInputs(
        input_ids=torch.zeros((1, width), dtype=torch.long, device=device),
        seq_length=torch.tensor([width], device=device),
        history_lengths=torch.zeros(1, dtype=torch.long, device=device),
        block_offsets=torch.tensor([[30, 31]], device=device),
        is_decoding=True, is_dummy=inactive,
        num_ignored_history=torch.zeros(1, dtype=torch.long, device=device),
        state_offsets=torch.tensor([8], device=device),
        max_q_seqlen=width, max_kv_seqlen=width, sum_kv_seqlen=width,
        target_position_ids=torch.arange(width, device=device)[None], dp_meta=meta)
    model = SimpleNamespace(backend_config=SimpleNamespace(eager_mode=False),
                            get_meta=lambda: SimpleNamespace(),
                            _get_capture_tokens=lambda n: 4, update_inputs=lambda x: x)
    proposer = SimpleNamespace(model=model, specdecode_config=SimpleNamespace(mask_token_id=99))
    cache = SimpleNamespace(cache_config=SimpleNamespace(
        block_size=32, kernel_block_size=16, num_reserved_gpu_blocks=0))
    query = prepare_query(proposer, inp, cache)
    # Dummy values deliberately differ; only real cache isolation is required.
    values = torch.arange(1, 5, device=device).repeat_interleave(width).to(torch.float16)
    keys = values[:, None, None].expand(-1, 1, 32).contiguous()
    key_cache = torch.full((40, 16, 1, 32), -123, dtype=torch.float16, device=device)
    starts = torch.arange(4, device=device) * width
    def fill():
        fill_kv_cache(keys, None, key_cache, None, starts, query.seq_length,
                      query.seq_length + query.history_lengths, width, query.block_offsets)

    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fill()
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            fill()
        for _ in range(3):
            captured.replay()
    else:
        fill()
    expected = torch.full_like(key_cache, -123)
    if not inactive:
        expected[30] = 1
        expected[31, :width - 16] = 1
    # Block 0 is writable scratch, not a no-write sentinel. Dummy collisions
    # may change its contents, but cannot touch any other request's KV pages.
    torch.testing.assert_close(key_cache[1:], expected[1:], rtol=0, atol=0)
    assert query.block_offsets[0 if inactive else 1:].count_nonzero() == 0
    assert query.state_offsets.tolist() == ([-1] * 4 if inactive else [8, -1, -1, -1])


@pytest.mark.parametrize('inactive', [False, True])
@pytest.mark.parametrize('graph', [False, True])
def test_block_query_physical_padding_and_metadata_ownership(
        monkeypatch, inactive, graph):
    width = 5  # Cross a kernel-block boundary; Q=N/N+1 layouts are covered below.
    monkeypatch.setattr(
        DPMeta, 'build',
        staticmethod(lambda n, counts: DPMeta(tp_sizes=list(counts),
                                              moe_tp_sizes=list(counts))))
    plan = BlockDraftStepPlan((2, 1), (True, not inactive), True)
    meta = DPMeta(tp_sizes=[12, 6],
                  moe_tp_sizes=[12, 6],
                  dp_batches=[2, 1],
                  dp_is_decoding=True,
                  block_plan=plan)
    inp = ModelInputs(input_ids=torch.arange(width)[None],
                      seq_length=torch.tensor([width]),
                      history_lengths=torch.tensor([64]),
                      block_offsets=torch.tensor([[30, 31]]),
                      is_decoding=True,
                      num_ignored_history=torch.zeros(1, dtype=torch.long),
                      max_q_seqlen=width,
                      max_kv_seqlen=64 + width,
                      sum_kv_seqlen=64 + width,
                      dp_meta=meta,
                      state_offsets=torch.tensor([8]),
                      target_position_ids=torch.arange(64, 64 + width)[None],
                      is_chunk=inactive,
                      is_last_chunk=False)
    graph_meta = SimpleNamespace(padding_batch_size=None)

    def update(inputs):
        inputs.dp_meta.sync_tp_size(inputs.input_ids.numel())
        return inputs

    model = SimpleNamespace(
        backend_config=SimpleNamespace(eager_mode=not graph),
        get_meta=lambda: graph_meta,
        _get_capture_tokens=lambda n: 4,
        update_inputs=update)
    proposer = SimpleNamespace(
        model=model, specdecode_config=SimpleNamespace(mask_token_id=99))
    cache = SimpleNamespace(cache_config=SimpleNamespace(
        block_size=8, kernel_block_size=4, num_reserved_gpu_blocks=0))
    out = prepare_query(proposer, inp, cache)
    bound = 4 if graph else 2
    assert out.input_ids.shape == (1, bound * width)
    assert out.dp_meta.tp_sizes == [bound * width] * 2
    assert out.dp_meta is not meta and meta.tp_sizes == [12, 6]
    assert graph_meta.padding_batch_size == bound
    real = 0 if inactive else 1
    assert out.state_offsets.tolist() == ([8] if real else
                                          []) + [-1] * (bound - real)
    assert out.block_offsets[real:].count_nonzero() == 0
    assert out.history_lengths[real:].count_nonzero() == 0
    if real:
        torch.testing.assert_close(out.block_offsets[:real], inp.block_offsets)


def test_block_metadata_exchange_preserves_ar_schema():
    flags = dict(is_spec_enabled=True, is_microbatch_enabled=False)
    row = DPForwardMeta(True, False, 12, False, 2, 12, block_query_ready=True)
    assert len(row.values(**flags)) == 6
    rows = torch.tensor([row.values(**flags, is_block_spec_enabled=True)] * 2)
    gathered = GatheredDPForwardMeta.from_values(rows,
                                                 **flags,
                                                 is_block_spec_enabled=True)
    assert gathered.block_query_ready.tolist() == [1, 1]
    assert gathered.all_draft_num_tokens == [12, 12]
    assert not BlockDraftStepPlan((1, 1),
                                  (False, False), False).run_query


@pytest.mark.parametrize('mode', ['dp1', 'prefill', 'decode', 'dummy'])
def test_context_local_metadata_preserves_cuda_backend_inputs(monkeypatch, mode):
    """Actual context/backend construction; only the KV writer is a spy.

    Compare the old fresh-DPMeta contract with no-DPMeta local prefill, including ragged chunks and graph-padded target
    metadata ownership. This is a CPU metadata test, not a GPU projection/kernel parity claim.
    """
    from dataclasses import fields

    from lmdeploy.pytorch import model_inputs as inputs_module
    from lmdeploy.pytorch.backends.cuda.attention.v4 import TritonV4AttentionImpl
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from lmdeploy.pytorch.model_inputs import StepContextManager, step_ctx_manager
    from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash

    monkeypatch.setattr(inputs_module, 'get_backend', lambda: CudaOpsBackend)
    lengths = [3, 3] if mode == 'decode' else [2, 3]
    n = sum(lengths)
    meta = None if mode == 'dp1' else DPMeta(
        tp_sizes=[32, 32], moe_tp_sizes=[32, 32], dp_batches=[2, 4],
        dp_is_decoding=mode == 'decode')
    inputs = ModelInputs(
        input_ids=torch.arange(n)[None], seq_length=torch.tensor(lengths),
        history_lengths=torch.tensor([8, 16]), block_offsets=torch.tensor([[1, 2], [3, 4]]),
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        is_decoding=False, is_dummy=mode == 'dummy', is_chunk=mode == 'prefill',
        max_q_seqlen=3, max_kv_seqlen=19, sum_kv_seqlen=n + 24,
        state_offsets=torch.tensor([1, 2]), dp_meta=meta)
    mgr = StepContextManager()
    model_config = ModelConfig(8, 1, 1, 1, 0, [1], 8)
    cache_config = CacheConfig(max_batches=4, block_size=16, num_gpu_blocks=8, num_cpu_blocks=0)
    cache = SimpleNamespace(cache_config=cache_config, gpu_cache=[], block_caches={},
                            state_cache_engine=None)
    # The old helper produced fresh counts and forced global prefill mode.
    old_meta = None if meta is None else DPMeta(
        tp_sizes=[n, 9], moe_tp_sizes=[n, 9], dp_batches=[2, 4], dp_is_decoding=False)
    with step_ctx_manager(mgr):
        reference = mgr.build_context(inputs.clone(dp_meta=old_meta), model_config, cache_config, [])

    def unexpected_build(*args, **kwargs):
        pytest.fail('local context must not rebuild DPMeta')

    monkeypatch.setattr(DPMeta, 'build', unexpected_build)
    calls = []

    def write(**kwargs):
        context = mgr.current_context()
        assert context.dp_meta is None and not context.global_is_decoding()
        assert context.state_offsets is inputs.state_offsets
        torch.testing.assert_close(kwargs['position_ids'], reference.position_ids.flatten())
        for field in fields(reference.attn_metadata):
            actual = getattr(kwargs['attn_metadata'], field.name)
            expected = getattr(reference.attn_metadata, field.name)
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected, field.name
        # The supported V4 full-context path consumes local ring-write
        # metadata, not MLP/MoE token counts. Keep the real metadata builder.
        v4 = SimpleNamespace(compress_ratio=0, window_size=8, ring_storage_capacity=12)
        actual = TritonV4AttentionImpl.build_cache_write_metadata(
            v4, context.attn_metadata, context.position_ids, context.state_offsets, n)
        expected = TritonV4AttentionImpl.build_cache_write_metadata(
            v4, reference.attn_metadata, reference.position_ids, reference.state_offsets, n)
        torch.testing.assert_close(actual.slot, expected.slot, rtol=0, atol=0)
        torch.testing.assert_close(actual.ring_pos, expected.ring_pos, rtol=0, atol=0)
        calls.append(kwargs['max_q_seqlen'])

    proposer = DFlash.__new__(DFlash)
    proposer.model = SimpleNamespace(ctx_mgr=mgr, precompute_and_store_context_kv=write)
    proposer.specdecode_config = SimpleNamespace(model_config=model_config)
    proposer._materialize_context(inputs, torch.ones(n, 8), cache)
    assert calls == ([] if mode == 'dummy' else [3])
    assert inputs.dp_meta is meta
    if meta is not None:
        assert meta.tp_sizes == meta.moe_tp_sizes == [32, 32]
        assert meta.dp_is_decoding == (mode == 'decode')
    else:
        assert context_inputs(inputs) is inputs


@pytest.mark.parametrize('method', ['dflash', 'dspark', 'qwen3_5_mtp'])
@pytest.mark.parametrize('kind',
                         ['normal', 'nonlast', 'last', 'dummy', 'scoring'])
def test_host_plan_query_readiness_and_mtp_chunk_counts(
        monkeypatch, method, kind):
    import asyncio

    from lmdeploy.pytorch.engine.model_agent import agent as module

    monkeypatch.setattr(
        DPMeta, 'build',
        staticmethod(lambda n, counts: DPMeta(tp_sizes=list(counts),
                                              moe_tp_sizes=list(counts))))
    captured = []

    class Gather:

        def __init__(self, values, *args, **kwargs):
            captured.append(values)
            self.values = torch.tensor([values, values])

        async def async_wait(self):
            return self.values

    monkeypatch.setattr(module, 'DistGatherScalar', Gather)
    inputs = ModelInputs(input_ids=torch.arange(8)[None],
                         seq_length=torch.tensor([8]),
                         history_lengths=torch.tensor([0]),
                         block_offsets=torch.tensor([[20]]),
                         is_decoding=False,
                         num_ignored_history=torch.tensor([0]),
                         max_q_seqlen=8,
                         max_kv_seqlen=8,
                         sum_kv_seqlen=8,
                         is_dummy=kind == 'dummy',
                         is_chunk=kind in ('nonlast', 'last'),
                         is_first_chunk=kind == 'nonlast',
                         is_last_chunk=kind == 'last')
    if kind == 'scoring':
        inputs.logits_indices = torch.tensor([0])
        inputs.seq_logit_length = torch.tensor([1])
    agent = module.BaseModelAgent.__new__(module.BaseModelAgent)
    agent.dist_config = SimpleNamespace(world_size=2)
    agent.dist_ctx = SimpleNamespace(cpu_group=None)
    agent.state = SimpleNamespace(is_sleeping=False)
    agent.enable_microbatch = False
    agent.spec_agent = BaseSpecModelAgent.__new__(BaseSpecModelAgent)
    agent.spec_agent.method = method
    agent.spec_agent._enabled = True
    agent.spec_agent.proposer = None
    agent.patched_model = SimpleNamespace(update_inputs=lambda inputs: inputs)
    output, sleeping = asyncio.run(agent._prepare_dp_v1(inputs))
    values = captured[0]
    if method == 'qwen3_5_mtp':
        assert len(values) == 6
        assert values[5] == (7 if kind == 'nonlast' else
                             9 if kind == 'last' else 8)
    else:
        assert len(values) == 7
        assert values[5] == 8
        assert values[6] == (kind in ('normal', 'last'))
        if output is not None:
            assert output.dp_meta.block_plan.run_query == (kind in ('normal',
                                                                    'last'))
    assert (output is None) == (kind == 'dummy')


@pytest.mark.parametrize('need_output', [False, True])
def test_scoring_only_rank_participates_with_aux_hidden(
        monkeypatch, need_output):
    import asyncio

    from lmdeploy.pytorch.engine.model_agent import agent as module

    cfg = SimpleNamespace(attn_tp=1, dp=2)
    monkeypatch.setattr(
        module, 'get_dist_manager', lambda: SimpleNamespace(
            current_context=lambda: SimpleNamespace(dist_config=cfg)))
    monkeypatch.setattr(module, 'cache_swapping', lambda *args, **kwargs: None)
    inputs = ModelInputs(input_ids=torch.arange(3)[None],
                         seq_length=torch.tensor([3]),
                         history_lengths=torch.tensor([0]),
                         block_offsets=torch.tensor([[20]]),
                         is_decoding=False,
                         num_ignored_history=torch.tensor([0]),
                         max_q_seqlen=3,
                         max_kv_seqlen=3,
                         sum_kv_seqlen=3,
                         logits_indices=torch.tensor([0, 1]),
                         seq_logit_length=torch.tensor([2]),
                         dp_meta=DPMeta(tp_sizes=[3, 4]))
    aux = torch.zeros(1, 3, 12)
    calls = []

    async def prepare(inputs):
        return inputs, False

    async def forward(*args, **kwargs):
        return dict(hidden_states=[torch.zeros(1, 3, 4)],
                    aux_hidden_states=aux,
                    logits=torch.zeros(2, 9))

    async def draft(inputs, extra, sampling):
        assert inputs.is_dummy and inputs.dp_meta is not None
        assert extra.target_hidden_states is aux
        calls.append('query')

    agent = module.BaseModelAgent.__new__(module.BaseModelAgent)
    agent.rank = 0
    agent.kv_connector = None
    agent.cache_engine = None
    agent.memdecode_agent = None
    agent.need_output = need_output
    agent._prepare_inputs_prefill = lambda inputs, delta: inputs
    agent._prepare_dp_v1 = prepare
    agent._async_model_forward = forward
    agent.spec_agent = BaseSpecModelAgent.__new__(BaseSpecModelAgent)
    agent.spec_agent.method = 'dflash'
    agent.spec_agent.proposer = None
    agent.spec_agent.async_model_forward = draft
    agent._get_outputs_with_logprobs = lambda *args: SimpleNamespace(
        kv_connector_output=None)
    agent._push_output = lambda *args: calls.append('output')
    asyncio.run(
        agent._async_step(inputs=inputs,
                          sampling_inputs=SimpleNamespace(max_num_logprobs=0),
                          stopping_criteria=None,
                          swap_in_map={},
                          swap_out_map={}))
    assert calls == ['query'] + (['output'] if need_output else [])


@pytest.mark.parametrize('method', ['dflash', 'dspark'])
@pytest.mark.parametrize('explicit', [False, True])
def test_low_capacity_block_query_warmup(monkeypatch, method, explicit):
    """Public config -> final cache sizing -> real proposer -> target
    allocator."""
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.config import CacheConfig, DistConfig
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
    from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _WorkerCachePlanSizes
    from lmdeploy.pytorch.paging.block_manager import build_block_manager
    from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash
    from lmdeploy.pytorch.spec_decode.proposers.dspark import DSpark
    from lmdeploy.pytorch.strategies.ar_spec.model_inputs import ARSpecModelInputsStrategy

    # Original regression: B32 would reserve 33 private blocks in a 32-block pool.
    capacity = 32
    public = PytorchEngineConfig(dp=2, tp=2, max_batch_size=32,
                                 num_gpu_blocks=capacity if explicit else 0,
                                 cache_max_entry_count=0.5)
    executor = ExecutorBase.__new__(ExecutorBase)
    cfg = executor.cache_config = ConfigBuilder.build_cache_config(public)
    draft = CacheConfig(max_batches=32, block_size=64, kernel_block_size=64,
                        num_cpu_blocks=0, num_gpu_blocks=public.num_gpu_blocks)
    executor.dist_config = DistConfig(dp=2, tp=2)
    executor.specdecode_config = SimpleNamespace(
        method=method, num_speculative_tokens=3, dspark_draft_query_len=3,
        cache_config=draft, mask_token_id=9)
    executor._sync_spec_cache_block_size()
    mib = 1024 * 1024
    # Synthetic byte sizes; both explicit/auto final-capacity resolution are real.
    executor._update_num_gpu_blocks([capacity * 4 * mib] * 2,
                                    [_WorkerCachePlanSizes(mib, mib)] * 2, draft)
    cls = DFlash if method == 'dflash' else DSpark
    proposer = cls.__new__(cls)
    proposer.num_speculative_tokens = 3
    proposer.specdecode_config = executor.specdecode_config
    proposer._materialize_context = lambda *args: None
    proposer._configure_context_materialization = lambda *args: None
    proposer._full_context_materialization = True
    proposer.model = SimpleNamespace(
        backend_config=SimpleNamespace(eager_mode=False), _get_capture_tokens=lambda n: n,
        get_meta=lambda: SimpleNamespace(padding_batch_size=None), update_inputs=lambda x: x)
    width = 4 if method == 'dflash' else 3
    inputs = ARSpecModelInputsStrategy(3).make_dummy(
        32, is_decoding=True, device='cpu', vocab_size=16,
        max_q_seqlen=width, target_hidden_size=4)
    inputs.dp_meta = DPMeta(tp_sizes=[32 * width] * 2, moe_tp_sizes=[32 * width] * 2,
                            dp_batches=[32, 32], dp_is_decoding=True)
    monkeypatch.setattr(DPMeta, 'build', staticmethod(
        lambda n, counts: DPMeta(tp_sizes=list(counts), moe_tp_sizes=list(counts))))
    query = proposer.prepare_warmup_forward(inputs, SimpleNamespace(cache_config=draft))
    assert cfg.num_gpu_blocks == draft.num_gpu_blocks == capacity
    assert cfg.num_reserved_gpu_blocks == 1 and draft.num_reserved_gpu_blocks == 0
    assert query.block_offsets.count_nonzero() == 0
    assert build_block_manager(cfg).get_num_free_gpu_blocks() == capacity - 1


@pytest.mark.parametrize('method', ['dflash', 'dspark'])
@pytest.mark.parametrize('unsupported',
                         ['architecture', 'microbatch', 'transfer', 'pd'])
def test_distributed_block_capability_guards_before_model_loading(
        monkeypatch, method, unsupported):
    from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
    from lmdeploy.pytorch.config import CacheConfig, DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.disagg.config import EngineRole
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder

    arch = 'UnknownDraft' if unsupported == 'architecture' else 'DFlashDraftModel'
    resolved = SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch])))
    monkeypatch.setattr(SpecDecodeConfig, 'from_config',
                        lambda **kwargs: resolved)
    engine = PytorchEngineConfig(dp=2,
                                 ep=2,
                                 enable_microbatch=unsupported == 'microbatch')
    cache = CacheConfig(max_batches=8,
                        block_size=64,
                        num_cpu_blocks=0,
                        num_gpu_blocks=128)
    if unsupported == 'transfer':
        cache.kv_transfer_config = {}
    if unsupported == 'pd':
        cache.role = EngineRole.Prefill
    with pytest.raises(ValueError, match='not support'):
        ConfigBuilder.build_specdecode_config('unused',
                                              SpeculativeConfig(method=method),
                                              engine, cache,
                                              DistConfig(dp=2, ep=2))


@pytest.mark.parametrize('method,width', [('dflash', 3), ('dspark', 2), ('dspark', 3)])
@pytest.mark.parametrize('kinds', [('idle', 'decode'), ('nonlast', 'decode'),
                                 ('nonlast', 'nonlast'), ('scoring', 'decode'),
                                 ('prefill', 'prefill')])
def test_block_producer_consumer_participation_and_output_ownership(monkeypatch, method, width, kinds):
    import asyncio

    from lmdeploy.pytorch.spec_decode.proposers.base import ProposalContext
    from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash
    from lmdeploy.pytorch.spec_decode.proposers.dspark import DSpark
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

    monkeypatch.setattr(DPMeta, 'build', staticmethod(
        lambda n, counts: DPMeta(tp_sizes=list(counts), moe_tp_sizes=list(counts))))
    ready = tuple(kind in ('decode', 'prefill') for kind in kinds)
    plan = BlockDraftStepPlan((1, 3), ready, False)
    traces = []
    for rank, kind in enumerate(kinds):
        batch = plan.batches[rank]
        meta = DPMeta(tp_sizes=[3, 9], moe_tp_sizes=[3, 9], dp_batches=[1, 3], block_plan=plan)
        inputs = ModelInputs(input_ids=torch.zeros((1, batch * 3), dtype=torch.long),
                             seq_length=torch.full((batch,), 3), history_lengths=torch.full((batch,), 64),
                             block_offsets=torch.arange(20, 20 + batch).reshape(batch, 1),
                             is_decoding=kind in ('idle', 'decode'), is_dummy=kind in ('idle', 'scoring'),
                             is_chunk=kind == 'nonlast', is_last_chunk=False,
                             num_ignored_history=torch.zeros(batch, dtype=torch.long),
                             max_q_seqlen=3, max_kv_seqlen=67, sum_kv_seqlen=batch * 67, dp_meta=meta)
        cls = DFlash if method == 'dflash' else DSpark
        proposer = cls.__new__(cls)
        proposer.num_speculative_tokens = 2
        proposer.specdecode_config = SimpleNamespace(mask_token_id=9, dspark_draft_query_len=width)
        proposer._full_context_materialization = True
        proposer.guided_helper = SimpleNamespace(get_processors=lambda *args: None)
        proposer.model = SimpleNamespace(backend_config=SimpleNamespace(eager_mode=True),
                                         get_meta=lambda: SimpleNamespace(padding_batch_size=0),
                                         update_inputs=lambda inputs: inputs)
        trace = []
        graph_outputs = []

        def materialize(inputs, hidden, cache):
            context = context_inputs(inputs)
            assert context.dp_meta is None
            assert inputs.dp_meta is meta and meta.tp_sizes == [3, 9]

        def forward(inputs, cache_engine):
            assert inputs.dp_meta is not meta
            assert inputs.dp_meta.tp_sizes == [3 * width] * 2
            trace.append(('query', inputs.input_ids.numel()))
            if kind in ('idle', 'scoring', 'nonlast'):
                assert inputs.history_lengths.tolist() == [0] * 3
                assert inputs.block_offsets[:, 0].tolist() == [0, 0, 0]
            if method == 'dflash':
                return dict(hidden_states=torch.zeros(1, 3 * width, 4))
            trace.append(('head', 6))
            ids = torch.zeros((3, 2), dtype=torch.long)
            graph_outputs.append(ids)
            return dict(draft_token_ids=ids)

        def head(hidden):
            trace.append(('head', hidden.size(1)))
            return torch.zeros(1, hidden.size(1), 16)

        proposer._materialize_context = materialize
        proposer._forward = forward
        proposer.get_logits = head
        cache = SimpleNamespace(cache_config=SimpleNamespace(block_size=64, kernel_block_size=64,
                                                              num_reserved_gpu_blocks=0))
        extra = ARSpecExtraInputs(next_token_ids=torch.zeros(batch, dtype=torch.long),
                                  target_hidden_states=torch.zeros(1, batch * 3, 4))
        result = asyncio.run(proposer.propose(inputs, extra, None, ProposalContext(cache_engine=cache)))
        assert result.output_draft_token_ids.shape == (batch, 2)
        for ids in graph_outputs:
            ids.fill_(7)
        assert result.output_draft_token_ids.count_nonzero() == 0
        assert meta.tp_sizes == [3, 9]
        traces.append(trace)
    assert traces[0] == traces[1] == ([('query', 3 * width), ('head', 6)] if any(ready) else [])
