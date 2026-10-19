# Copyright (c) OpenMMLab. All rights reserved.
"""Retained worker plan sizing/allocation, with explicit development guards."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import (
    BackendConfig,
    CacheConfig,
    CompactStateCapacity,
    DistConfig,
    MiscConfig,
    ModelConfig,
    StateCacheSpec,
)
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _WorkerCachePlanSizes
from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
from lmdeploy.pytorch.engine.model_agent import agent as agent_module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def _configs(width=1, family='qwen3_5'):
    specs = [StateCacheSpec('state_0', (2, 8, 3 + width), torch.bfloat16, compact_shape=(2, 8, 3)),
             StateCacheSpec('state_1', (2, width, 2, 4, 4), torch.float32, compact_shape=(2, 2, 4, 4))]
    model = ModelConfig(hidden_size=16, num_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         bos_token_id=1, eos_token_id=[2], head_dim=8, is_gated_delta=True,
                         state_cache_specs=specs, states_shapes=[(s.shape, s.dtype) for s in specs],
                         hf_config=SimpleNamespace(model_type=family), model_paradigm='ar' if width == 1 else 'ar_spec')
    cache = CacheConfig(max_batches=2, block_size=16, num_cpu_blocks=0, num_gpu_blocks=32,
                         states_shapes=model.states_shapes, compact_state_capacity=CompactStateCapacity(4, width))
    spec = None if width == 1 else SimpleNamespace(
        method='deepseek_mtp' if family == 'glm5_next' else 'qwen3_5_mtp', num_speculative_tokens=width - 1,
        model_config=SimpleNamespace(states_shapes=[], num_layers=1), cache_config=replace(cache,
                                                                                          compact_state_capacity=None))
    return model, cache, spec


def _executor(model, cache, spec=None, **kwargs):
    return ExecutorBase('unused', model, cache, BackendConfig(), DistConfig(), MiscConfig(),
                         specdecode_config=spec, **kwargs)


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('family', ['qwen3_5', 'glm5_next'])
@pytest.mark.parametrize('allow_input_alias', [False, True])
def test_worker_retains_exact_plan_for_sizing_real_allocation_and_dummy(monkeypatch, width, family, allow_input_alias):
    model, cache, spec = _configs(width, family)
    cache.compact_state_capacity = replace(cache.compact_state_capacity, allow_input_alias=allow_input_alias)
    executor = _executor(model, cache, spec)
    assert cache.num_state_caches == 4  # max_batches+2 unchanged
    assert cache.compact_state_capacity.request_capacity == 3
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.rank, agent.device = 0, 'cuda'
    agent.dist_config = DistConfig()
    agent.model_config, agent.cache_config = model, cache
    agent.patched_model = torch.nn.Module()
    agent.memdecode_agent = None
    draft_calls = []
    agent.spec_agent = SimpleNamespace(build_cache_plan=lambda _: 0,
                                       build_cache_engine=lambda _: draft_calls.append('allocate'))
    agent.kv_connector = None
    agent.cache_stream = torch.cuda.Stream()
    agent.step_inputs = SimpleNamespace(compact_runtime=None)
    agent.make_dummy_meta = SimpleNamespace(compact_bank_width=0)
    sizes = _WorkerCachePlanSizes(*agent.build_cache_plans(cache))
    plan = agent.compact_state_plan
    assert sizes.compact_state == sum(plan.get_nbytes()) + 8 * 4
    assert sizes.total == sizes.target  # State bytes are NOT per-KV-block.
    free = 1 << 24
    assert executor._reserve_state_cache_mem([free], [sizes]) == [free - sizes.compact_state]
    with pytest.raises(ValueError, match='retained worker'):
        executor._get_state_cache_mem()
    monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: SimpleNamespace(
        current_context=lambda: SimpleNamespace(attn_tp_group=SimpleNamespace(rank=0))))
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *a, **k: object())
    monkeypatch.setattr(agent_module, 'build_kv_connector', lambda *a, **k: None)
    # StateCacheEngine and CompactStateRuntime use real CUDA allocations.
    agent.build_cache_engine()
    assert agent.state_cache_engine.compact_plan is plan
    assert agent.state_cache_engine.allocation.nbytes + agent.compact_runtime.current_ids.nbytes == sizes.compact_state
    assert agent.step_inputs.compact_runtime is agent.compact_runtime
    assert agent.make_dummy_meta.compact_bank_width == width + int(not allow_input_alias)
    assert agent.make_dummy_meta.compact_allow_input_alias == allow_input_alias
    assert draft_calls == ['allocate']
    assert agent.compact_runtime.current_ids.tolist() == [-1] * 4
    assert agent.state_cache_engine.state_caches[0].shape[0] == cache.compact_state_capacity.num_units
    assert cache.num_gpu_blocks == 32 and cache.num_state_caches == 4
    # A changed geometry must not allocate a different pool than the sized plan.
    agent.cache_config = replace(cache, compact_state_capacity=CompactStateCapacity(5, width))
    with pytest.raises(ValueError, match='exact retained'):
        agent.build_cache_engine()
    agent.cache_config = cache
    model.state_cache_specs = [replace(model.state_cache_specs[0], dtype=torch.float32), *model.state_cache_specs[1:]]
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *a, **k: pytest.fail('Must reject before KV allocation'))
    with pytest.raises(ValueError, match='declarations changed'):
        agent.build_cache_engine()


@pytest.mark.parametrize('case', ['device', 'family', 'paradigm', 'pd', 'memory',
                                  'auto_kv', 'capacity', 'method', 'draft_state', 'draft_layers', 'draft_compact'])
def test_unsupported_selection_rejects_before_worker_creation(case):
    model, cache, spec = _configs(4)
    executor = ExecutorBase.__new__(ExecutorBase)
    executor.cache_config, executor.model_config = cache, model
    executor.specdecode_config, executor.device_type = spec, 'cuda'
    executor.misc_config = MiscConfig()
    if case == 'device':
        executor.device_type = 'cpu'
    elif case == 'family':
        model.hf_config.model_type = 'unadapted'
    elif case == 'paradigm':
        model.model_paradigm = 'dllm'
    elif case == 'pd':
        cache.role = EngineRole.Prefill
    elif case == 'memory':
        executor.misc_config.memdecode_config = object()
    elif case == 'auto_kv':
        cache.num_gpu_blocks = 0
    elif case == 'capacity':
        cache.compact_state_capacity = CompactStateCapacity(3, 4)
    elif case == 'method':
        spec.method = 'dflash'
    elif case == 'draft_state':
        spec.model_config.states_shapes = [((2,), torch.float32)]
    elif case == 'draft_layers':
        spec.model_config.num_layers = 2
    else:
        spec.cache_config.compact_state_capacity = cache.compact_state_capacity
    with pytest.raises((ValueError, NotImplementedError)):
        executor._validate_compact_state_config()


def test_stateless_selection_has_no_capacity_or_worker_metadata():
    model, cache, _ = _configs()
    model.states_shapes, model.state_cache_specs = [], []
    cache.states_shapes = []
    cache.compact_state_capacity = CompactStateCapacity(0, 0)
    executor = _executor(model, cache)
    assert cache.compact_state_capacity is None
    assert cache.num_state_caches is None
    assert executor._get_state_cache_mem() == 0


@pytest.mark.parametrize('compact,store', [(True, True), (True, False), (False, True)])
def test_empty_init_selection_is_allowed_with_worker_reload_gate(compact, store):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs()
    if not compact:
        cache.compact_state_capacity = None
    if store:
        cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_consumer',
            kv_connector_extra_config=dict(compact_model_identity='immutable-target'))
    executor = ExecutorBase.__new__(ExecutorBase)
    executor.model_config, executor.cache_config = model, cache
    executor.specdecode_config, executor.device_type = spec, 'cuda'
    executor.misc_config = MiscConfig(empty_init=True)
    executor._validate_compact_state_config()


def test_deprecated_mp_rejects_before_process_or_pool_creation():
    model, cache, _ = _configs()
    with pytest.raises(NotImplementedError, match='deprecated MP'):
        MPExecutor('unused', model, cache, BackendConfig(), DistConfig(tp=2), MiscConfig())


def test_rank_local_compact_reservation_and_fixed_kv_never_silently_shrink():
    model, cache, _ = _configs()
    executor = _executor(model, cache)
    sizes = [_WorkerCachePlanSizes(64, 16, compact_state=128),
             _WorkerCachePlanSizes(96, 16, compact_state=256)]
    assert executor._reserve_state_cache_mem([1024, 2048], sizes) == [896, 1792]
    with pytest.raises(ValueError, match='every retained'):
        executor._reserve_state_cache_mem([1024, 2048], sizes[:1])
    with pytest.raises(ValueError, match='every retained'):
        executor._reserve_state_cache_mem([1024], [_WorkerCachePlanSizes(64)])
    with pytest.raises(RuntimeError, match='unchanged request'):
        executor._reserve_state_cache_mem([100, 2048], sizes)
    with pytest.raises(RuntimeError, match='baseline KV'):
        executor._update_num_gpu_blocks([32 * 80, 32 * 112 - 1], sizes, None)
    assert cache.num_gpu_blocks == 32
    executor._update_num_gpu_blocks([32 * 80, 32 * 112], sizes, None)
    assert cache.num_gpu_blocks == 32


def test_compact_sizing_cannot_silently_lower_prefill_budget(monkeypatch):
    model, cache, _ = _configs()
    executor = _executor(model, cache)
    before = cache.max_prefill_token_num
    monkeypatch.setattr(executor, '_get_runtime_size', lambda *a: (1024, before // 2))
    with pytest.raises(RuntimeError, match='configured prefill budget'):
        executor._reserve_runtime_mem([1 << 30], [_WorkerCachePlanSizes(16, compact_state=128)], None)
    assert cache.max_prefill_token_num == before


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('role', [None, 'kv_producer', 'kv_consumer', 'kv_both'])
def test_internal_startup_accepts_apc_store_without_shrinking_request_or_kv_capacity(width, apc, role):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(width)
    cache.enable_prefix_caching = apc
    cache.prefix_cache_state_budget = 2 if apc else 0
    if role is not None:
        cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role,
            kv_connector_extra_config=dict(compact_model_identity='immutable-target',
                                           compact_draft_identity='immutable-draft'))
        cache.mooncake_state_save_slots = 2
    cache.compact_state_capacity = CompactStateCapacity(4, width, cache.prefix_cache_state_budget,
                                                        cache.num_store_state_caches)
    _executor(model, cache, spec)
    assert cache.compact_state_capacity.request_slots == 4
    assert cache.compact_state_capacity.request_capacity == 3
    assert cache.num_gpu_blocks == 32
    assert cache.num_state_caches == 4 + cache.prefix_cache_state_budget + cache.num_store_state_caches
    assert cache.enable_prefix_caching == apc


@pytest.mark.parametrize('case', ['identity', 'draft_identity', 'cpu_paging', 'unknown_connector'])
def test_store_startup_rejects_unsafe_config_before_model_construction(case):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(4)
    extra = dict(compact_model_identity='immutable-target', compact_draft_identity='immutable-draft')
    if case == 'identity':
        del extra['compact_model_identity']
    if case == 'draft_identity':
        extra['compact_draft_identity'] = ' '
    cache.kv_transfer_config = KVTransferConfig(
        kv_connector='UnknownConnector' if case == 'unknown_connector' else 'MooncakeStoreConnector',
        kv_role='kv_consumer', kv_connector_extra_config=extra)
    if case == 'cpu_paging':
        cache.num_cpu_blocks = 1
    with pytest.raises((ValueError, NotImplementedError), match='Compact'):
        _executor(model, cache, spec)


@pytest.mark.parametrize('role', ['kv_producer', 'kv_consumer', 'kv_both'])
@pytest.mark.parametrize('width', [1, 4])
def test_initial_executor_uses_model_statefulness_before_cache_shapes_are_populated(role, width):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(width)
    cache.states_shapes = []  # Real ConfigBuilder has not copied model fields yet.
    cache.mooncake_state_save_slots = 2
    cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role,
        kv_connector_extra_config=dict(compact_model_identity='target-rev', compact_draft_identity='draft-rev'))
    credits = 0 if role == 'kv_consumer' else 2
    cache.compact_state_capacity = CompactStateCapacity(4, width, 0, credits)
    executor = _executor(model, cache, spec)
    assert cache.num_state_caches == 4 + credits
    cache.states_shapes = model.states_shapes
    executor._validate_compact_state_config()
    assert cache.num_state_caches == 4 + credits


@pytest.mark.parametrize('family', ['glm_moe_dsa', 'qwen3_5', 'glm5_next'])
@pytest.mark.parametrize('store', [False, True])
def test_stateless_worker_build_allocates_no_state_ids_or_compact_schema(monkeypatch, family, store):
    from lmdeploy.pytorch import compact_state

    model, cache, _ = _configs(family=family)
    if store:
        from lmdeploy.messages import KVTransferConfig
        cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both')
    model.states_shapes, model.state_cache_specs = [], []
    cache.states_shapes = []
    cache.compact_state_capacity = CompactStateCapacity(0, 0)
    _executor(model, cache)
    assert cache.compact_state_capacity is None
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.rank, agent.device = 0, 'cuda'
    agent.dist_config = DistConfig()
    agent.model_config, agent.cache_config = model, cache
    agent.patched_model = torch.nn.Module()
    agent.memdecode_agent = agent.kv_connector = agent.compact_runtime = None
    agent.cache_stream = None
    agent.step_inputs = SimpleNamespace(compact_runtime=None)
    agent.make_dummy_meta = SimpleNamespace(compact_bank_width=0, compact_allow_input_alias=False)
    draft_calls = []
    agent.spec_agent = SimpleNamespace(build_cache_plan=lambda _: 0,
                                       build_cache_engine=lambda _: draft_calls.append('allocate'),
                                       cache_engine=None, specdecode_config=None)
    # Leave worker plan selection and StateCacheEngine real. Only unrelated KV
    # allocation and distributed process context are replaced; no CUDA required.
    def forbidden(*args, **kwargs):
        pytest.fail('A stateless worker must not create state storage, ID tables or state kernels')

    monkeypatch.setattr(compact_state, 'CompactStateRuntime', forbidden)
    sizes = agent.build_cache_plans(cache)
    assert len(sizes) == 3 and agent.compact_state_plan is None
    registrations = []
    connector = SimpleNamespace(register_kv_caches=lambda *args, **kwargs: registrations.append((args, kwargs)))
    monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: SimpleNamespace(
        current_context=lambda: SimpleNamespace(attn_tp_group=SimpleNamespace(rank=0))))
    kv = {'layer.k': object(), 'layer.v': object()}
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *args, **kwargs: SimpleNamespace(connector_kv_caches=kv))
    monkeypatch.setattr(agent_module, 'build_kv_connector', lambda *args, **kwargs: connector)
    # Legacy empty-layout selection is allowed; tensor allocation is not.
    monkeypatch.setattr(torch, 'empty', forbidden)
    monkeypatch.setattr(torch, 'zeros', forbidden)
    monkeypatch.setattr(torch, 'full', forbidden)
    agent.build_cache_engine()
    assert agent.state_cache_engine.allocation.nbytes == 0
    assert not agent.state_cache_engine.state_caches
    assert not agent.state_cache_engine.connector_state_cache_pools
    assert agent.compact_runtime is agent.step_inputs.compact_runtime is None
    assert agent.make_dummy_meta.compact_bank_width == 0
    assert not agent.make_dummy_meta.compact_allow_input_alias
    assert draft_calls == ['allocate']
    assert len(registrations) == 1
    args, kwargs = registrations[0]
    assert args == (kv,)
    assert not kwargs['state_cache_pools']
    assert 'state_cache_schema' not in kwargs
