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
    DistConfig,
    MiscConfig,
    ModelConfig,
    SSMStateCapacity,
    StateCacheSpec,
)
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _WorkerCachePlanSizes
from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
from lmdeploy.pytorch.engine.model_agent import agent as agent_module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def _configs(width=1, family='qwen3_5'):
    specs = [StateCacheSpec('state_0', (2, 8, 3), torch.bfloat16),
             StateCacheSpec('state_1', (2, 2, 4, 4), torch.float32)]
    model = ModelConfig(hidden_size=16, num_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         bos_token_id=1, eos_token_id=[2], head_dim=8, k_head_dim=8, is_gated_delta=True,
                         state_cache_is_indexed=True, state_cache_specs=specs,
                         states_shapes=[(s.shape, s.dtype) for s in specs],
                         hf_config=SimpleNamespace(model_type=family), model_paradigm='ar' if width == 1 else 'ar_spec')
    cache = CacheConfig(max_batches=2, block_size=16, num_cpu_blocks=0, num_gpu_blocks=32,
                         states_shapes=model.states_shapes, compact_state_capacity=SSMStateCapacity(4, width))
    spec = None if width == 1 else SimpleNamespace(
        method='deepseek_mtp' if family == 'glm5_next' else 'qwen3_5_mtp', num_speculative_tokens=width - 1,
        model_config=SimpleNamespace(states_shapes=[], num_layers=1, mla_index_topk=None), cache_config=replace(cache,
                                                                                          compact_state_capacity=None))
    return model, cache, spec


def _executor(model, cache, spec=None, **kwargs):
    return ExecutorBase('unused', model, cache, BackendConfig(), DistConfig(), MiscConfig(),
                         specdecode_config=spec, **kwargs)


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('family', ['qwen3_5', 'glm5_next'])
@pytest.mark.parametrize('from_factory', [False, True])
def test_worker_retains_exact_plan_for_sizing_real_allocation_and_dummy(
        monkeypatch, width, family,  from_factory):
    model, cache, spec = _configs(width, family)
    cache.compact_state_capacity = replace(cache.compact_state_capacity)
    if from_factory:
        cache.compact_state_capacity = None
        executor = _build_from_factory(monkeypatch, model, cache, spec)
    else:
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
    agent.step_inputs = SimpleNamespace(ssm_state_cursor=None)
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
    # StateCacheEngine and SSMStateCursor use real CUDA allocations.
    agent.build_cache_engine()
    assert agent.state_cache_engine.compact_plan is plan
    assert agent.state_cache_engine.allocation.nbytes + agent.ssm_state_cursor.current_ids.nbytes == sizes.compact_state
    assert agent.step_inputs.ssm_state_cursor is agent.ssm_state_cursor
    assert agent.make_dummy_meta.compact_bank_width == width + 0
    assert draft_calls == ['allocate']
    assert agent.ssm_state_cursor.current_ids.tolist() == [-1] * 4
    assert agent.state_cache_engine.state_caches[0].shape[0] == cache.compact_state_capacity.num_units
    assert cache.num_gpu_blocks == 32 and cache.num_state_caches == 4
    # A changed geometry must not allocate a different pool than the sized plan.
    agent.cache_config = replace(cache, compact_state_capacity=SSMStateCapacity(5, width))
    with pytest.raises(ValueError, match='exact retained'):
        agent.build_cache_engine()
    agent.cache_config = cache
    model.state_cache_specs = [replace(model.state_cache_specs[0], dtype=torch.float32), *model.state_cache_specs[1:]]
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *a, **k: pytest.fail('Must reject before KV allocation'))
    with pytest.raises(ValueError, match='model state bindings'):
        agent.build_cache_engine()


@pytest.mark.parametrize('case', ['device', 'family', 'paradigm', 'pd', 'memory',
                                  'negative_kv', 'capacity', 'method', 'draft_state', 'draft_layers', 'draft_compact'])
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
        model.state_cache_is_indexed = False
        cache.compact_state_capacity = SSMStateCapacity(4, 1)
    elif case == 'paradigm':
        model.model_paradigm = 'dllm'
    elif case == 'pd':
        cache.role = EngineRole.Prefill
    elif case == 'memory':
        executor.misc_config.memdecode_config = object()
    elif case == 'negative_kv':
        cache.num_gpu_blocks = -1
    elif case == 'capacity':
        cache.compact_state_capacity = SSMStateCapacity(3, 4)
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
    cache.compact_state_capacity = SSMStateCapacity(0, 0)
    executor = _executor(model, cache)
    assert cache.compact_state_capacity is None
    assert cache.num_state_caches is None
    assert executor._get_state_cache_mem() == 0


@pytest.mark.parametrize('compact,store', [(True, True), (True, False), (False, True)])
def test_empty_init_selection_uses_baseline_weight_lifecycle(compact, store):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs()
    if not compact:
        cache.compact_state_capacity = None
    if store:
        cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_consumer',
            kv_connector_extra_config={})
    executor = ExecutorBase.__new__(ExecutorBase)
    executor.model_config, executor.cache_config = model, cache
    executor.specdecode_config, executor.device_type = spec, 'cuda'
    executor.misc_config = MiscConfig(empty_init=True)
    executor._validate_compact_state_config()


@pytest.mark.parametrize('factory', [False, True])
def test_deprecated_mp_rejects_before_process_or_pool_creation(monkeypatch, factory):
    model, cache, _ = _configs()
    with pytest.raises(NotImplementedError, match='deprecated MP'):
        if factory:
            _build_from_factory(monkeypatch, model, cache, backend='mp')
        else:
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
            kv_connector_extra_config={})
        cache.mooncake_state_save_slots = 2
    cache.compact_state_capacity = SSMStateCapacity(4, width, cache.prefix_cache_state_budget,
                                                        cache.num_store_state_caches)
    _executor(model, cache, spec)
    assert cache.compact_state_capacity.request_slots == 4
    assert cache.compact_state_capacity.request_capacity == 3
    assert cache.num_gpu_blocks == 32
    assert cache.num_state_caches == 4 + cache.prefix_cache_state_budget + cache.num_store_state_caches
    assert cache.enable_prefix_caching == apc


@pytest.mark.parametrize('case', ['cpu_paging', 'unknown_connector'])
def test_store_startup_rejects_unsafe_config_before_model_construction(case):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(4)
    extra = {}
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
        kv_connector_extra_config={})
    credits = 0 if role == 'kv_consumer' else 2
    cache.compact_state_capacity = SSMStateCapacity(4, width, 0, credits)
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
    cache.compact_state_capacity = SSMStateCapacity(0, 0)
    _executor(model, cache)
    assert cache.compact_state_capacity is None
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.rank, agent.device = 0, 'cuda'
    agent.dist_config = DistConfig()
    agent.model_config, agent.cache_config = model, cache
    agent.patched_model = torch.nn.Module()
    agent.memdecode_agent = agent.kv_connector = agent.ssm_state_cursor = None
    agent.cache_stream = None
    agent.step_inputs = SimpleNamespace(ssm_state_cursor=None)
    agent.make_dummy_meta = SimpleNamespace(compact_bank_width=0)
    draft_calls = []
    agent.spec_agent = SimpleNamespace(build_cache_plan=lambda _: 0,
                                       build_cache_engine=lambda _: draft_calls.append('allocate'),
                                       cache_engine=None, specdecode_config=None)
    # Leave worker plan selection and StateCacheEngine real. Only unrelated KV
    # allocation and distributed process context are replaced; no CUDA required.
    def forbidden(*args, **kwargs):
        pytest.fail('A stateless worker must not create state storage, ID tables or state kernels')

    monkeypatch.setattr(compact_state, 'SSMStateCursor', forbidden)
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
    assert agent.ssm_state_cursor is agent.step_inputs.ssm_state_cursor is None
    assert agent.make_dummy_meta.compact_bank_width == 0
    assert draft_calls == ['allocate']
    assert len(registrations) == 1
    args, kwargs = registrations[0]
    assert args == (kv,)
    assert not kwargs['state_cache_pools']
    assert 'state_cache_schema' not in kwargs


def _build_from_factory(monkeypatch, model, cache, spec=None, backend='uni', device='cuda'):
    """Use normal executor selection; only replace model loading and
    workers."""
    from lmdeploy.pytorch.engine.executor import build_executor, uni_executor

    monkeypatch.setattr(ModelConfig, 'from_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(uni_executor, 'UniExecutor', lambda **kw: ExecutorBase(dist_config=DistConfig(), **kw))
    return build_executor('unused', cache, BackendConfig(), DistConfig(), MiscConfig(),
                          device_type=device, distributed_executor_backend=backend, specdecode_config=spec)


@pytest.mark.parametrize('value', [None, '0', '1', 'invalid'])
def test_removed_compact_env_does_not_select_a_second_implementation(monkeypatch, value):
    import runpy

    from lmdeploy.pytorch import envs
    monkeypatch.delenv('LMDEPLOY_ENABLE_COMPACT_STATE', raising=False)
    if value is not None:
        monkeypatch.setenv('LMDEPLOY_ENABLE_COMPACT_STATE', value)
    assert 'enable_compact_state' not in runpy.run_path(envs.__file__)
    model, cache, _ = _configs()
    cache.compact_state_capacity = None
    _build_from_factory(monkeypatch, model, cache)
    assert cache.compact_state_capacity is not None


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('family', ['qwen3_5', 'qwen3_5_moe', 'glm5_next'])
@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('role', [None, 'kv_producer', 'kv_consumer', 'kv_both'])
def test_indexed_model_selects_normal_startup_geometry(monkeypatch, width, family, apc, role):
    from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder

    model, _, spec = _configs(width, family)
    transfer = None if role is None else KVTransferConfig(
        kv_connector='MooncakeStoreConnector', kv_role=role,
        kv_connector_extra_config={})
    cache = ConfigBuilder.build_cache_config(PytorchEngineConfig(
        max_batch_size=2, num_gpu_blocks=32, enable_prefix_caching=apc,
        prefix_cache_state_budget=2 if apc else 0, mooncake_state_save_slots=3, kv_transfer_config=transfer))
    assert not cache.states_shapes and cache.compact_state_capacity is None
    _build_from_factory(monkeypatch, model, cache, spec)
    store = 3 if role in ('kv_producer', 'kv_both') else 0
    assert cache.compact_state_capacity == SSMStateCapacity(4, width, 2 if apc else 0, store)
    assert cache.num_state_caches == 4 + (2 if apc else 0) + store
    assert cache.num_gpu_blocks == 32
    assert spec is None or spec.cache_config.compact_state_capacity is None


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('stateful', [False, True])
def test_model_layout_and_stateless_control_allocation(monkeypatch, indexed, stateful):

    model, cache, _ = _configs()
    model.state_cache_is_indexed = indexed
    if not stateful:
        model.states_shapes, model.state_cache_specs, cache.states_shapes = [], [], []
    cache.compact_state_capacity = None
    _build_from_factory(monkeypatch, model, cache)
    assert (cache.compact_state_capacity is not None) == (indexed and stateful)


@pytest.mark.parametrize('case', ['device', 'family', 'mp', 'negative_kv'])
def test_factory_keeps_supported_execution_guards(monkeypatch, case):

    model, cache, _ = _configs()
    cache.compact_state_capacity = None
    if case == 'family':
        model.hf_config.model_type = 'unadapted'
        model.state_cache_is_indexed = False
        cache.compact_state_capacity = SSMStateCapacity(4, 1)
    elif case == 'negative_kv':
        cache.num_gpu_blocks = -1
    with pytest.raises((ValueError, NotImplementedError), match='Compact'):
        _build_from_factory(monkeypatch, model, cache, backend='mp' if case == 'mp' else 'uni',
                        device='cpu' if case == 'device' else 'cuda')


def test_factory_preserves_explicit_internal_geometry(monkeypatch):

    model, cache, _ = _configs()
    cache.num_state_caches = 6
    cache.compact_state_capacity = None
    _build_from_factory(monkeypatch, model, cache)
    assert cache.compact_state_capacity == SSMStateCapacity(6, 1)
    # Explicit geometry preserves the same fixed production contract.
    cache.compact_state_capacity = SSMStateCapacity(6, 1)
    _build_from_factory(monkeypatch, model, cache)


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('ratio', [0.5, 0.7, 0.9])
@pytest.mark.parametrize('limiting_rank', [0, 1])
def test_compact_auto_kv_reserves_state_and_runtime_before_fitting_target_and_draft(
        monkeypatch, width, ratio, limiting_rank):

    model, cache, spec = _configs(width, 'glm5_next')
    model.vocab_size = 128
    cache.num_gpu_blocks = 0
    cache.compact_state_capacity = None
    cache.cache_max_entry_count = ratio
    cache.max_prefill_token_num = 64
    if spec is not None:
        spec.cache_config.num_gpu_blocks = 0
    executor = _build_from_factory(monkeypatch, model, cache, spec)
    capacity = cache.compact_state_capacity
    state_slots = cache.num_state_caches
    # Simulate retained rank-local plans, including auxiliary state/ID bytes.
    plans = [_WorkerCachePlanSizes(1024, 256 if spec else 0, compact_state=4096),
             _WorkerCachePlanSizes(2048, 512 if spec else 0, compact_state=6144)]
    free_mems = [1 << 23, 1 << 23]
    free_mems[limiting_rank] = 1 << 20
    monkeypatch.setattr(executor, '_prepare_worker_cache_plans', lambda *a: plans)
    monkeypatch.setattr(executor, 'gather_free_mem', lambda: free_mems)
    updates = []
    monkeypatch.setattr(executor, 'set_cache_config', lambda *args: updates.append(args))
    monkeypatch.setattr(executor, 'set_model_config', lambda *args: None)
    executor.update_configs()
    runtime_bytes = (64 + cache.max_batches * 2) * model.vocab_size * 2
    per_rank = [int((free - plan.compact_state - runtime_bytes) * ratio) // plan.total
                for free, plan in zip(free_mems, plans)]
    assert cache.num_gpu_blocks == min(per_rank) == per_rank[limiting_rank]
    assert cache.compact_state_capacity == capacity and cache.num_state_caches == state_slots
    assert cache.max_batches == 2 and cache.max_prefill_token_num == 64
    assert updates == [(cache, None if spec is None else spec.cache_config)]
    if spec is not None:
        assert spec.cache_config.num_gpu_blocks == cache.num_gpu_blocks
        assert spec.cache_config.max_prefill_token_num == 64
    # Finalized configs still satisfy validation; geometry cannot depend on KV count.
    executor._validate_compact_state_config()


@pytest.mark.parametrize('shortage', ['state', 'kv'])
def test_compact_auto_kv_insufficient_memory_does_not_reduce_request_capacity(monkeypatch, shortage):

    model, cache, spec = _configs(6)
    model.vocab_size = 128
    cache.num_gpu_blocks = spec.cache_config.num_gpu_blocks = 0
    cache.compact_state_capacity = None
    cache.max_prefill_token_num = 64
    executor = _build_from_factory(monkeypatch, model, cache, spec)
    capacity = cache.compact_state_capacity
    plans = [_WorkerCachePlanSizes(1024, 256, compact_state=4096)]
    runtime_bytes = (64 + cache.max_batches * 2) * model.vocab_size * 2
    free = 4096 if shortage == 'state' else 4096 + runtime_bytes + 1
    monkeypatch.setattr(executor, '_prepare_worker_cache_plans', lambda *a: plans)
    monkeypatch.setattr(executor, 'gather_free_mem', lambda: [free])
    monkeypatch.setattr(executor, 'set_cache_config', lambda *a: pytest.fail('Must reject before worker allocation'))
    with pytest.raises(RuntimeError, match='unchanged request capacity' if shortage == 'state' else 'kv cache'):
        executor.update_configs()
    assert cache.compact_state_capacity == capacity
    assert cache.max_batches == 2 and cache.max_prefill_token_num == 64
    assert cache.num_gpu_blocks == spec.cache_config.num_gpu_blocks == 0


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('role', [None, 'kv_producer', 'kv_consumer', 'kv_both'])
@pytest.mark.parametrize('state_slots', [None, 17])
@pytest.mark.parametrize('from_factory', [False, True])
def test_compact_capacity_defaults_match_factory_and_direct_executor(
        monkeypatch, width, role, state_slots, from_factory):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(width)
    cache.states_shapes = []  # Not populated until update_configs; Store budget must still count.
    cache.num_state_caches = state_slots
    cache.prefix_cache_state_budget = 2
    cache.mooncake_state_save_slots = 3
    if role is not None:
        cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role,
                                                   kv_connector_extra_config={})
    store = 3 if role in ('kv_producer', 'kv_both') else 0
    slots = 4 + 2 + store if state_slots is None else state_slots
    expected = SSMStateCapacity(slots - 2 - store, width, 2, store)
    cache.compact_state_capacity = None if from_factory else expected
    if from_factory:
        executor = _build_from_factory(monkeypatch, model, cache, spec)
    else:
        executor = _executor(model, cache, spec)
    capacity = cache.compact_state_capacity
    assert capacity == expected and cache.num_state_caches == slots
    executor._validate_compact_state_config()
    assert cache.compact_state_capacity is capacity and cache.num_state_caches == slots


@pytest.mark.parametrize('field', ['num_state_caches', 'prefix_cache_state_budget',
                                   'mooncake_state_save_slots', 'num_speculative_tokens'])
def test_config_update_hook_cannot_change_retained_compact_capacity(monkeypatch, field):
    from lmdeploy.messages import KVTransferConfig

    model, cache, spec = _configs(6)
    cache.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_producer',
                                               kv_connector_extra_config={})
    cache.compact_state_capacity = None
    executor = _build_from_factory(monkeypatch, model, cache, spec)
    capacity = cache.compact_state_capacity

    def update(config):
        target = spec if field == 'num_speculative_tokens' else config
        setattr(target, field, getattr(target, field) + 1)

    model.update_cache_config_func = update
    monkeypatch.setattr(executor, '_prepare_worker_cache_plans',
                        lambda *a: pytest.fail('Changed capacity must fail before worker plan sizing'))
    with pytest.raises(ValueError, match='preserve configured M/B/S'):
        executor.update_configs()
    assert cache.compact_state_capacity is capacity
