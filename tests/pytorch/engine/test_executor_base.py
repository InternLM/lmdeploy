# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import PytorchEngineConfig, QuantPolicy
from lmdeploy.pytorch.config import CacheConfig, StateCacheSpec
from lmdeploy.pytorch.configurations.deepseek_v4 import update_cache_config as update_deepseek_v4_cache_config
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy
from lmdeploy.pytorch.engine.executor import base as executor_base
from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _WorkerCachePlanSizes
from lmdeploy.pytorch.engine.executor.uni_executor import UniExecutor


class _RecordingExecutor(ExecutorBase):

    def __init__(self, empty_init: bool):
        super().__init__(
            model_path='',
            model_config=SimpleNamespace(sliding_window=None, states_shapes=None),
            cache_config=SimpleNamespace(role=EngineRole.Hybrid),
            backend_config=SimpleNamespace(),
            dist_config=SimpleNamespace(dp=1, world_size=1),
            misc_config=SimpleNamespace(empty_init=empty_init, memdecode_config=None),
        )
        self.calls = []

    def build_model(self):
        self.calls.append('build_model')

    def update_configs(self):
        self.calls.append('update_configs')

    def build_graph_runner(self):
        self.calls.append('build_graph_runner')

    def build_cache_engine(self):
        self.calls.append('build_cache_engine')

    def warmup(self):
        self.calls.append('warmup')


def test_finalize_sparse_mla_cache_policy_before_executor_build():
    model_config = SimpleNamespace(mla_index_topk=2048, mla_kv_cache_dtype='bfloat16')
    cache_config = SimpleNamespace(quant_policy=QuantPolicy.FP8)

    _finalize_sparse_mla_cache_policy([model_config], cache_config)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'
    assert cache_config.quant_policy == QuantPolicy.NONE


@pytest.mark.parametrize('quant_policy',
                         [QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.FP8_E5M2, QuantPolicy.TURBO_QUANT])
def test_finalize_sparse_mla_cache_policy_rejects_other_quantization(quant_policy):
    model_config = SimpleNamespace(mla_index_topk=2048, mla_kv_cache_dtype='bfloat16')
    cache_config = SimpleNamespace(quant_policy=quant_policy)

    with pytest.raises(ValueError, match='Sparse MLA does not support quant_policy'):
        _finalize_sparse_mla_cache_policy([model_config], cache_config)


def test_init_warms_up_model_by_default():
    executor = _RecordingExecutor(empty_init=False)

    executor.init()

    assert executor.calls == [
        'build_model',
        'update_configs',
        'build_graph_runner',
        'build_cache_engine',
        'warmup',
    ]


def test_init_skips_model_warmup_for_empty_init():
    executor = _RecordingExecutor(empty_init=True)

    executor.init()

    assert executor.calls == [
        'build_model',
        'update_configs',
        'build_graph_runner',
        'build_cache_engine',
    ]


def test_get_num_gpu_blocks_without_spec_cache():
    available_mem = 4096
    cache_block_size = 256

    num_gpu_blocks = ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size)

    assert num_gpu_blocks == 16


def test_get_num_gpu_blocks_with_spec_cache():
    available_mem = 4096
    cache_block_size = 256
    spec_cache_block_size = 256

    num_gpu_blocks = ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size, spec_cache_block_size)

    assert num_gpu_blocks == 8


def test_get_num_gpu_blocks_rejects_empty_cache_block():
    with pytest.raises(RuntimeError, match='No enough gpu memory for kv cache.'):
        ExecutorBase._get_num_gpu_blocks(available_mem=4096, cache_block_size=0)


def test_dsa_score_workspace_uses_configured_logits_budget(monkeypatch):
    monkeypatch.setattr(executor_base._envs, 'dsa_indexer_max_logits_mb', 7)
    executor = object.__new__(ExecutorBase)
    executor.model_config = SimpleNamespace(mla_index_topk=2048)

    assert executor._get_dsa_score_workspace_size() == 7 << 20

    executor.model_config.mla_index_topk = None
    assert executor._get_dsa_score_workspace_size() == 0


def test_runtime_size_reserves_dsa_score_workspace(monkeypatch):
    monkeypatch.setattr(executor_base._envs, 'dsa_indexer_max_logits_mb', 1)
    executor = object.__new__(ExecutorBase)
    executor.model_config = SimpleNamespace(mla_index_topk=2048)
    executor.cache_config = SimpleNamespace(
        cache_max_entry_count=1.0,
        max_prefill_token_num=16,
        max_batches=2,
    )
    executor.specdecode_config = None

    runtime_size, max_prefill_token_num = executor._get_runtime_size(
        [4 << 20], [_WorkerCachePlanSizes(target=1024)], vocab_size=100)

    generic_runtime_size = (16 + 2 * 2) * 100 * 2
    assert runtime_size == (1 << 20) + generic_runtime_size
    assert max_prefill_token_num == 16


def test_get_min_num_gpu_blocks_rejects_worker_count_mismatch():
    with pytest.raises(ValueError, match='same worker ranks'):
        ExecutorBase._get_min_num_gpu_blocks([4096, 4096], [256])


def test_sync_spec_cache_block_size_updates_kernel_block_size():
    executor = object.__new__(ExecutorBase)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=32,
                                        kernel_block_size=16,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0)
    spec_cache_config = CacheConfig(max_batches=1,
                                    block_size=64,
                                    kernel_block_size=64,
                                    num_cpu_blocks=0,
                                    num_gpu_blocks=0)
    executor.specdecode_config = SimpleNamespace(cache_config=spec_cache_config)

    executor._sync_spec_cache_block_size()

    assert spec_cache_config.block_size == 32
    assert spec_cache_config.kernel_block_size == 16


def test_adjust_block_size_uses_deepseek_v4_hook_and_syncs_model_config():
    executor = object.__new__(ExecutorBase)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=192,
                                        kernel_block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0)
    executor.model_config = SimpleNamespace(block_size=192,
                                            k_head_dim=512,
                                            use_flash_mla=False,
                                            update_cache_config_func=update_deepseek_v4_cache_config)

    executor._adjust_block_size()

    assert executor.cache_config.block_size == 256
    assert executor.cache_config.kernel_block_size == 256
    assert executor.cache_config.window_size == -1
    assert executor.model_config.block_size == 256


def test_executor_disables_prefix_cache_with_generic_sliding_window():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               enable_prefix_caching=True)
    model_config = SimpleNamespace(sliding_window=4096, update_cache_config_func=None)

    ExecutorBase(model_path='',
                 model_config=model_config,
                 cache_config=cache_config,
                 backend_config=SimpleNamespace(),
                 dist_config=SimpleNamespace(dp=1, world_size=1),
                 misc_config=SimpleNamespace())

    assert cache_config.window_size == 4096
    assert not cache_config.enable_prefix_caching


def test_executor_keeps_prefix_cache_after_deepseek_v4_window_normalization():
    cache_config = CacheConfig(max_batches=1,
                               block_size=192,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               enable_prefix_caching=True)
    model_config = SimpleNamespace(sliding_window=4096,
                                   update_cache_config_func=update_deepseek_v4_cache_config)

    executor = ExecutorBase(model_path='',
                            model_config=model_config,
                            cache_config=cache_config,
                            backend_config=SimpleNamespace(),
                            dist_config=SimpleNamespace(dp=1, world_size=1),
                            misc_config=SimpleNamespace())

    assert cache_config.enable_prefix_caching

    executor._adjust_block_size()
    executor._maybe_disable_unsupported_prefix_caching()

    assert cache_config.block_size == 256
    assert cache_config.kernel_block_size == 256
    assert cache_config.window_size == -1
    assert cache_config.enable_prefix_caching


def test_executor_keeps_prefix_cache_with_spec_decode():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               enable_prefix_caching=True)
    model_config = SimpleNamespace(sliding_window=None)

    ExecutorBase(model_path='',
                 model_config=model_config,
                 cache_config=cache_config,
                 backend_config=SimpleNamespace(),
                 dist_config=SimpleNamespace(dp=1, world_size=1),
                 misc_config=SimpleNamespace(),
                 specdecode_config=SimpleNamespace())

    assert cache_config.enable_prefix_caching


def test_executor_disables_prefix_cache_with_pd_role():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               enable_prefix_caching=True,
                               role=EngineRole.Prefill)
    model_config = SimpleNamespace(sliding_window=None)

    ExecutorBase(model_path='',
                 model_config=model_config,
                 cache_config=cache_config,
                 backend_config=SimpleNamespace(),
                 dist_config=SimpleNamespace(dp=1, world_size=1),
                 misc_config=SimpleNamespace())

    assert not cache_config.enable_prefix_caching


def test_get_rank_cache_block_sizes_uses_worker_local_plans():
    plans = [
        _WorkerCachePlanSizes(target=256, spec=128),
        _WorkerCachePlanSizes(target=256),
        _WorkerCachePlanSizes(target=192, spec=96, memory=64),
    ]

    cache_block_sizes = ExecutorBase._get_rank_cache_block_sizes(plans)

    assert cache_block_sizes == [384, 256, 352]


def test_uni_executor_prepares_named_worker_cache_plan_sizes():
    executor = object.__new__(UniExecutor)
    cache_config = object()
    spec_cache_config = object()
    calls = []

    def build_cache_plans(received_cache_config, received_spec_cache_config):
        calls.append((received_cache_config, received_spec_cache_config))
        return 256, 128, 64

    executor.model_agent = SimpleNamespace(build_cache_plans=build_cache_plans)

    cache_block_sizes = executor._prepare_worker_cache_plans(cache_config, spec_cache_config)

    assert cache_block_sizes == [
        _WorkerCachePlanSizes(target=256, spec=128, memory=64),
    ]
    assert calls == [(cache_config, spec_cache_config)]


def test_validate_memdecode_configs_rejects_specdecode():
    executor = object.__new__(ExecutorBase)
    executor.misc_config = SimpleNamespace(
        memdecode_config=SimpleNamespace(memory_model_config=SimpleNamespace()))
    executor.specdecode_config = SimpleNamespace()

    with pytest.raises(ValueError, match='MemDecode and speculative decoding cannot be enabled together.'):
        executor._validate_memdecode_configs()


def test_validate_memdecode_configs_rejects_state_cache_mismatch():
    executor = object.__new__(ExecutorBase)
    executor.model_config = SimpleNamespace(states_shapes=[])
    executor.misc_config = SimpleNamespace(
        memdecode_config=SimpleNamespace(memory_model_config=SimpleNamespace(states_shapes=[(1, 2, 3)])))
    executor.specdecode_config = None

    with pytest.raises(ValueError, match='Base and memory model must both use SSM state caches or both not use them.'):
        executor._validate_memdecode_configs()


def test_update_num_gpu_blocks_can_be_limited_by_non_spec_rank():
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=2)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        cache_max_entry_count=1.0)
    spec_cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)

    cache_block_sizes = [
        _WorkerCachePlanSizes(target=256, spec=256),
        _WorkerCachePlanSizes(target=256),
    ]
    executor._update_num_gpu_blocks([2048, 768], cache_block_sizes, spec_cache_config)

    assert executor.cache_config.num_gpu_blocks == 3
    assert spec_cache_config.num_gpu_blocks == 3


def test_get_state_cache_mem_uses_prefix_cache_state_budget():
    executor = object.__new__(ExecutorBase)
    state_shapes = [((2, ), torch.float32)]
    executor.cache_config = CacheConfig(max_batches=4,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        states_shapes=state_shapes,
                                        prefix_cache_state_budget=3)

    mem = executor._get_state_cache_mem()

    expected_num_state_caches = 4 + 2 + 3
    expected_mem = StateCacheEngine.get_state_slot_nbytes(state_shapes) * expected_num_state_caches
    assert executor.cache_config.num_state_caches == expected_num_state_caches
    assert mem == expected_mem


def test_get_mem_state_cache_mem_uses_memory_model_state_specs():
    executor = object.__new__(ExecutorBase)
    state_specs = [StateCacheSpec('memory_state', (96, ), torch.float32, layer_ids=[1, 3])]
    state_shapes = [(spec.shape, spec.dtype) for spec in state_specs]
    memory_model_config = SimpleNamespace(states_shapes=state_shapes, state_cache_specs=state_specs)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        num_state_caches=2)
    executor.misc_config = SimpleNamespace(
        memdecode_config=SimpleNamespace(memory_model_config=memory_model_config))

    mem = executor._get_mem_state_cache_mem()

    expected = StateCacheEngine.get_state_slot_nbytes(state_shapes, state_specs=state_specs) * 2
    assert mem == expected


def test_get_state_cache_mem_keeps_ssm_prefix_cache_enabled_without_extra_budget():
    executor = object.__new__(ExecutorBase)
    state_shapes = [((2, ), torch.float32)]
    executor.cache_config = CacheConfig(max_batches=4,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        states_shapes=state_shapes,
                                        enable_prefix_caching=True,
                                        prefix_cache_state_budget=0)

    executor._get_state_cache_mem()

    assert executor.cache_config.num_state_caches == 4 + 2
    assert executor.cache_config.enable_prefix_caching


def test_get_state_cache_mem_keeps_budgeted_ssm_prefix_cache_enabled():
    executor = object.__new__(ExecutorBase)
    state_shapes = [((2, ), torch.float32)]
    executor.cache_config = CacheConfig(max_batches=4,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        states_shapes=state_shapes,
                                        enable_prefix_caching=True,
                                        prefix_cache_state_budget=2)

    executor._get_state_cache_mem()

    assert executor.cache_config.num_state_caches == 4 + 2 + 2
    assert executor.cache_config.enable_prefix_caching


def test_get_state_cache_mem_leaves_non_ssm_prefix_cache_enabled():
    executor = object.__new__(ExecutorBase)
    executor.cache_config = CacheConfig(max_batches=4,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        enable_prefix_caching=True,
                                        prefix_cache_state_budget=0)

    mem = executor._get_state_cache_mem()

    assert mem == 0
    assert executor.cache_config.enable_prefix_caching


def test_build_cache_config_carries_prefix_cache_state_budget():
    engine_config = PytorchEngineConfig(max_batch_size=4,
                                        prefix_cache_state_budget=3,
                                        prefix_cache_decode_state_interval=128)

    cache_config = ConfigBuilder.build_cache_config(engine_config)

    assert cache_config.prefix_cache_state_budget == 3
    assert cache_config.prefix_cache_decode_state_interval == 128


def test_engine_config_rejects_unaligned_prefix_cache_decode_state_interval():
    with pytest.raises(AssertionError):
        PytorchEngineConfig(max_batch_size=4, prefix_cache_decode_state_interval=96)
