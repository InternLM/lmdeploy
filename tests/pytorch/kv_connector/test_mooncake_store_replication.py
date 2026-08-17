# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.configurations.deepseek_v2 import DeepseekV2ModelConfigBuilder
from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder
from lmdeploy.pytorch.configurations.glm_moe_dsa import GlmMoeDsaModelConfigBuilder
from lmdeploy.pytorch.configurations.llava_hf import LlavaHfModelConfigBuilder
from lmdeploy.pytorch.engine.executor.base import ExecutorBase
from lmdeploy.pytorch.kv_connector import KVConnectorRole, build_kv_connector
from lmdeploy.pytorch.kv_connector.mooncake.store import connector as connector_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    MOONCAKE_BLOCK_HASH_BYTES,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreRegistration,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
    build_store_key,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.worker import (
    KVCacheStoreSendingThread,
    _StoreTask,
)
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


class _ReadyEvent:

    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1


class _RecordingStore:

    def __init__(self):
        self.existing = set()
        self.queries = []
        self.puts = []

    def batch_is_exist(self, keys):
        keys = list(keys)
        self.queries.append(keys)
        return [int(key in self.existing) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes,
                                     replicate_config):
        keys = list(keys)
        self.puts.append((keys, addresses, sizes, replicate_config))
        self.existing.update(keys)
        return [0] * len(keys)


def _cache_config() -> CacheConfig:
    return CacheConfig(
        max_batches=1,
        block_size=4,
        num_cpu_blocks=0,
        num_gpu_blocks=32,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
        ),
    )


def _default_hf_config(num_kv_heads: int):
    return SimpleNamespace(
        architectures=['GenericForCausalLM'],
        model_type='generic',
        hidden_size=4096,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=num_kv_heads,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=32000,
    )


@pytest.mark.parametrize(
    ('attention_kind', 'num_kv_heads', 'tp_size', 'expected_effective_heads',
     'expected_replica_num'),
    [
        ('mha', 32, 8, 32, 1),
        ('mha_one_head_per_rank', 8, 8, 8, 1),
        ('gqa_heads_above_tp', 16, 8, 16, 1),
        ('gqa_heads_below_tp', 2, 8, 8, 4),
        ('mqa', 1, 8, 8, 8),
    ],
)
def test_default_model_config_records_kv_head_replication(
    attention_kind,
    num_kv_heads,
    tp_size,
    expected_effective_heads,
    expected_replica_num,
):
    del attention_kind
    model_config = DefaultModelConfigBuilder.build(
        _default_hf_config(num_kv_heads),
        tp=tp_size,
    )

    assert model_config.num_key_value_heads == expected_effective_heads
    assert model_config.num_replicate_key_value_heads == expected_replica_num


def test_mla_model_config_records_full_tp_replication(monkeypatch):
    from lmdeploy.pytorch.configurations import deepseek_v2 as deepseek_v2_module

    monkeypatch.setattr(deepseek_v2_module, 'flash_mla_available',
                        lambda: False)
    hf_config = SimpleNamespace(
        architectures=['DeepseekV2ForCausalLM'],
        model_type='deepseek_v2',
        hidden_size=4096,
        num_hidden_layers=2,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=32000,
    )

    model_config = DeepseekV2ModelConfigBuilder.build(hf_config, tp=8)

    assert model_config.num_key_value_heads == 8
    assert model_config.num_replicate_key_value_heads == 8


def test_dsa_model_config_records_full_tp_replication(monkeypatch):
    from lmdeploy.pytorch.configurations import deepseek_v2 as deepseek_v2_module

    monkeypatch.setattr(deepseek_v2_module, 'flash_mla_available', lambda: True)
    hf_config = SimpleNamespace(
        architectures=['GlmMoeDsaForCausalLM'],
        model_type='glm_moe_dsa',
        hidden_size=4096,
        num_hidden_layers=2,
        num_attention_heads=64,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_topk=2048,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=32000,
    )

    model_config = GlmMoeDsaModelConfigBuilder.build(hf_config, tp=8)

    assert model_config.num_key_value_heads == 8
    assert model_config.num_replicate_key_value_heads == 8
    assert model_config.mla_index_topk == 2048


def test_llava_gqa_model_config_records_replication():
    text_config = _default_hf_config(num_kv_heads=2)
    hf_config = SimpleNamespace(
        architectures=['LlavaForConditionalGeneration'],
        text_config=text_config,
    )

    model_config = LlavaHfModelConfigBuilder.build(hf_config, tp=8)

    assert model_config.num_key_value_heads == 8
    assert model_config.num_replicate_key_value_heads == 4


def test_factory_and_connector_forward_explicit_replica_geometry(monkeypatch):
    captured_factory = {}
    connector_result = object()

    def fake_connector(role, cache_config, **kwargs):
        captured_factory.update(role=role, cache_config=cache_config, **kwargs)
        return connector_result

    monkeypatch.setattr(connector_module, 'MooncakeStoreConnector',
                        fake_connector)
    cache_config = _cache_config()

    result = build_kv_connector(
        KVConnectorRole.WORKER,
        cache_config,
        global_rank=7,
        tp_rank=5,
        tp_size=8,
        kv_head_replica_num=4,
    )

    assert result is connector_result
    assert captured_factory == {
        'role': KVConnectorRole.WORKER,
        'cache_config': cache_config,
        'global_rank': 7,
        'tp_rank': 5,
        'tp_size': 8,
        'kv_head_replica_num': 4,
    }


def test_connector_forwards_replica_geometry_to_worker(monkeypatch):
    captured_worker = {}
    worker_result = object()

    def fake_worker(cache_config, **kwargs):
        captured_worker.update(cache_config=cache_config, **kwargs)
        return worker_result

    monkeypatch.setattr(connector_module, 'MooncakeStoreWorker', fake_worker)
    cache_config = _cache_config()

    connector = connector_module.MooncakeStoreConnector(
        KVConnectorRole.WORKER,
        cache_config,
        global_rank=7,
        tp_rank=5,
        tp_size=8,
        kv_head_replica_num=4,
    )

    assert connector.connector_worker is worker_result
    assert captured_worker == {
        'cache_config': cache_config,
        'global_rank': 7,
        'tp_rank': 5,
        'tp_size': 8,
        'kv_head_replica_num': 4,
    }


def test_model_agent_forwards_model_replica_geometry(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    captured = {}
    connector = SimpleNamespace(register_kv_caches=lambda _caches: None)
    cache_engine = SimpleNamespace(connector_kv_caches={'kv': object()})
    agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
    agent.all_context = nullcontext
    agent.cache_config = _cache_config()
    agent.model_config = SimpleNamespace(num_replicate_key_value_heads=4)
    agent.rank = 7
    agent.cache_stream = object()
    agent.dist_config = SimpleNamespace(attn_tp=8)
    agent.kv_connector = None
    agent.memdecode_agent = None
    agent.spec_agent = SimpleNamespace(build_cache_engine=lambda _stream: None)

    monkeypatch.setattr(agent_module, 'CacheEngine',
                        lambda *args, **kwargs: cache_engine)
    monkeypatch.setattr(agent_module, 'StateCacheEngine',
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(
        agent_module,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
            attn_tp_group=SimpleNamespace(rank=5), ), ),
    )

    def fake_build_connector(role, cache_config, **kwargs):
        captured.update(role=role, cache_config=cache_config, **kwargs)
        return connector

    monkeypatch.setattr(agent_module, 'build_kv_connector',
                        fake_build_connector)

    agent.build_cache_engine()

    assert captured['kv_head_replica_num'] == 4
    assert captured['tp_rank'] == 5
    assert captured['tp_size'] == 8


def _key_metadata(tp_size: int, replica_num: int) -> MooncakeStoreKeyMetadata:
    return MooncakeStoreKeyMetadata(
        model_name='test-model',
        cache_prefix='tenant/a',
        tp_size=tp_size,
        block_size=4,
        kv_head_replica_num=replica_num,
    )


def _save_request(save_id: int,
                  physical_ids: tuple[int, ...]) -> MooncakeStoreSaveRequest:
    token_len = len(physical_ids) * 4
    return MooncakeStoreSaveRequest(
        req_id=9,
        save_id=save_id,
        generation=0,
        token_len=token_len,
        block_ids=physical_ids,
        block_hashes=build_prefix_block_hashes(range(token_len), 4),
    )


def _sender(
        store,
        *,
        tp_rank: int,
        tp_size: int,
        replica_num: int,
        row_block_sizes: tuple[int, ...] = (10, ),
) -> KVCacheStoreSendingThread:
    registrations = tuple(
        MooncakeStoreRegistration(
            name=f'row.{index}',
            address=0x1000 + index * 0x1000,
            size=32 * block_bytes,
        ) for index, block_bytes in enumerate(row_block_sizes))
    return KVCacheStoreSendingThread(
        store=store,
        registrations=registrations,
        row_block_sizes=row_block_sizes,
        num_gpu_blocks=32,
        key_metadata=_key_metadata(tp_size, replica_num),
        global_rank=tp_rank,
        tp_rank=tp_rank,
        tp_size=tp_size,
        completion_callback=lambda _save_id: None,
        replicate_config=object(),
    )


def _handle(sender, request):
    event = _ReadyEvent()
    task = _StoreTask(request=request, ready_event=event, enqueue_time=0.0)
    assert sender._handle_request(task)
    assert event.calls == 1


@pytest.mark.parametrize(
    ('attention_kind', 'tp_size', 'replica_num'),
    [
        ('mha_or_gqa_heads_at_least_tp', 8, 1),
        ('mqa_or_mla', 8, 8),
        ('gqa_heads_below_tp', 8, 4),
    ],
)
def test_replica_group_striping_covers_each_canonical_key_exactly_once(
    attention_kind,
    tp_size,
    replica_num,
):
    del attention_kind
    store = _RecordingStore()
    physical_ids = (17, 2, 29, 4, 23, 6, 19, 8, 13)
    request = _save_request(31, physical_ids)
    metadata = _key_metadata(tp_size, replica_num)
    queried_per_rank = []

    for tp_rank in range(tp_size):
        query_count = len(store.queries)
        _handle(
            _sender(
                store,
                tp_rank=tp_rank,
                tp_size=tp_size,
                replica_num=replica_num,
            ),
            request,
        )
        queried = store.queries[query_count:] or [[]]
        assert len(queried) == 1
        queried_per_rank.append(queried[0])

        owned_ordinals = range(tp_rank % replica_num, len(physical_ids),
                               replica_num)
        assert queried[0] == [
            build_store_key(metadata, tp_rank // replica_num,
                            request.block_hashes[index])
            for index in owned_ordinals
        ]

    expected_keys = [
        build_store_key(metadata, kv_head_rank, block_hash)
        for kv_head_rank in range(tp_size // replica_num)
        for block_hash in request.block_hashes
    ]
    all_queried = [key for keys in queried_per_rank for key in keys]
    all_put = [key for keys, *_ in store.puts for key in keys]
    assert Counter(all_queried) == Counter(expected_keys)
    assert Counter(all_put) == Counter(expected_keys)


def test_striping_uses_absolute_ordinal_not_physical_block_id():
    store = _RecordingStore()
    # Every physical ID is 0 modulo 4. Rank 1 must nevertheless own request
    # ordinals 1 and 5, proving that allocator slots do not choose ownership.
    physical_ids = (28, 24, 20, 16, 12, 8, 4)
    request = _save_request(32, physical_ids)
    sender = _sender(
        store,
        tp_rank=1,
        tp_size=8,
        replica_num=4,
        row_block_sizes=(10, 20),
    )

    _handle(sender, request)

    assert store.queries[0] == [
        build_store_key(sender.key_metadata, 0, request.block_hashes[index])
        for index in (1, 5)
    ]
    assert store.puts[0][1] == [[
        0x1000 + physical_ids[index] * 10, 0x2000 + physical_ids[index] * 20
    ] for index in (1, 5)]


def test_chunked_prefix_keeps_absolute_stripe_phase_across_physical_reallocation(
):
    store = _RecordingStore()
    sender = _sender(store, tp_rank=0, tp_size=8, replica_num=4)
    first = _save_request(40, (7, 2, 9, 4, 11))
    second = _save_request(41, (21, 20, 19, 18, 17, 16, 15, 14, 13))
    assert second.block_hashes[:len(first.block_hashes)] == first.block_hashes

    _handle(sender, first)
    _handle(sender, second)

    expected_first = [
        build_store_key(sender.key_metadata, 0, first.block_hashes[index])
        for index in (0, 4)
    ]
    expected_second = [
        build_store_key(sender.key_metadata, 0, second.block_hashes[index])
        for index in (0, 4, 8)
    ]
    assert store.queries == [expected_first, expected_second]
    assert store.puts[0][0] == expected_first
    assert store.puts[1][0] == expected_second[2:]
    assert store.puts[1][1] == [[0x1000 + second.block_ids[8] * 10]]


def test_store_key_matches_vllm_and_omits_deployment_geometry():
    block_hash = b'x' * MOONCAKE_BLOCK_HASH_BYTES
    base = _key_metadata(tp_size=8, replica_num=4)
    base_key = build_store_key(base, 0, block_hash)
    assert base_key == (
        f'tenant/a@test-model@tp_rank:0@group:0@{block_hash.hex()}')
    assert build_store_key(base, 1, block_hash) != base_key
    assert build_store_key(
        replace(base, cache_prefix='tenant/b'), 0, block_hash) != base_key
    assert build_store_key(
        replace(base, model_name='other-model'), 0, block_hash) != base_key

    # These fields remain runtime validation/layout inputs but, like vLLM,
    # are intentionally not encoded in a Store key. Deployments with
    # incompatible geometry must use separate Stores/cache prefixes.
    assert build_store_key(replace(base, tp_size=4), 0, block_hash) == base_key
    assert build_store_key(
        replace(base, kv_head_replica_num=2), 0, block_hash) == base_key
    assert build_store_key(replace(base, block_size=8), 0,
                           block_hash) == base_key


class _PinConnector:

    def __init__(self):
        self.updates = []

    def build_connector_meta(self, scheduler_output):
        token_len = scheduler_output.connector_token_lens[0]
        num_blocks = token_len // 4
        return MooncakeStoreConnectorMetadata(
            save_requests=(MooncakeStoreSaveRequest(
                req_id=scheduler_output.running[0].seq_id,
                save_id=71,
                generation=scheduler_output.connector_generations[0],
                token_len=token_len,
                block_ids=scheduler_output.connector_block_ids[0],
                block_hashes=(b'x' * MOONCAKE_BLOCK_HASH_BYTES, ) * num_blocks,
            ), ))

    def update_connector_output(self, output):
        self.updates.append(output)

    def shutdown(self):
        pass


def _paging_scheduler(connector):
    return Scheduler(
        scheduler_config=SchedulerConfig(
            max_batches=1,
            max_session_len=32,
            max_request_output_len=8,
            eviction_type='recompute',
        ),
        cache_config=CacheConfig(
            max_batches=1,
            block_size=4,
            num_cpu_blocks=0,
            num_gpu_blocks=4,
        ),
        seq_meta=SequenceMeta(4, strategy=ARSequenceStrategy()),
        kv_connector=connector,
    )


def test_tp_completion_intersection_holds_and_releases_scheduler_pin_once():
    connector = _PinConnector()
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(1).add_sequence(np.arange(4, dtype=np.int64))
    scheduler.block_manager.allocate(seq)
    logical_ids = seq.logical_blocks.get_real_blocks().copy()
    allocator = scheduler.block_manager.allocator
    metadata = scheduler.build_kv_connector_metadata([seq], (4, ))
    save_id = metadata.save_requests[0].save_id
    scheduler.block_manager.free(seq)

    executor = ExecutorBase.__new__(ExecutorBase)
    executor._kv_connector_acknowledged_sending = set()
    incomplete = executor._aggregate_kv_connector_outputs(
        [({save_id}, None)] * 7 + [(None, None)], )
    scheduler.update_connector_output(incomplete)

    assert not incomplete
    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([1]))
    assert scheduler.has_pending_kv_connector_work()

    completed = executor._aggregate_kv_connector_outputs([({save_id}, None)] *
                                                         8, )
    scheduler.update_connector_output(completed)

    assert completed.completed_save_ids == {save_id}
    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([0]))
    assert not scheduler.has_pending_kv_connector_work()

    # A sticky/repeated acknowledgement must not free the same pin twice.
    scheduler.update_connector_output({save_id})
    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([0]))
