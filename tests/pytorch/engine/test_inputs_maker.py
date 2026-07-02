# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import lmdeploy.pytorch.engine.inputs_maker as inputs_maker_module
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.engine_loop import EngineLoop, RunableEventAsync
from lmdeploy.pytorch.engine.inputs_maker import (
    InputsMakerAsync,
    InputsMakerConfig,
    LongContextChunker,
    _compact_state_prefix_cache_restore_offsets,
    _compact_state_prefix_cache_save_offsets,
)
from lmdeploy.pytorch.messages import MessageStatus


@dataclass
class _DummyMultiModal:
    start: int
    end: int


class _DummySeq:

    def __init__(self,
                 history_ids: int,
                 token_ids: int,
                 all_multimodals: dict,
                 input_multimodals: dict,
                 match_start_step: int = -1):
        self.num_history_ids = history_ids
        self.num_token_ids = token_ids
        self.history_multimodals = SimpleNamespace(multimodals=all_multimodals)
        self._input_multimodals = input_multimodals
        self.prefix_cache = SimpleNamespace(match_start_step=match_start_step)
        self.return_logits = False
        self.return_routed_experts = False
        self.return_ce_loss = False
        self.status = MessageStatus.RUNNING

    def set_step(self, step: int):
        self.num_history_ids = step

    def get_input_multimodals(self):
        return self._input_multimodals

    def get_chunk_limit_multimodals(self):
        match_start = self.prefix_cache.match_start_step
        if match_start >= 0 and self.num_history_ids > match_start:
            end = self.num_history_ids + self.num_token_ids
            return {
                key: [mm for mm in value if match_start <= mm.start and mm.end <= end]
                for key, value in self.history_multimodals.multimodals.items()
            }
        return self.get_input_multimodals()


def _state_seq(logical_state: int, restore_state: int = -1):
    return SimpleNamespace(logical_state=logical_state,
                           prefix_cache=SimpleNamespace(restore_state=restore_state))


class _FakeScheduler:

    def __init__(self, running, waiting=None, num_ready=0, num_running=0):
        self.running = running
        self.waiting = waiting or []
        self._num_ready = num_ready
        self._num_running = num_running

    def schedule(self,
                 is_prefill: bool,
                 prealloc_size: int,
                 allow_long_prefill: bool = True,
                 prefer_long_prefill: bool = False):
        self.allow_long_prefill = allow_long_prefill
        self.prefer_long_prefill = prefer_long_prefill
        return SimpleNamespace(running=self.running, swap_in_map={}, swap_out_map={})

    def reserve_long_context_chunk(self, seq, chunk_size: int, prealloc_size: int = 0, is_last_chunk: bool = False):
        return True

    def has_waiting(self):
        return len(self.waiting) > 0

    def has_waiting_long_prefill(self):
        return False

    def num_ready(self):
        return self._num_ready

    def num_running(self):
        return self._num_running


class _FakeEngineStrategy:

    def get_prealloc_size(self, is_prefill: bool):
        return 0


class _FakeSamplingStrategy:

    def make_sampling_inputs(self, running):
        return None


class _FakeModelAgentStrategy:

    def make_extra_inputs(self, running, inputs):
        return None

    def make_stopping_criteria(self, running):
        return None


def _fake_model_inputs(is_chunk: bool = False):
    return SimpleNamespace(is_decoding=False,
                           is_chunk=is_chunk,
                           is_first_chunk=False,
                           is_last_chunk=False,
                           is_chunk_multimodal=False)


def test_engine_loop_skips_prefix_cache_publish_when_disabled():

    class _DisabledBlockTrie:
        enable = False

        def commit_state_checkpoints(self, seqs):
            raise AssertionError('disabled prefix cache must not commit state checkpoints')

        def release_state_checkpoint_restores(self, seqs):
            raise AssertionError('disabled prefix cache must not release state checkpoint restores')

    loop = EngineLoop.__new__(EngineLoop)
    loop.scheduler = SimpleNamespace(block_trie=_DisabledBlockTrie())

    loop._publish_forward_prefix_cache([object()], has_state_checkpoint_save=True)


def test_engine_loop_keeps_state_save_pinned_until_output_boundary():
    events = []

    class _BlockTrie:
        enable = True
        pinned = False

        def commit_state_checkpoints(self, seqs, acquire_save_ref=False):
            events.append(('commit', acquire_save_ref))
            assert acquire_save_ref
            self.pinned = True

        def release_state_checkpoint_restores(self, seqs):
            events.append(('release_restore', self.pinned))

        def release_state_checkpoint_saves(self, seqs):
            events.append(('release_save', self.pinned))
            self.pinned = False

    class _InputsMaker:

        def __init__(self, block_trie):
            self.block_trie = block_trie

        def update_running_seqs(self, running, model_inputs):
            events.append('update_running')

        async def prefetch_next_inputs(self):
            events.append(('prefetch', self.block_trie.pinned))
            return None, None

    class _Executor:

        def __init__(self, block_trie):
            self.block_trie = block_trie

        async def get_output_async(self):
            events.append(('get_output', self.block_trie.pinned))
            return None

    block_trie = _BlockTrie()
    loop = EngineLoop.__new__(EngineLoop)
    loop.scheduler = SimpleNamespace(block_trie=block_trie, collect_migration_done=lambda: None)
    loop.inputs_maker = _InputsMaker(block_trie)
    loop.executor = _Executor(block_trie)
    model_inputs = SimpleNamespace(state_prefix_cache_save_offsets=[1])
    forward_inputs = dict(inputs=model_inputs, delta=None)

    forward_inputs, next_running = asyncio.run(loop._main_loop_get_outputs([object()], forward_inputs))

    assert forward_inputs is None
    assert next_running is None
    assert events == [
        'update_running',
        ('commit', True),
        ('release_restore', True),
        ('prefetch', True),
        ('get_output', True),
        ('release_save', True),
    ]
    assert not block_trie.pinned


def test_engine_loop_treats_pending_long_context_chunk_as_runnable():
    events = []

    class _Scheduler:

        def has_unfinished(self):
            return False

        def collect_migration_done(self):
            events.append('collect_migration_done')

    class _InputsMaker:

        def has_pending_long_context_chunk(self):
            return True

        async def send_next_inputs(self):
            events.append('send_next_inputs')
            return 'forward_inputs', ['long-seq']

    loop = EngineLoop.__new__(EngineLoop)
    loop.scheduler = _Scheduler()
    loop.inputs_maker = _InputsMaker()
    loop.has_runable_event = RunableEventAsync(loop.scheduler, loop.inputs_maker.has_pending_long_context_chunk)
    loop._sleep_requested = False

    result = asyncio.run(asyncio.wait_for(loop._main_loop_try_send_next_inputs(), timeout=1.0))

    assert result == ('forward_inputs', ['long-seq'])
    assert events == ['collect_migration_done', 'send_next_inputs']


def _make_policy_maker(long_seq, decode_seq=None):
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = False
    maker.scheduler = _FakeScheduler([])
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.long_context_chunker.set_seq(long_seq)
    maker.running_seqs = [] if decode_seq is None else [decode_seq]
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker._last_forward_kind = None
    maker._short_prefill_turns_since_long_chunk = 0
    maker._short_prefill_turns_per_long_chunk = 3
    return maker


def test_inputs_maker_reads_opt_ttft_short_turns_env(monkeypatch):
    monkeypatch.setattr(inputs_maker_module._envs, 'opt_ttft_short_turns', 5)
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=16, kernel_block_size=16))
    config = InputsMakerConfig(max_batches=1, max_prefill_token_num=512, role=EngineRole.Decode)

    maker = InputsMakerAsync(
        executor=SimpleNamespace(device_type='cpu'),
        scheduler=scheduler,
        adapter_manager=SimpleNamespace(),
        engine_strategy=_FakeEngineStrategy(),
        sampling_strategy=_FakeSamplingStrategy(),
        model_agent_strategy=_FakeModelAgentStrategy(),
        config=config,
    )

    assert maker._short_prefill_turns_per_long_chunk == 5


def test_inputs_maker_clamps_opt_ttft_short_turns_to_one(monkeypatch):
    monkeypatch.setattr(inputs_maker_module._envs, 'opt_ttft_short_turns', 0)
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=16, kernel_block_size=16))
    config = InputsMakerConfig(max_batches=1, max_prefill_token_num=512, role=EngineRole.Decode)

    maker = InputsMakerAsync(
        executor=SimpleNamespace(device_type='cpu'),
        scheduler=scheduler,
        adapter_manager=SimpleNamespace(),
        engine_strategy=_FakeEngineStrategy(),
        sampling_strategy=_FakeSamplingStrategy(),
        model_agent_strategy=_FakeModelAgentStrategy(),
        config=config,
    )

    assert maker._short_prefill_turns_per_long_chunk == 1


def test_long_context_chunker_uses_cached_multimodal_size_for_chunk_limit():
    image = _DummyMultiModal(start=512, end=5888)
    seq = _DummySeq(
        history_ids=5888,
        token_ids=1056,
        all_multimodals={'image': [image]},
        input_multimodals={},
        match_start_step=0,
    )

    chunker = LongContextChunker(max_prefill_token_num=512)
    assert chunker.is_long_context(seq)

    chunker.set_seq(seq)

    assert chunker.max_prefill_num == 5376
    assert chunker.is_last_chunk()
    chunk_size, multimodals = chunker.next_chunk_size()
    assert chunk_size == 1056
    assert multimodals is None


def test_long_context_chunker_only_tracks_remaining_multimodals():
    cached_image = _DummyMultiModal(start=512, end=5888)
    remaining_image = _DummyMultiModal(start=6400, end=7424)
    seq = _DummySeq(
        history_ids=5888,
        token_ids=2000,
        all_multimodals={'image': [cached_image, remaining_image]},
        input_multimodals={'image': [remaining_image]},
        match_start_step=0,
    )

    chunker = LongContextChunker(max_prefill_token_num=512)
    chunker.set_seq(seq)
    chunk_size, multimodals = chunker.next_chunk_size()

    assert chunker.max_prefill_num == 5376
    assert chunk_size == 2000
    assert multimodals == {'image': [remaining_image]}


def test_single_forward_multimodal_long_context_stays_normal_prefill_for_spec_decoding():
    image = _DummyMultiModal(start=0, end=1024)
    seq = _DummySeq(
        history_ids=0,
        token_ids=1024,
        all_multimodals={'image': [image]},
        input_multimodals={'image': [image]},
    )
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=False,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = True
    maker.scheduler = _FakeScheduler([seq])
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker.create_model_inputs = lambda seqs, is_prefill: model_inputs
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is model_inputs
    assert not model_inputs.is_chunk
    assert not model_inputs.is_first_chunk
    assert not model_inputs.is_last_chunk
    assert not model_inputs.is_chunk_multimodal


def test_spec_decoding_text_turn_ignores_previous_multimodal_chunk_limit():
    previous_image = _DummyMultiModal(start=512, end=5888)
    seq = _DummySeq(
        history_ids=5888,
        token_ids=1056,
        all_multimodals={'image': [previous_image]},
        input_multimodals={},
    )
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=False,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = True
    maker.scheduler = _FakeScheduler([seq])
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: model_inputs

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is model_inputs
    assert model_inputs.is_first_chunk
    assert not model_inputs.is_chunk_multimodal


def test_prefix_resumed_long_context_suffix_starts_new_chunk_chain():
    seq = _DummySeq(
        history_ids=1024,
        token_ids=2000,
        all_multimodals={},
        input_multimodals={},
        match_start_step=0,
    )
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=False,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = False
    maker.scheduler = _FakeScheduler([seq])
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    captured = {}

    def _create_model_inputs_long_context(seq_arg, chunk_size, multimodals):
        captured['seq'] = seq_arg
        captured['chunk_size'] = chunk_size
        captured['multimodals'] = multimodals
        return model_inputs

    maker.create_model_inputs_long_context = _create_model_inputs_long_context

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is model_inputs
    assert captured == {
        'seq': seq,
        'chunk_size': 512,
        'multimodals': None,
    }
    assert model_inputs.is_first_chunk
    assert not model_inputs.is_last_chunk
    assert not model_inputs.is_chunk_multimodal


def test_long_context_final_chunk_preserves_multimodal_flag_for_spec_decoding():
    image = _DummyMultiModal(start=0, end=1024)
    seq = _DummySeq(
        history_ids=512,
        token_ids=512,
        all_multimodals={'image': [image]},
        input_multimodals={},
    )

    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=False,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = True
    maker.scheduler = _FakeScheduler([])
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    seq.status = MessageStatus.RUNNING
    maker.long_context_chunker.seq = seq
    maker.long_context_chunker.has_multimodal = True
    maker.long_context_chunker.max_prefill_num = 512
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker.create_model_inputs = lambda seqs, is_prefill: model_inputs
    maker.create_model_inputs_delta_valid_only = lambda: (None, [seq], [])

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is model_inputs
    assert model_inputs.is_chunk
    assert not model_inputs.is_first_chunk
    assert model_inputs.is_last_chunk
    assert model_inputs.is_chunk_multimodal
    assert not maker.long_context_chunker.enabled()


def test_long_context_chunk_defers_to_decode_after_chunk_forward():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    delta = SimpleNamespace(is_decoding=True)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._last_forward_kind = 'long_context_chunk'
    maker.create_model_inputs_delta = lambda: (delta, [decode_seq], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('long chunk should wait behind decode'))

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['inputs'] is None
    assert forward_inputs['delta'] is delta
    assert maker.to_evict_seqs == []


def test_long_context_chunk_runs_after_decode_forward():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=True,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._last_forward_kind = 'decode'
    maker.engine_strategy = SimpleNamespace(get_prealloc_size=lambda is_prefill: 99)
    reserve_calls = []

    def _reserve_long_context_chunk(seq, chunk_size, prealloc_size=0, is_last_chunk=False):
        reserve_calls.append((seq, chunk_size, prealloc_size, is_last_chunk))
        return True

    maker.scheduler.reserve_long_context_chunk = _reserve_long_context_chunk
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not repeat'))
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: model_inputs

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['inputs'] is model_inputs
    assert forward_inputs['delta'] is None
    assert not model_inputs.is_first_chunk
    assert not model_inputs.is_last_chunk
    assert reserve_calls == [(long_seq, 512, 0, False)]


def test_abandoned_long_context_chunk_is_dropped_without_cleanup_forward():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    maker = _make_policy_maker(long_seq)
    long_seq.status = MessageStatus.STOPPED
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('abandoned chunk should not continue'))

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs is None
    assert not maker.long_context_chunker.enabled()


def test_deferred_long_context_chunk_runs_when_decode_has_no_valid_seqs():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=True,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._decode_count = 3
    maker._last_forward_kind = 'long_context_chunk'
    maker.create_model_inputs_delta = lambda: (None, [], [decode_seq])
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: model_inputs

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['inputs'] is model_inputs
    assert forward_inputs['delta'] is None
    assert maker.to_evict_seqs == [decode_seq]
    assert maker._decode_count == 0


def test_long_context_chunk_falls_back_to_decode_when_chunk_reservation_fails():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    delta = SimpleNamespace(is_decoding=True)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._last_forward_kind = 'decode'
    maker.scheduler.reserve_long_context_chunk = lambda *args, **kwargs: False
    maker.create_model_inputs_delta = lambda: (delta, [decode_seq], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('chunk inputs should not be created without KV reservation'))

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['inputs'] is None
    assert forward_inputs['delta'] is delta


def test_last_long_context_chunk_waits_for_prefill_turn_with_decode_ready():
    long_seq = _DummySeq(history_ids=512, token_ids=256, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    delta = SimpleNamespace(is_decoding=True)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._last_forward_kind = 'long_context_chunk'
    maker.create_model_inputs_delta = lambda: (delta, [decode_seq], [])

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['inputs'] is None
    assert forward_inputs['delta'] is delta
    assert maker.long_context_chunker.enabled()


def test_active_long_context_chunk_round_robin_does_not_starve_with_waiting_short_prefills():
    long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    short_seqs = [
        _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={}) for _ in range(4)
    ]
    short_batches = [[seq] for seq in short_seqs]
    chunk_inputs = _fake_model_inputs(is_chunk=True)

    class _RoundRobinScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[object()])
            self.schedule_calls = 0

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            assert not allow_long_prefill
            assert not prefer_long_prefill
            self.schedule_calls += 1
            running = short_batches.pop(0)
            return SimpleNamespace(running=running, swap_in_map={}, swap_out_map={})

        def has_waiting(self):
            return len(short_batches) > 0

    maker = _make_policy_maker(long_seq)
    maker.scheduler = _RoundRobinScheduler()
    maker._last_forward_kind = 'long_context_chunk'
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs = lambda seqs, is_prefill: _fake_model_inputs()
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: chunk_inputs

    first = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    second = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    third = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    fourth = maker._make_forward_inputs(prefill=True)

    assert first['running'] == [short_seqs[0]]
    assert not first['inputs'].is_chunk
    assert second['running'] == [short_seqs[1]]
    assert not second['inputs'].is_chunk
    assert third['running'] == [short_seqs[2]]
    assert not third['inputs'].is_chunk
    assert fourth['running'] == [long_seq]
    assert fourth['inputs'] is chunk_inputs
    assert fourth['inputs'].is_chunk
    assert not fourth['inputs'].is_last_chunk
    assert maker.scheduler.schedule_calls == 3


def test_active_long_context_chunk_obeys_short_prefill_quota_after_decode_turn():
    long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    short_seq = _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={})
    calls = []

    class _ShortPrefillScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[short_seq])

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            assert not allow_long_prefill
            assert not prefer_long_prefill
            return SimpleNamespace(running=[short_seq], swap_in_map={}, swap_out_map={})

        def has_waiting(self):
            return True

    maker = _make_policy_maker(long_seq)
    maker.scheduler = _ShortPrefillScheduler()
    maker._last_forward_kind = 'decode'
    maker.create_model_inputs = lambda seqs, is_prefill: _fake_model_inputs()
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('long chunk should wait behind short prefill quota'))

    forward_inputs = maker._make_forward_inputs(prefill=False)

    assert forward_inputs['running'] == [short_seq]
    assert not forward_inputs['inputs'].is_chunk
    assert forward_inputs['delta'] is None
    assert maker._short_prefill_turns_since_long_chunk == 1
    assert calls == [(False, False)]


def test_active_long_context_chunk_does_not_start_another_waiting_long_prefill():
    active_long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    waiting_long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    waiting_long_seq.status = MessageStatus.WAITING
    short_seqs = [
        _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={}) for _ in range(3)
    ]
    short_batches = [[seq] for seq in short_seqs]
    chunk_inputs = _fake_model_inputs(is_chunk=True)
    calls = []

    class _ActiveLongScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[waiting_long_seq])

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            assert not allow_long_prefill
            assert not prefer_long_prefill
            return SimpleNamespace(running=short_batches.pop(0), swap_in_map={}, swap_out_map={})

        def has_waiting(self):
            return True

        def has_waiting_long_prefill(self):
            return True

    maker = _make_policy_maker(active_long_seq)
    maker.scheduler = _ActiveLongScheduler()
    maker._last_forward_kind = 'long_context_chunk'
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs = lambda seqs, is_prefill: _fake_model_inputs()
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: chunk_inputs

    first = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    second = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    third = maker._make_forward_inputs(prefill=True)
    maker._last_forward_kind = 'prefill'
    fourth = maker._make_forward_inputs(prefill=True)

    assert first['running'] == [short_seqs[0]]
    assert second['running'] == [short_seqs[1]]
    assert third['running'] == [short_seqs[2]]
    assert fourth['running'] == [active_long_seq]
    assert fourth['inputs'] is chunk_inputs
    assert fourth['inputs'].is_chunk
    assert not fourth['inputs'].is_last_chunk
    assert waiting_long_seq.status == MessageStatus.WAITING
    assert calls == [(False, False), (False, False), (False, False)]


def test_active_long_context_chunk_reservation_failure_blocks_short_prefill_and_drains_decode():
    active_long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    short_seq = _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={})
    delta = SimpleNamespace(is_decoding=True)
    calls = []

    class _ReserveFailScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[short_seq])

        def reserve_long_context_chunk(self,
                                       seq,
                                       chunk_size: int,
                                       prealloc_size: int = 0,
                                       is_last_chunk: bool = False):
            return False

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            raise AssertionError('short prefill should wait while active long chunk is KV-blocked')

        def has_waiting(self):
            return True

    maker = _make_policy_maker(active_long_seq, decode_seq)
    maker.scheduler = _ReserveFailScheduler()
    maker._last_forward_kind = 'prefill'
    maker._short_prefill_turns_since_long_chunk = maker._short_prefill_turns_per_long_chunk
    maker.create_model_inputs = lambda seqs, is_prefill: (_ for _ in ()).throw(
        AssertionError('short prefill should not run'))
    maker.create_model_inputs_delta = lambda: (delta, [decode_seq], [])
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('chunk inputs should not be created without KV reservation'))

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is None
    assert forward_inputs['delta'] is delta
    assert forward_inputs['running'] == [decode_seq]
    assert maker.long_context_chunker.enabled()
    assert maker.long_context_chunker.next_step == 0
    assert calls == []


def test_active_long_context_chunk_reservation_failure_blocks_short_prefill_without_decode():
    active_long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    short_seq = _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={})
    calls = []

    class _ReserveFailScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[short_seq])

        def reserve_long_context_chunk(self,
                                       seq,
                                       chunk_size: int,
                                       prealloc_size: int = 0,
                                       is_last_chunk: bool = False):
            return False

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            raise AssertionError('short prefill should wait while active long chunk is KV-blocked')

        def has_waiting(self):
            return True

    maker = _make_policy_maker(active_long_seq)
    maker.scheduler = _ReserveFailScheduler()
    maker._last_forward_kind = 'prefill'
    maker._short_prefill_turns_since_long_chunk = maker._short_prefill_turns_per_long_chunk
    maker.create_model_inputs = lambda seqs, is_prefill: (_ for _ in ()).throw(
        AssertionError('short prefill should not run'))
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('chunk inputs should not be created without KV reservation'))

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs is None
    assert maker.long_context_chunker.enabled()
    assert maker.long_context_chunker.next_step == 0
    assert calls == []


def test_waiting_long_context_first_chunk_gets_round_robin_turn_after_short_prefills():
    long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    short_seqs = [
        _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={}) for _ in range(3)
    ]
    chunk_inputs = _fake_model_inputs(is_chunk=True)
    calls = []

    class _RoundRobinScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[object()])

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            if prefer_long_prefill:
                return SimpleNamespace(running=[long_seq], swap_in_map={}, swap_out_map={})
            running = [short_seqs[len(calls) - 1]]
            return SimpleNamespace(running=running, swap_in_map={}, swap_out_map={})

        def has_waiting(self):
            return True

        def has_waiting_long_prefill(self):
            return True

    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = False
    maker.scheduler = _RoundRobinScheduler()
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker._last_forward_kind = None
    maker._short_prefill_turns_since_long_chunk = 0
    maker._short_prefill_turns_per_long_chunk = 3
    maker.create_model_inputs = lambda seqs, is_prefill: _fake_model_inputs()
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda seq, chunk_size, multimodals: chunk_inputs

    first = maker._make_forward_inputs(prefill=True)
    second = maker._make_forward_inputs(prefill=True)
    third = maker._make_forward_inputs(prefill=True)
    fourth = maker._make_forward_inputs(prefill=True)

    assert first['running'] == [short_seqs[0]]
    assert not first['inputs'].is_chunk
    assert second['running'] == [short_seqs[1]]
    assert not second['inputs'].is_chunk
    assert third['running'] == [short_seqs[2]]
    assert not third['inputs'].is_chunk
    assert fourth['running'] == [long_seq]
    assert fourth['inputs'] is chunk_inputs
    assert fourth['inputs'].is_first_chunk
    assert calls == [(False, False), (False, False), (False, False), (True, True)]


def test_waiting_long_context_admission_failure_falls_back_to_short_prefill():
    long_seq = _DummySeq(history_ids=0, token_ids=2048, all_multimodals={}, input_multimodals={})
    short_seq = _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={})
    calls = []

    class _WaitingLongFailScheduler(_FakeScheduler):

        def __init__(self):
            super().__init__([], waiting=[long_seq, short_seq])

        def schedule(self,
                     is_prefill: bool,
                     prealloc_size: int,
                     allow_long_prefill: bool = True,
                     prefer_long_prefill: bool = False):
            calls.append((allow_long_prefill, prefer_long_prefill))
            if prefer_long_prefill:
                return SimpleNamespace(running=[], swap_in_map={}, swap_out_map={})
            assert not allow_long_prefill
            return SimpleNamespace(running=[short_seq], swap_in_map={}, swap_out_map={})

        def has_waiting(self):
            return True

        def has_waiting_long_prefill(self):
            return True

    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode)
    maker.spec_decoding = False
    maker.scheduler = _WaitingLongFailScheduler()
    maker.engine_strategy = _FakeEngineStrategy()
    maker.sampling_strategy = _FakeSamplingStrategy()
    maker.model_agent_strategy = _FakeModelAgentStrategy()
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.running_seqs = []
    maker.to_evict_seqs = []
    maker._decode_count = 0
    maker._last_forward_kind = None
    maker._short_prefill_turns_since_long_chunk = 3
    maker._short_prefill_turns_per_long_chunk = 3
    maker.create_model_inputs = lambda seqs, is_prefill: _fake_model_inputs()
    maker.create_model_inputs_delta = lambda: (_ for _ in ()).throw(AssertionError('decode should not run'))
    maker.create_model_inputs_delta_valid_only = lambda: (None, [], [])
    maker.create_model_inputs_long_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('long prefill should not create chunk inputs after admission failure'))

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['running'] == [short_seq]
    assert not forward_inputs['inputs'].is_chunk
    assert forward_inputs['delta'] is None
    assert calls == [(True, True), (False, False)]


def test_normal_prefill_can_update_running_while_long_chunker_is_active():
    long_seq = _DummySeq(history_ids=0, token_ids=1024, all_multimodals={}, input_multimodals={})
    short_seq = _DummySeq(history_ids=0, token_ids=16, all_multimodals={}, input_multimodals={})
    model_inputs = _fake_model_inputs()
    maker = _make_policy_maker(long_seq)

    maker.update_running_seqs([short_seq], model_inputs)

    assert maker.running_seqs == [short_seq]
    assert maker.long_context_chunker.enabled()
    assert maker.long_context_chunker.next_step == 0


def test_last_long_context_chunk_runs_as_prefill_on_prefill_turn():
    image = _DummyMultiModal(start=600, end=700)
    long_seq = _DummySeq(history_ids=512,
                         token_ids=256,
                         all_multimodals={'image': [image]},
                         input_multimodals={'image': [image]})
    decode_seq = _DummySeq(history_ids=0, token_ids=1, all_multimodals={}, input_multimodals={})
    model_inputs = SimpleNamespace(is_decoding=False,
                                   is_chunk=False,
                                   is_first_chunk=False,
                                   is_last_chunk=False,
                                   is_chunk_multimodal=False)
    maker = _make_policy_maker(long_seq, decode_seq)
    maker._last_forward_kind = 'long_context_chunk'
    maker.engine_strategy = SimpleNamespace(get_prealloc_size=lambda is_prefill: 7)
    reserve_calls = []

    def _reserve_long_context_chunk(seq, chunk_size, prealloc_size=0, is_last_chunk=False):
        reserve_calls.append((seq, chunk_size, prealloc_size, is_last_chunk))
        return True

    maker.scheduler.reserve_long_context_chunk = _reserve_long_context_chunk
    maker.create_model_inputs = lambda seqs, is_prefill: model_inputs
    maker.create_model_inputs_delta_valid_only = lambda: (None, [decode_seq], [])

    forward_inputs = maker._make_forward_inputs(prefill=True)

    assert forward_inputs['inputs'] is model_inputs
    assert model_inputs.is_chunk
    assert model_inputs.is_last_chunk
    assert model_inputs.is_chunk_multimodal
    assert not maker.long_context_chunker.enabled()
    assert reserve_calls == [(long_seq, 256, 7, True)]


def test_do_prefill_default_forces_pending_last_chunk_prefill():
    long_seq = _DummySeq(history_ids=512, token_ids=256, all_multimodals={}, input_multimodals={})
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Decode,
                                   max_prefill_token_num=512,
                                   max_batches=1,
                                   prefill_interval=100)
    maker.scheduler = _FakeScheduler([], num_ready=1, num_running=1)
    maker.long_context_chunker = LongContextChunker(max_prefill_token_num=512)
    maker.long_context_chunker.set_seq(long_seq)
    maker._decode_count = 0

    assert maker.do_prefill_default()


def test_state_prefix_cache_restore_offsets_are_compact():
    messages = [_state_seq(4, 11), _state_seq(5, -1), _state_seq(6, 13)]

    src_offsets, dst_offsets = _compact_state_prefix_cache_restore_offsets(messages)

    assert src_offsets == (11, 13)
    assert dst_offsets == (4, 6)


def test_state_prefix_cache_save_offsets_are_compact():
    messages = [_state_seq(4), _state_seq(5), _state_seq(6)]

    src_offsets, dst_offsets = _compact_state_prefix_cache_save_offsets(messages, [-1, 21, 22])

    assert src_offsets == (5, 6)
    assert dst_offsets == (21, 22)


@pytest.mark.parametrize('max_q_seqlen', [1, 4])  # standard decode, then spec/MTP
def test_create_model_inputs_delta_valid_only_matches_one_decode_advance(max_q_seqlen):
    # Regression for #4024. The delta is built from the (stale) scheduler seqs
    # at the current state, then applied after the model-agent's StepInputs has
    # advanced one decode step. So delta.max/sum_kv_seqlen must equal the base
    # kv (num_all_ids of the valid seqs) advanced by EXACTLY one decode step --
    # the invariant the engine uses in ModelInputs.step (model_inputs.py) and
    # get_model_inputs_next_decoding (strategies/ar/model_inputs.py):
    #     max_kv_seqlen += max_q_seqlen
    #     sum_kv_seqlen += num_valid_seqs * max_q_seqlen
    # Parametrizing max_q_seqlen proves the offset is one max_q_seqlen, not the
    # old double (num_all_ids + 2 * max_q_seqlen) nor zero (num_all_ids alone).
    num_all_ids = [100, 250]  # valid seqs' kv at the (stale) build state
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.engine_strategy = SimpleNamespace(get_num_decode_tokens=lambda: max_q_seqlen)
    maker.running_seqs = [
        SimpleNamespace(status=MessageStatus.RUNNING, num_all_ids=num_all_ids[0]),
        SimpleNamespace(status=MessageStatus.RUNNING, num_all_ids=num_all_ids[1]),
        SimpleNamespace(status=MessageStatus.STOPPED, num_all_ids=70),  # dropped
    ]

    output, valid_seqs, invalid_seqs = maker.create_model_inputs_delta_valid_only()

    assert [seq.num_all_ids for seq in valid_seqs] == num_all_ids
    assert len(invalid_seqs) == 1
    # base kv at the (stale) build state + one canonical decode advance
    assert output.max_kv_seqlen == max(num_all_ids) + max_q_seqlen
    assert output.sum_kv_seqlen == sum(num_all_ids) + len(valid_seqs) * max_q_seqlen
