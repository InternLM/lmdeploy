# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.engine_loop import EngineLoop
from lmdeploy.pytorch.engine.inputs_maker import (
    InputsMakerAsync,
    LongContextChunker,
    _compact_state_prefix_cache_restore_offsets,
    _compact_state_prefix_cache_save_offsets,
)


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

    def __init__(self, running):
        self.running = running

    def schedule(self, is_prefill: bool, prealloc_size: int):
        return SimpleNamespace(running=self.running, swap_in_map={}, swap_out_map={})


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
