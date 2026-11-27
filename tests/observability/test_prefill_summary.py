# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise real prefill accounting with host-only input descriptors."""

import importlib.util
import pickle
import sys
import unittest
from dataclasses import asdict, fields
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_request_trace import SOURCE, RecordTestCase, core, obs, output

spec = importlib.util.spec_from_file_location('prefill_trace_under_test', SOURCE / 'lmdeploy/pytorch/trace.py')
native = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'lmdeploy.trace': core}):
    spec.loader.exec_module(native)


def sequence(start=0, end=96, cached=0):
    return NS(input_start_pos=start, input_end_pos=end, cached_tokens=cached,
              prefill_trace=native.PrefillTrace())


def inputs(starts, lengths, final=True):
    def tensor(values):
        return NS(device=NS(type='cpu'), tolist=lambda: values)
    return NS(history_lengths=tensor(starts), seq_length=tensor(lengths),
              is_decoding=False, is_chunk=True, is_last_chunk=final)


def dispatch(seqs, starts, lengths, now, final=True):
    with patch.object(native.time, 'monotonic', return_value=now):
        batch = native.begin_prefill(seqs, inputs(starts, lengths, final))
    return (batch, final) if batch is not None else None


def complete(batch, now):
    records, final = batch
    with patch.object(native.time, 'monotonic', return_value=now):
        native.complete_prefill(records)
    # Unit-level scenarios accept successful final outputs. Rejected outputs
    # and actual engine acceptance are covered through EngineLoop separately.
    if final:
        for state, *_ in records:
            state.accept()


class PrefillTests(unittest.TestCase):
    def test_chunks_cache_sources_and_real_host_wait(self):
        seq = sequence(cached=16)
        state = seq.prefill_trace
        state.external_start, state.external_end = 8, 16
        first = dispatch([seq], [16], [40], 1, final=False)
        complete(first, 3)
        self.assertIsNone(state.summary)
        second = dispatch([seq], [56], [40], 5)
        complete(second, 8)
        summary = state.summary
        self.assertEqual(summary.chunk_count, 2)
        self.assertEqual(summary.computed_tokens, 80)
        self.assertEqual(summary.computed_input_tokens, 80)
        self.assertEqual(summary.recomputed_tokens, 0)
        self.assertEqual((summary.local_cached_tokens, summary.external_cached_tokens), (8, 8))
        self.assertEqual((summary.min_chunk_tokens, summary.max_chunk_tokens), (40, 40))
        self.assertEqual((summary.inter_chunk_wait_seconds, summary.max_inter_chunk_wait_seconds), (2, 2))
        package = ModuleType('lmdeploy')
        package.trace = core
        with patch.dict(sys.modules, {'lmdeploy': package, 'lmdeploy.trace': core}):
            self.assertEqual(pickle.loads(pickle.dumps(summary)), summary)

    def test_prefetched_chunk_has_no_host_gap_or_mutable_range(self):
        seq = sequence()
        first = dispatch([seq], [0], [64], 1, final=False)
        second = dispatch([seq], [64], [32], 2)
        # Async prefetch and new turns can change live sequence state. Completion
        # must use the captured input bounds and accumulator owner.
        state = seq.prefill_trace
        seq.input_start_pos = seq.input_end_pos = 999
        seq.prefill_trace = native.PrefillTrace()
        complete(first, 3)
        complete(second, 4)
        self.assertEqual(state.summary.inter_chunk_wait_seconds, 0)
        self.assertEqual(state.summary.computed_input_tokens, 96)
        self.assertIsNone(seq.prefill_trace.summary)

    def test_batch_ownership_and_retained_session_history(self):
        first, second = sequence(start=64, end=80, cached=8), sequence(end=32)
        batch = dispatch([first, second], [72, 0], [8, 32], 1)
        complete(batch, 2)
        self.assertEqual(first.prefill_trace.summary.computed_input_tokens, 8)
        self.assertEqual(first.prefill_trace.summary.local_cached_tokens, 8)
        self.assertEqual(second.prefill_trace.summary.computed_input_tokens, 32)
        self.assertEqual(second.prefill_trace.summary.local_cached_tokens, 0)

    def test_preemption_replay_and_readmission_are_counted_once(self):
        seq = sequence(end=80, cached=16)
        state = seq.prefill_trace
        state.preempt()  # A waiter with no executed work is not preempted.
        first = dispatch([seq], [16], [32], 1, final=False)
        complete(first, 2)
        state.preempt()
        state.preempt()  # Deactivation and block reclamation share one episode.
        replay = dispatch([seq], [0], [80], 5)
        complete(replay, 6)
        self.assertEqual(state.summary.preemption_count, 1)
        self.assertEqual(state.summary.reschedule_count, 1)
        self.assertEqual(state.summary.recomputed_tokens, 48)
        self.assertEqual(state.summary.computed_tokens, 112)
        self.assertEqual(state.summary.inter_chunk_wait_seconds, 3)
        self.assertEqual(state.summary.local_cached_tokens, 16)
        state.preempt()  # Decode preemption cannot rewrite TTFT diagnostics.
        self.assertIsNone(dispatch([seq], [0], [80], 8))
        self.assertEqual(state.summary.preemption_count, 1)

    def test_external_reuse_is_clipped_to_committed_current_input(self):
        seq = sequence(start=40, end=100, cached=24)
        seq.prefill_trace.external_start, seq.prefill_trace.external_end = 32, 64
        complete(dispatch([seq], [64], [36], 1), 2)
        self.assertEqual(seq.prefill_trace.summary.external_cached_tokens, 24)
        self.assertEqual(seq.prefill_trace.summary.local_cached_tokens, 0)

    def test_unexecuted_and_failed_loads_do_not_fabricate_summary(self):
        seq = sequence(cached=16)
        self.assertIsNone(seq.prefill_trace.summary)
        # No successful publish: lookup matches and attempted load bytes alone
        # do not populate the external range.
        complete(dispatch([seq], [16], [80], 1), 2)
        self.assertEqual(seq.prefill_trace.summary.external_cached_tokens, 0)

    def test_state_size_is_independent_of_chunk_count(self):
        seq = sequence(end=10000)
        state_keys = tuple(field.name for field in fields(seq.prefill_trace))
        for chunk in range(10000):
            batch = dispatch([seq], [chunk], [1], chunk * 2, final=chunk == 9999)
            complete(batch, chunk * 2 + 1)
        state = seq.prefill_trace
        self.assertFalse(hasattr(state, '__dict__'))
        self.assertFalse(any(isinstance(getattr(state, key), (list, dict, set)) for key in state_keys))
        self.assertEqual(state.summary.chunk_count, 10000)
        self.assertEqual(state.summary.inter_chunk_wait_seconds, 9999)

    def test_observation_error_does_not_raise(self):
        seq = sequence()
        with patch.object(native, '_observation_errors', 0), self.assertLogs(native.logger, level='WARNING') as logs:
            self.assertIsNone(native.begin_prefill([seq], inputs([], [])))
            self.assertIsNone(native.begin_prefill([seq], inputs([], [])))
            self.assertEqual(native.observation_error_count(), 1)
        self.assertTrue(seq.prefill_trace.invalid)
        self.assertEqual(seq.prefill_trace.pending, 0)
        self.assertEqual(len(logs.records), 1)
        healthy = sequence()
        complete(dispatch([healthy], [0], [96], 1), 2)
        self.assertEqual(healthy.prefill_trace.summary.computed_tokens, 96)

    def test_completion_error_is_local_and_does_not_log_payload(self):
        bad, good = sequence(), sequence()
        batch = dispatch([bad, good], [0, 0], [96, 96], 1)
        original = native.PrefillTrace.complete
        def fail_one(state, *args):
            if state is bad.prefill_trace:
                raise ValueError('private-input-text')
            return original(state, *args)
        with patch.object(native.PrefillTrace, 'complete', fail_one), \
                patch.object(native, '_observation_errors', 0), self.assertLogs(native.logger, 'WARNING') as logs:
            complete(batch, 2)
            self.assertEqual(native.observation_error_count(), 1)
        self.assertEqual(bad.prefill_trace.pending, 0)
        self.assertIsNone(bad.prefill_trace.summary)
        self.assertEqual(good.prefill_trace.summary.computed_tokens, 96)
        self.assertIn('operation=complete error_type=ValueError', logs.output[0])
        self.assertNotIn('private-input-text', str(logs.output))

    def test_diagnostic_handler_failure_does_not_escape_or_disable_later_inputs(self):
        bad, good = sequence(), sequence()
        batch = dispatch([bad], [0], [96], 1)
        with patch.object(native, '_observation_errors', 0), \
                patch.object(native.logger, 'warning', side_effect=OSError('broken handler')):
            native.discard_prefill(batch[0], 'output', 'CancelledError')
            self.assertIsNone(native.begin_prefill([sequence()], inputs([], [])))
            complete(dispatch([good], [0], [96], 2), 3)
            self.assertEqual(native.observation_error_count(), 2)
        self.assertTrue(bad.prefill_trace.invalid)
        self.assertEqual(bad.prefill_trace.pending, 0)
        self.assertIsNone(bad.prefill_trace.summary)
        self.assertEqual(good.prefill_trace.summary.computed_tokens, 96)

    def test_no_tensor_transfer_for_diagnostics(self):
        descriptor = inputs([0], [96])
        descriptor.seq_length.device.type = 'cuda'
        descriptor.seq_length.tolist = lambda: self.fail('GPU tensor was read')
        self.assertIsNone(native.begin_prefill([sequence()], descriptor))


class ObserverPrefillTests(RecordTestCase):
    def test_snapshot_survives_coalescing_without_recounting(self):
        seq = sequence()
        complete(dispatch([seq], [0], [96], 1), 2)
        t = obs.RequestTrace()
        first = output([1], timestamp=10)
        first.req_metrics.prefill = seq.prefill_trace.summary
        t.observe(first)
        t.observe(first)
        t.finish('success')
        self.assertEqual(self.traces[0].prefill_metrics['chunk_count'], 1)
        self.assertEqual(self.traces[0].prefill_metrics['computed_tokens'], 96)
        self.assertEqual(t.prefill_metrics, asdict(seq.prefill_trace.summary))

    def test_unsupported_native_backend_remains_unknown(self):
        t = obs.RequestTrace()
        t.observe(output([1], timestamp=10))
        t.finish('success')
        self.assertIsNone(self.traces[0].prefill_metrics)
