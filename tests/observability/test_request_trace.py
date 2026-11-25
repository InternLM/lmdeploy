"""CPU-only tests: run with python -m unittest discover -s tests/observability -v.

Load the stdlib-only observer without importing lmdeploy's GPU dependencies.
The engine integration tests compile the actual production generation methods
and inject fake backend outputs, rather than duplicating their control flow.
"""
import ast
import asyncio
import dataclasses
import enum
import importlib.util
import json
import logging
import sys
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT
core_spec = importlib.util.spec_from_file_location('lmdeploy.trace', SOURCE / 'lmdeploy/trace.py')
core = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core)
spec = importlib.util.spec_from_file_location(
    'request_observability', SOURCE / 'lmdeploy/serve/observability/request_trace.py')
obs = importlib.util.module_from_spec(spec)
otel_spec = importlib.util.spec_from_file_location(
    'otel_trace', SOURCE / 'lmdeploy/serve/observability/otel_trace.py')
otel = importlib.util.module_from_spec(otel_spec)
otel_spec.loader.exec_module(otel)
output_spec = importlib.util.spec_from_file_location(
    'output_trace', SOURCE / 'lmdeploy/serve/observability/output_trace.py')
output_trace = importlib.util.module_from_spec(output_spec)
sys.modules[output_spec.name] = output_trace
output_spec.loader.exec_module(output_trace)
previous_observability = sys.modules.get('lmdeploy.serve.observability')
sys.modules['lmdeploy.serve.observability'] = NS(otel_trace=otel, output_trace=output_trace)
previous_core = sys.modules.get('lmdeploy.trace')
sys.modules['lmdeploy.trace'] = core
try:
    spec.loader.exec_module(obs)
finally:
    if previous_observability is None:
        sys.modules.pop('lmdeploy.serve.observability', None)
    else:
        sys.modules['lmdeploy.serve.observability'] = previous_observability
    if previous_core is None:
        sys.modules.pop('lmdeploy.trace', None)
    else:
        sys.modules['lmdeploy.trace'] = previous_core


def output(tokens=(), timestamp=0, events=(), status='SUCCESS'):
    return NS(token_ids=list(tokens), req_metrics=NS(token_timestamp=timestamp, engine_events=list(events),
                                                   cached_tokens=2), status=status,
              logprobs=None, logits=None, last_hidden_state=None, routed_experts=None, cache_block_ids=None)


def event(name, timestamp):
    return NS(type=NS(name=name), timestamp=timestamp)


class RecordTestCase(unittest.TestCase):
    def setUp(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.skipTest('install lmdeploy[serve] to inspect exported spans')
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.addCleanup(provider.shutdown)
        self.traces = []
        finish_engine = otel.finish_engine
        def capture(trace, status):
            finish_engine(trace, status)
            self.traces.append(trace)
        for p in (patch.object(obs, 'enabled', return_value=True),
                  patch.object(otel, 'initialize'),
                  patch.object(otel, '_tracer', provider.get_tracer('test')),
                  patch.object(otel, 'finish_engine', side_effect=capture)):
            p.start()
            self.addCleanup(p.stop)

    @property
    def http_spans(self):
        return [s for s in self.exporter.get_finished_spans() if s.kind.name == 'SERVER']

    @property
    def prompt_spans(self):
        return [s for s in self.exporter.get_finished_spans() if s.name == 'inference']

    def exported_attributes(self):
        return [dict(s.attributes) for s in self.exporter.get_finished_spans()]


class TimingTests(RecordTestCase):
    def test_stage_boundaries_and_no_payload(self):
        t = obs.RequestTrace()
        t.observe(output([1], 15, [event('QUEUED', 10), event('SCHEDULED', 12)]))
        t.observe(output([2, 3], 19, [event('QUEUED', 10), event('SCHEDULED', 12),
                                               event('PREEMPTED', 16), event('SCHEDULED', 17)]))
        t.finish('success')
        row = self.traces[0]
        self.assertEqual(row.otel_span.attributes['lmdeploy.queue_seconds'], 2)
        self.assertEqual(obs.interval(row.scheduled, row.first_token), 3)
        self.assertEqual(obs.interval(row.first_token, row.last_token), 4)
        self.assertIsNone(row.output_tokens)
        self.assertNotIn('token_ids', row.otel_span.attributes)
        t.finish('error')
        self.assertEqual(len(self.traces), 1)


    def test_missing_metrics_unknown_not_zero(self):
        t = obs.RequestTrace()
        t.observe(NS(token_ids=[], req_metrics=None))
        t.finish('error')
        for field in ('queue_seconds', 'kv_load_seconds'):
            self.assertNotIn('lmdeploy.' + field, t.otel_span.attributes)
        self.assertEqual([s.name for s in self.exporter.get_finished_spans()], ['llm_request'])


    def test_scoring_has_no_invented_first_token(self):
        t = obs.RequestTrace()
        t.observe(output([], 15, [event('QUEUED', 10), event('SCHEDULED', 12)]))
        t.finish('success')
        self.assertIsNone(obs.interval(self.traces[0].scheduled, self.traces[0].first_token))
        self.assertIsNone(obs.interval(self.traces[0].first_token, self.traces[0].last_token))


    def test_single_token_decode_zero(self):
        t = obs.RequestTrace()
        t.observe(output([1], 15))
        t.finish('success')
        self.assertEqual(obs.interval(self.traces[0].first_token, self.traces[0].last_token), 0)


    def test_wall_clock_regression_is_unknown(self):
        self.assertIsNone(obs.interval(10, 9))
        self.assertIsNone(obs.interval(float('nan'), 20))

    def test_kv_multiple_loads_and_failure(self):
        t = obs.RequestTrace()
        events = [event('KV_LOAD_STARTED', 10), event('KV_LOAD_FINISHED', 12),
                  event('KV_LOAD_STARTED', 14), event('KV_LOAD_FAILED', 17)]
        t.observe(output(events=events))
        t.finish('error')
        self.assertEqual(self.traces[0].kv_seconds, 5)
        self.assertEqual(self.traces[0].kv_status, 'failed')


    def test_preprocess_error_and_cancellation(self):
        async def run():
            for exc in (ValueError('private input'), asyncio.CancelledError()):
                @obs.trace_preprocess
                async def fail():
                    raise exc
                with self.assertRaises(type(exc)):
                    await fail()
        asyncio.run(run())
        self.assertEqual([r.engine_status for r in self.traces], ['error', 'cancelled'])
        self.assertNotIn('private input', json.dumps(self.exported_attributes()))


    def test_fanout_and_concurrent_context_isolation(self):
        @obs.trace_preprocess
        async def prepare():
            await asyncio.sleep(0)
            return NS(session=NS(session_id=7), input_token_len=2, trace=None)
        async def child(rid):
            context = {'traces': [], 'request_id': rid, 'otel_span': otel._tracer.start_span('llm_request')}
            tok = obs.request_outcome_context.set(context)
            try:
                return await prepare(), await prepare()
            finally:
                obs.request_outcome_context.reset(tok)
        pairs = asyncio.run(self._gather(child('a'), child('b')))
        ids = [p.trace.otel_span.get_span_context().span_id for pair in pairs for p in pair]
        self.assertEqual([p.trace.request_id for pair in pairs for p in pair], ['a', 'a', 'b', 'b'])
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual([p.trace.prompt_index for pair in pairs for p in pair], [0, 1, 0, 1])
        self.assertIsNone(obs.request_outcome_context.get())

    def test_resolved_adapter_and_protocol_model_are_metadata_only(self):
        class Engine:
            model_name = 'base-model'

            @obs.trace_preprocess
            async def preprocess(self, adapter_name=None):
                return NS(input_token_len=3, adapter_name=adapter_name)

        async def run():
            engine = Engine()
            first = await engine.preprocess('finance-lora')
            self.assertEqual(first.trace.model, 'finance-lora')
            self.assertEqual(first.trace.base_model, 'base-model')
            second = await engine.preprocess('finance-lora', trace_model='served-model-alias')
            self.assertEqual(second.trace.model, 'served-model-alias')
            self.assertEqual(second.adapter_name, 'finance-lora')
            third = await engine.preprocess('Bearer private')
            self.assertIsNone(third.trace.model)
            third.trace.finish('success')
        asyncio.run(run())
        self.assertNotIn('private', json.dumps(self.exported_attributes()))


    @staticmethod
    async def _gather(*coroutines):
        return await asyncio.gather(*coroutines)


class ASGITests(RecordTestCase):
    # Exercise the ASGI layer without importing the GPU-dependent server package.
    async def invoke(self, bodies=(), status=200, headers=(), failure=None, sse=True):
        sent = []
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'text/event-stream' if sse else b'application/json')]})
            for b in bodies:
                await send({'type': 'http.response.body', 'body': b, 'more_body': True})
            if failure:
                raise failure
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        async def send(message):
            sent.append(message)
        async def receive():
            return {'type': 'http.disconnect'}
        with patch.object(otel, 'initialize'):
            middleware = obs.RequestTraceMiddleware(app)
        await middleware({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': headers},
                         receive, send)
        return sent


    def test_fragmented_sse_payload_unchanged_and_stage_observation(self):
        data = [b'data: {"choices":[{"delta":{"reasoning_content":"private thought"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"private answer"}}]}\n\n', b'data: [DONE]\n\n']
        body = b''.join(data)
        pieces = [body[i:i+7] for i in range(0, len(body), 7)]
        sent = asyncio.run(self.invoke(pieces, headers=[(obs.HEADER.encode(), b'client-123')]))
        self.assertEqual(b''.join(m.get('body', b'') for m in sent), body)
        row = self.http_spans[-1]
        self.assertNotEqual(row.attributes.get('request.id'), 'client-123')
        self.assertIsNone(row.attributes.get('request.id'))
        self.assertEqual(row.attributes.get('lmdeploy.outcome'), 'success')
        self.assertIsNotNone(row.attributes.get('lmdeploy.reasoning_to_content_seconds'))
        self.assertNotIn('private', json.dumps(dict(row.attributes)))
        self.assertFalse(any(k == obs.HEADER.encode() for k, _ in sent[0]['headers']))
        self.assertIsNone(obs.request_outcome_context.get())


    def test_error_sse_is_error(self):
        asyncio.run(self.invoke([b'data: {"error":{"message":"private"}}\n\ndata: [DONE]\n\n']))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'error')


    def test_http_error_and_invalid_id(self):
        asyncio.run(self.invoke(status=422, headers=[(obs.HEADER.encode(), b'bad\nvalue')]))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'error')
        self.assertNotEqual(self.http_spans[-1].attributes.get('request.id'), 'bad\nvalue')


    def test_send_failure_propagates(self):
        with self.assertRaises(BrokenPipeError):
            asyncio.run(self.invoke(failure=BrokenPipeError()))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'cancelled')


    def test_abort_protocol_is_cancelled_with_http_200(self):
        asyncio.run(self.invoke([b'data: {"choices":[{"finish_reason":"abort"}]}\ndata: [DONE]\n']))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'cancelled')


    def test_accepted_terminal_survives_later_disconnect(self):
        with self.assertRaises(BrokenPipeError):
            asyncio.run(self.invoke([b'data: [DONE]\n\n'], failure=BrokenPipeError()))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'success')


    def test_partial_terminal_event_is_not_success(self):
        prefixes = [b'event: response.completed\n',
                    b'event: message_stop\ndata: {"type":"message_stop"}\n',
                    b'data: {"type":"response.completed"}\n', b'data: [DONE]\n',
                    b'data: [DONE]\n \n']
        for prefix in prefixes:
            for failure in (None, BrokenPipeError()):
                with self.subTest(prefix=prefix, disconnected=failure is not None):
                    if failure is None:
                        asyncio.run(self.invoke([prefix]))
                    else:
                        with self.assertRaises(BrokenPipeError):
                            asyncio.run(self.invoke([prefix], failure=failure))
                    self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'cancelled')


    def test_large_fragmented_terminal_waits_for_event_boundary(self):
        header = b'event: response.completed\n'
        data = b'data: {"output":"' + b'x' * (1024 * 1024) + b'"}\n'
        for complete in (False, True):
            bodies = [header, data[:2], data[2:]] + ([b'\n'] if complete else [])
            sent = asyncio.run(self.invoke(bodies))
            self.assertEqual(b''.join(m.get('body', b'') for m in sent), b''.join(bodies))
            self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'),
                             'success' if complete else 'cancelled')


    def test_truncated_sse_is_not_success(self):
        asyncio.run(self.invoke([b'data: {"choices":[{"delta":{"content":"x"}}]}\n']))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'cancelled')


    def test_responses_cancelled_is_not_failed(self):
        body = b'event: response.failed\ndata: {"type":"response.failed","response":{"status":"cancelled"}}\n\n'
        asyncio.run(self.invoke([body]))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'cancelled')


    def test_event_header_requires_data_and_dispatch_boundary(self):
        observer = obs.OutputObservation(0)
        observer.feed(b'event: response.completed\n\n')
        self.assertFalse(observer.terminal)
        observer.feed(b'event: response.failed\ndata: null\n')
        self.assertFalse(observer.terminal)
        self.assertFalse(observer.error)
        observer.feed(b'\n')
        self.assertTrue(observer.terminal)
        self.assertTrue(observer.error)

    def test_parse_error_is_not_success(self):
        body = b'data: {"choices":[{"finish_reason":"parse_error"}]}\n\ndata: [DONE]\n\n'
        asyncio.run(self.invoke([body]))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'error')


    def test_responses_tool_start_counts_as_first_output(self):
        body = (b'data: {"type":"response.output_item.added","item":{"type":"function_call",'
                b'"arguments":""}}\n\ndata: {"type":"response.completed"}\n\n')
        asyncio.run(self.invoke([body]))
        self.assertIsNotNone(self.http_spans[-1].attributes.get('lmdeploy.first_output_seconds'))
        self.assertIsNone(self.http_spans[-1].attributes.get('lmdeploy.content_first_seconds'))
        self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), 'success')


    def test_nonstream_has_unknown_visible_stages(self):
        asyncio.run(self.invoke([b'{"content":"private"}'], sse=False))
        self.assertIsNone(self.http_spans[-1].attributes.get('lmdeploy.content_first_seconds'))


    def test_observer_bounds_and_recovers(self):
        o = obs.OutputObservation(time.monotonic())
        o.feed(b'data: ' + b'x' * (1024 * 1024 + 10))
        self.assertLessEqual(len(o.buffer), 1024 * 1024)
        o.feed(b'\ndata: [DONE]\n\n')
        self.assertTrue(o.terminal)


# Compile original classes/methods with postponed annotations and CPU fakes.
# This catches integration bugs in the modified generate path and safe_run cleanup.
def load_engine():
    namespace = dict(enum=enum, asyncio=asyncio, time=time, dataclasses=dataclasses, Any=object,
                     asynccontextmanager=asynccontextmanager, logger=logging.getLogger('test'),
                     trace_preprocess=obs.trace_preprocess, trace_generate=obs.trace_generate)
    def exec_nodes(path, names):
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                           + nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    exec_nodes(SOURCE / 'lmdeploy/serve/core/exceptions.py',
               ['ErrorCode', 'RequestError', 'SafeRunException'])
    # RequestError's constructor uses these global maps.
    namespace['ERROR_MESSAGES'] = {v: v.value for v in namespace['ErrorCode']}
    namespace['ERROR_STATUS'] = {v: 500 for v in namespace['ErrorCode']}
    namespace['enum'] = enum
    exec_nodes(SOURCE / 'lmdeploy/serve/managers/session_manager.py',
               ['Session', 'RequestHandlePool', 'SessionManager'])
    namespace.update(weakref=__import__('weakref'), itertools=__import__('itertools'))
    path = SOURCE / 'lmdeploy/serve/core/async_engine.py'
    tree = ast.parse(path.read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncEngine')
    methods = ('generate', 'safe_run', '_request_error_from_status')
    engine.body = [n for n in engine.body if getattr(n, 'name', '') in methods]
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('GenOut', 'PreprocessedRequest')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                       + nodes + [engine], type_ignores=[])
    # dataclasses requires an existing __module__ entry when examining annotations.
    namespace['__name__'] = __name__
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


class EngineIntegrationTests(RecordTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_engine()

    def engine(self, mode='success'):
        ns = self.ns
        statuses = NS(SUCCESS='SUCCESS', FINISH='FINISH', CANCEL='CANCEL', INTERNAL_ENGINE_ERROR='ERROR',
                      INPUT_LENGTH_ERROR='LENGTH', NOT_SUPPORTED='UNSUPPORTED', SESSION_REPEAT='REPEAT',
                      PREFIX_CACHE_CONFLICT='CONFLICT', ENGINE_STOP_ERROR='STOP', SESSION_NOT_EXIST='NO_SESSION',
                      HANDLER_NOT_EXIST='NO_HANDLER', NO_QUEUE='NO_QUEUE')
        ns['ResponseType'] = statuses
        ns['EngineOutput'] = lambda status, tokens: output(tokens, status=status)
        # Keep the fake aligned with the current engine state, which also
        # accepts emitted_prefix_chars for multi-token stop-string handling.
        ns['DetokenizeState'] = lambda input_len, **kwargs: NS(ids_offset=input_len, **kwargs)
        ns['RequestStats'] = lambda **kw: NS(**kw)
        ns['IterationStats'] = lambda: NS()
        ns['metrics_processor'] = NS(**{name: lambda *a: None for name in (
            'increase_total_requests', 'increase_api_routed_requests', 'increase_failed_requests',
            'increase_succeeded_requests', 'decrease_api_routed_requests', 'queue_update')})
        class Handle:
            cancelled = False
            async def async_stream_infer(self, session_id, **kw):
                self.stream_output = kw['stream_output']
                if mode == 'scoring':
                    yield output([], 19, [event('QUEUED', 10), event('SCHEDULED', 12)], 'FINISH')
                    return
                if mode == 'stoponly':
                    yield output([12], 19, [event('QUEUED', 10), event('SCHEDULED', 12)], 'FINISH')
                    return
                if mode == 'finalonly':
                    yield output([11, 12], 19, [event('QUEUED', 10), event('SCHEDULED', 12)], 'FINISH')
                    return
                yield output([11], 15, [event('QUEUED', 10), event('SCHEDULED', 12)])
                if mode == 'error_metrics':
                    failure = output([], status='ERROR')
                    failure.req_metrics.cached_tokens = 0
                    yield failure
                    return
                if mode == 'error':
                    raise RuntimeError('backend failed')
                if mode == 'cancel':
                    raise asyncio.CancelledError()
                yield output([12], 19, status='FINISH')
            async def async_cancel(self, session_id):
                self.cancelled = True
        h = Handle()
        e = ns['AsyncEngine']()
        mgr = ns['SessionManager']()
        mgr.build_request_handle_pool(NS(create_instance=lambda: h), 1)
        e.session_mgr = mgr
        e.num_spec_token = 0
        e.epoch = 0
        e._if_session_stale = lambda *a: None
        e.tokenizer = NS(detokenize_incrementally=lambda ids, state, **kw: ('generated text', NS(ids_offset=len(ids))))
        e.request_logger = NS(log_response=lambda *a: None)
        trace = obs.RequestTrace()
        trace.preprocessed = time.monotonic()
        config = NS(logprob_start_len=-1, ignore_eos=True, stop_token_ids=[], skip_special_tokens=True,
                    spaces_between_special_tokens=True, include_stop_str_in_output=False)
        request = ns['PreprocessedRequest'](session=mgr.get(7), inputs={'input_ids': [1]},
                                             input_token_len=1, gen_config=config, adapter_name=None, trace=trace)
        return e, request, h

    def test_real_generate_nonstream_keeps_original_mode(self):
        e, req, handle = self.engine()
        async def run():
            return [out async for out in e.generate(req, stream_response=False)]
        result = asyncio.run(run())
        self.assertEqual(result[-1].finish_reason, 'length')
        self.assertFalse(handle.stream_output)
        row = self.traces[-1]
        self.assertEqual(obs.interval(row.scheduled, row.first_token), 3)
        self.assertEqual(obs.interval(row.first_token, row.last_token), 4)
        self.assertEqual(row.engine_status, 'success')
        self.assertEqual(e.session_mgr.sessions, {})


    def test_real_generate_final_only_keeps_nonstream_and_unknown_split(self):
        e, req, handle = self.engine('finalonly')
        async def run():
            return [out async for out in e.generate(req, stream_response=False)]
        result = asyncio.run(run())
        self.assertFalse(handle.stream_output)
        self.assertEqual(result[-1].generate_token_len, 2)
        row = self.traces[-1]
        self.assertIsNone(obs.interval(row.scheduled, row.first_token))
        self.assertIsNone(obs.interval(row.first_token, row.last_token))
        self.assertEqual(row.output_tokens, 2)
        self.assertEqual(row.finish_reason, 'length')
        self.assertTrue(row.token_counts_complete)


    def test_usage_matches_actual_generation_result_when_stop_token_is_filtered(self):
        for mode in ('success', 'finalonly', 'stoponly'):
            for stream in (False, True):
                for include_stop in (False, True):
                    with self.subTest(mode=mode, stream=stream, include_stop=include_stop):
                        engine, request, handle = self.engine(mode)
                        request.gen_config.ignore_eos = False
                        request.gen_config.stop_token_ids = [12]
                        request.gen_config.stop_words = []
                        request.gen_config.include_stop_str_in_output = include_stop
                        engine.tokenizer.decode = lambda *a, **kw: '<stop>'

                        async def run():
                            return [item async for item in engine.generate(request, stream_response=stream)]

                        result = asyncio.run(run())
                        final = result[-1]
                        self.assertEqual(final.finish_reason, 'stop')
                        expected = (1 if mode == 'stoponly' else 2) - (not include_stop)
                        self.assertEqual(final.generate_token_len, expected)
                        row = self.traces[-1]
                        self.assertEqual(row.output_tokens, final.generate_token_len)
                        self.assertEqual(request.trace.generated_tokens, 1 if mode == 'stoponly' else 2)
                        self.assertEqual(row.prompt_tokens, final.input_token_len)
                        self.assertEqual(row.cached_tokens, final.cached_tokens)
                        if mode == 'success':
                            self.assertEqual(row.otel_span.attributes.get('lmdeploy.tpot_seconds'), 4)
                        self.assertEqual(handle.stream_output, stream)
                        self.assertTrue(row.token_counts_complete)


    def test_scoring_result_reports_zero_without_inventing_token_timing(self):
        engine, request, _ = self.engine('scoring')
        request.gen_config.logprob_start_len = 0

        async def run():
            return [item async for item in engine.generate(request, stream_response=False)]

        result = asyncio.run(run())
        row = self.traces[-1]
        self.assertEqual(row.output_tokens, result[-1].generate_token_len)
        self.assertEqual(row.output_tokens, 0)
        self.assertTrue(row.token_counts_complete)
        self.assertNotIn('gen_ai.latency.time_to_first_token', row.otel_span.attributes)
        self.assertNotIn('lmdeploy.tpot_seconds', row.otel_span.attributes)
        self.assertEqual([s.name for s in self.exporter.get_finished_spans()], ['llm_request'])


    def test_real_generate_backend_error_cleans_up(self):
        e, req, handle = self.engine('error')
        async def run():
            return [out async for out in e.generate(req)]
        with self.assertRaises(self.ns['RequestError']):
            asyncio.run(run())
        self.assertEqual(self.traces[-1].engine_status, 'error')
        self.assertEqual(self.traces[-1].output_tokens, 1)
        self.assertFalse(self.traces[-1].token_counts_complete)
        self.assertTrue(handle.cancelled)
        self.assertEqual(e.session_mgr.sessions, {})


    def test_native_error_does_not_overwrite_last_api_usage_snapshot(self):
        engine, request, _ = self.engine('error_metrics')
        results = []

        async def run():
            async for item in engine.generate(request):
                results.append(item)

        with self.assertRaises(self.ns['RequestError']):
            asyncio.run(run())
        row = self.traces[-1]
        self.assertEqual(row.output_tokens, results[-1].generate_token_len)
        self.assertEqual(row.cached_tokens, results[-1].cached_tokens)
        self.assertEqual(row.cached_tokens, 2)
        self.assertFalse(row.token_counts_complete)
        self.assertEqual(row.engine_status, 'error')


    def test_real_generate_cancellation_suppressed_by_session_still_recorded(self):
        e, req, handle = self.engine('cancel')
        async def run():
            return [out async for out in e.generate(req)]
        asyncio.run(run())
        self.assertEqual(self.traces[-1].engine_status, 'cancelled')
        self.assertEqual(self.traces[-1].output_tokens, 1)
        self.assertFalse(self.traces[-1].token_counts_complete)
        self.assertTrue(handle.cancelled)
        self.assertEqual(e.session_mgr.sessions, {})


    def test_real_generate_early_close_cleans_up(self):
        e, req, handle = self.engine()
        async def run():
            gen = e.generate(req)
            await anext(gen)
            await gen.aclose()
        asyncio.run(run())
        self.assertEqual(self.traces[-1].engine_status, 'cancelled')
        self.assertEqual(self.traces[-1].output_tokens, 1)
        self.assertFalse(self.traces[-1].token_counts_complete)
        self.assertTrue(handle.cancelled)
        self.assertEqual(e.session_mgr.sessions, {})


    def test_disabled_trace_preserves_nonstream_backend_mode(self):
        e, req, handle = self.engine()
        req.trace = None
        async def run():
            return [out async for out in e.generate(req, stream_response=False)]
        asyncio.run(run())
        self.assertFalse(handle.stream_output)
        self.assertEqual(self.traces, [])


class AdditionalIntegrationTests(RecordTestCase):
    def test_actual_coordinator_records_load_success_and_failure(self):
        path = SOURCE / 'lmdeploy/pytorch/paging/kv_load_coordinator.py'
        tree = ast.parse(path.read_text())
        coordinator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KVLoadCoordinator')
        methods = ('_allocate_and_start_load', 'start_load', 'apply_load_results')
        coordinator.body = [n for n in coordinator.body if getattr(n, 'name', '') in methods]
        helper_names = ('_LoadPhase', '_DeferredLoadCleanup', '_LoadRecord')
        helpers = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in helper_names]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                           + helpers + [coordinator], type_ignores=[])
        names = dict(enum=enum, dataclass=dataclasses.dataclass, __name__=__name__, TRACE_ENABLED=True,
                     EventType=NS(KV_LOAD_STARTED='start', KV_LOAD_FINISHED='end',
                                  KV_LOAD_FAILED='fail', KV_LOAD_CANCELLED='cancel'))
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), names)
        for success, cleanup in ((True, 'NONE'), (False, 'NONE'), (True, 'STOP'),
                                 (False, 'STOP'), (True, 'END'), (False, 'END')):
            calls = []
            seq = NS(seq_id=1, block_size=8, num_blocks=0, logical_state=-1, status='waiting',
                     state=NS(begin_remote_load=lambda: None),
                     prefix_cache=NS(match_start_step=-1), num_history_ids=0,
                     record_event=lambda event_type, **kw: calls.append(event_type))
            c = names['KVLoadCoordinator']()
            c._loads = {}
            c._prefill_targets = {}
            c.block_manager = NS(allocate=lambda seq: None, get_block_table=lambda seq: [0])
            c.connector = NS(update_state_after_alloc=lambda *args: calls.append('bind'))
            c.state_manager = None
            c.record_cache_query = lambda *a: None
            c.track_prefill = lambda *a, **kw: None
            c._publish = lambda record: None
            c._rollback = lambda record: None
            c._finish_cancelled_or_failed = lambda record: None
            plan = NS(load_start_step=0, remote_step=8, target_blocks=1, original_kv_token_limit=None)
            c._allocate_and_start_load(seq, plan)
            c._loads[1].deferred_cleanup = names['_DeferredLoadCleanup'][cleanup]
            c.apply_load_results((NS(request_id=1, success=success),))
            terminal = 'cancel' if cleanup != 'NONE' else 'end' if success else 'fail'
            self.assertEqual(calls, ['bind', 'start', terminal])

    def test_anthropic_and_responses_visible_stages(self):
        for parts in (
            [b'data: {"type":"content_block_delta","delta":{"thinking":"private"}}\n\n',
             b'data: {"type":"content_block_delta","delta":{"text":"answer"}}\n\n',
             b'data: {"type":"message_stop"}\n\n'],
            [b'data: {"type":"response.reasoning_summary_text.delta","delta":"private"}\n\n',
             b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n',
             b'data: {"type":"response.completed"}\n\n']):
            o = obs.OutputObservation(time.monotonic())
            for part in parts:
                o.feed(part)
            self.assertTrue(o.terminal)
            self.assertIsNotNone(o.summary()['reasoning_to_content_seconds'])

    def test_consumed_request_does_not_finish_original_trace(self):
        trace = obs.RequestTrace()
        @obs.trace_generate
        async def generate(self, request):
            raise ValueError('already consumed')
            yield
        async def run():
            with self.assertRaises(ValueError):
                await anext(generate(None, NS(trace=trace, consumed=True)))
        asyncio.run(run())
        self.assertFalse(trace.finished)
        self.assertEqual(self.traces, [])


class ReviewedScopeTests(RecordTestCase):
    def test_invalid_native_timestamps_are_omitted(self):
        trace = obs.RequestTrace()
        trace.observe(output([1], float('inf'), [event('QUEUED', float('nan')),
                                                 event('SCHEDULED', float('inf'))]))
        trace.finish('error')
        row = self.traces[-1]
        self.assertIsNone(row.otel_span.attributes.get('lmdeploy.queue_seconds'))
        self.assertIsNone(row.queued)
        self.assertIsNone(row.first_token)
        json.dumps(dict(row.otel_span.attributes), allow_nan=False)


    def test_final_only_multitoken_output_does_not_invent_first_token(self):
        trace = obs.RequestTrace()
        trace.observe(output([1, 2, 3], 20, [event('QUEUED', 10), event('SCHEDULED', 12)], 'FINISH'))
        trace.finish('success')
        row = self.traces[-1]
        self.assertIsNone(obs.interval(row.scheduled, row.first_token))
        self.assertIsNone(obs.interval(row.first_token, row.last_token))
        self.assertIsNone(row.otel_span.attributes.get('lmdeploy.tpot_seconds'))
        self.assertEqual(row.otel_span.attributes.get('lmdeploy.queue_seconds'), 2)
        self.assertIsNone(row.output_tokens)
        self.assertFalse(row.token_counts_complete)
        self.assertEqual(trace.generated_tokens, 3)
        self.assertEqual(row.last_token, 20)


    def test_late_metrics_do_not_relabel_later_token_as_first(self):
        trace = obs.RequestTrace()
        trace.observe(NS(token_ids=[1], req_metrics=None, status='SUCCESS'))
        trace.observe(output([2], 20, status='FINISH'))
        trace.finish('success')
        self.assertIsNone(obs.interval(self.traces[-1].scheduled, self.traces[-1].first_token))
        self.assertIsNone(self.traces[-1].output_tokens)


    def test_partial_token_counts_and_absent_cache_are_explicit(self):
        trace = obs.RequestTrace()
        trace.observe(NS(token_ids=[1], req_metrics=None, status='SUCCESS'))
        trace.finish('cancelled')
        row = self.traces[-1]
        self.assertIsNone(row.output_tokens)
        self.assertFalse(row.token_counts_complete)
        self.assertIsNone(row.cached_tokens)
        self.assertIsNone(row.prompt_tokens)


    def test_visible_output_gap_excludes_metadata_and_terminal(self):
        o = obs.OutputObservation(0)
        with patch.object(obs.time, 'monotonic', side_effect=[1., 2., 52.]):
            o.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n')
            o.feed(b'data: {"choices":[{"delta":{"reasoning_content":"x"}}]}\n')
            o.feed(b'data: {"choices":[{"delta":{"content":"y"}}]}\n')
        o.feed(b'data: [DONE]\n')
        self.assertEqual(o.summary()['max_output_gap_seconds'], 50)
        self.assertEqual(o.summary()['first_output_seconds'], 2)

    def test_repeated_kv_failure_is_not_erased_by_later_success(self):
        trace = obs.RequestTrace()
        events = [NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8),
                  event('KV_LOAD_FAILED', 12),
                  NS(type=NS(name='KV_LOAD_STARTED'), timestamp=13, value=4),
                  event('KV_LOAD_FINISHED', 14)]
        trace.observe(output(events=events))
        trace.finish('success')
        row = self.traces[-1]
        self.assertEqual(row.kv_status, 'failed')
        self.assertEqual(row.kv_loaded_tokens, 4)
        self.assertEqual(row.kv_seconds, 3)


    def test_lookup_and_load_are_distinct_with_token_counts(self):
        trace = obs.RequestTrace()
        events = [event('KV_LOOKUP_STARTED', 10),
                  NS(type=NS(name='KV_LOOKUP_FINISHED'), timestamp=12, value=8),
                  NS(type=NS(name='KV_LOAD_STARTED'), timestamp=13, value=8),
                  event('KV_LOAD_FINISHED', 16)]
        trace.observe(output(events=events))
        trace.finish('success')
        row = self.traces[-1]
        self.assertEqual(row.lookup_seconds, 2)
        self.assertEqual(row.store_matched_tokens, 8)
        self.assertEqual(row.kv_seconds, 3)
        self.assertEqual(row.kv_loaded_tokens, 8)
        self.assertEqual(row.kv_status, 'success')


    def test_mooncake_polling_records_once_and_releases_lookup_state(self):
        path = SOURCE / 'lmdeploy/pytorch/kv_connector/mooncake/store/scheduler.py'
        tree = ast.parse(path.read_text())
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MooncakeStoreScheduler')
        methods = ('get_num_new_matched_tokens', '_lookup_range', '_get_request_block_hashes',
                   'prefetch_lookups', 'cancel_lookup', '_cancel_trace_lookup', 'request_finished')
        parent.body = [n for n in parent.body if getattr(n, 'name', '') in methods]
        helpers = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name in ('_RequestHashTracker', '_LookupPlan')]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                           + helpers + [parent], type_ignores=[])
        names = dict(dataclass=dataclasses.dataclass, __name__=__name__, TRACE_ENABLED=True,
                     logger=logging.getLogger('test'),
                     EventType=NS(KV_LOOKUP_STARTED='start', KV_LOOKUP_FINISHED='end', KV_LOOKUP_CANCELLED='cancel'),
                     EngineEvent=NS(new_event=lambda kind: event(kind, 1)),
                     build_prefix_block_hashes=lambda ids, bs, **kw: (b'hash',) * (len(ids) // bs))
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), names)
        for remote_tokens, prefetched in ((0, False), (8, False), (0, True), (8, True), (6, True)):
            c = names['MooncakeStoreScheduler']()
            c._cache_config = NS(block_size=4, prefix_cache_token_lookahead=0,
                                 state_lookup_alignment=4, max_batches=8)
            c._is_hybrid = remote_tokens == 6
            c._compact_capacity = None
            c._compact_loads = {}
            c._kv_transfer_config = NS(is_kv_consumer=True, is_kv_producer=True)
            c._request_hash_trackers = {}
            c._lookup_plans = {}
            c._failed_load_requests = set()
            c._next_save_block = {}
            c._pending_loads = {}
            c._inflight_loads = {}
            c._invalid_block_ids = set()
            results = iter((None, None, remote_tokens))
            discarded = []
            submissions = []
            def submit(req_id, *args, **kwargs):
                submissions.append(req_id)
                c.client.queries[req_id] = object()
            c.client = NS(lookup=lambda *a, **kw: next(results), discard=discarded.append,
                          submit=submit, queries={})
            calls = []
            request = NS(seq_id=1, adapter_name=None, all_ids=list(range(12)), engine_events=[],
                         history_multimodals=NS(empty=lambda: True), history_embeddings=[],
                         num_history_ids=0, num_valid_ids=12,
                         prefix_cache=NS(recompute_overlap=NS(recompute_blocks=0)),
                         get_prefix_cache_max_candidate_step=lambda: 12,
                         clamp_prefix_cache_match_step=lambda value: value,
                         record_event=lambda name, **kw: calls.append((name, kw.get('value'))))
            if prefetched:
                c.prefetch_lookups([request])
                c.prefetch_lookups([request])
                self.assertEqual(submissions, [1])
                self.assertEqual(calls, [('start', None)])
            self.assertEqual(c.get_num_new_matched_tokens(request, 0), (None, False))
            self.assertEqual(c.get_num_new_matched_tokens(request, 0), (None, False))
            expected = 0 if c._is_hybrid else remote_tokens
            self.assertEqual(c.get_num_new_matched_tokens(request, 0), (expected, expected > 0))
            self.assertEqual(calls, [('start', None), ('end', expected)])
            if expected:
                self.assertEqual(c.get_num_new_matched_tokens(request, 0), (remote_tokens, True))
                self.assertEqual(calls, [('start', None), ('end', remote_tokens)])
            if expected:
                tracker = c._request_hash_trackers[1]
                tracker.trace_lookup_pending = True
                tracker.trace_events = request.engine_events
                self.assertEqual(c.get_num_new_matched_tokens(request, 0), (remote_tokens, True))
                self.assertFalse(tracker.trace_lookup_pending)
                self.assertEqual(request.engine_events[-1].type.name, 'cancel')
            c._lookup_plans.clear()
            c.client.queries.clear()
            def fail(*args, **kwargs):
                raise RuntimeError('lookup failed')
            with patch.object(c.client, 'submit', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    c.prefetch_lookups([request])
            self.assertFalse(c._request_hash_trackers[1].trace_lookup_pending)
            with patch.object(c.client, 'lookup', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    c.get_num_new_matched_tokens(request, 0)
            self.assertFalse(c._request_hash_trackers[1].trace_lookup_pending)
            c.client.lookup = lambda *a, **kw: None
            self.assertEqual(c.get_num_new_matched_tokens(request, 0), (None, False))
            c.cancel_lookup(1)
            self.assertEqual(request.engine_events[-1].type.name, 'cancel')
            self.assertFalse(c._request_hash_trackers[1].trace_lookup_pending)
            c.request_finished(request)
            self.assertEqual(c._request_hash_trackers, {})
            self.assertEqual(c._lookup_plans, {})
            self.assertEqual(discarded, [1, 1])


class ReviewRegressionTests(RecordTestCase):
    def test_repeated_cumulative_events_do_not_multiply_store_metrics(self):
        trace = obs.RequestTrace()
        events = [event('QUEUED', 10), event('KV_LOOKUP_STARTED', 11),
                  NS(type=NS(name='KV_LOOKUP_FINISHED'), timestamp=13, value=8),
                  NS(type=NS(name='KV_LOAD_STARTED'), timestamp=14, value=8),
                  event('KV_LOAD_FINISHED', 17), event('SCHEDULED', 18)]
        for i in range(100):
            # Transport snapshots are distinct objects with identical values.
            snapshot = __import__('copy').deepcopy(events)
            trace.observe(output([i], 20 + i, snapshot, 'FINISH' if i == 99 else 'SUCCESS'))
        trace.finish('success')
        row = self.traces[-1]
        self.assertEqual(row.lookup_seconds, 2)
        self.assertEqual(row.kv_seconds, 3)
        self.assertEqual(row.kv_loaded_tokens, 8)
        self.assertIsNone(row.output_tokens)
        self.assertEqual(trace.generated_tokens, 100)
        self.assertEqual(trace.event_cursor, len(events))
        self.assertFalse(any(isinstance(value, list) for value in vars(trace).values()))


    def test_repeated_nonfinite_event_timestamp_still_deduplicates_tokens(self):
        trace = obs.RequestTrace()
        for _ in range(3):
            events = [NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8),
                      event('KV_LOAD_FINISHED', float('nan'))]
            trace.observe(output(events=events))
        trace.finish('success')
        self.assertEqual(self.traces[-1].kv_loaded_tokens, 8)
        self.assertIsNone(self.traces[-1].kv_seconds)
        json.dumps(dict(self.traces[-1].otel_span.attributes), allow_nan=False)


    def test_growing_cumulative_events_process_only_appended_boundaries(self):
        trace = obs.RequestTrace()
        events = [NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=4)]
        trace.observe(output(events=events))
        trace.observe(output(events=events))
        events.append(event('KV_LOAD_FINISHED', 12))
        trace.observe(output(events=events))
        events.extend([NS(type=NS(name='KV_LOAD_STARTED'), timestamp=14, value=8),
                       event('KV_LOAD_FINISHED', 17)])
        trace.observe(output(events=events))
        trace.observe(output(events=events))
        trace.finish('success')
        self.assertEqual(self.traces[-1].kv_seconds, 5)
        self.assertEqual(self.traces[-1].kv_loaded_tokens, 12)


    def test_changed_event_batches_are_marked_untrusted_without_replay(self):
        trace = obs.RequestTrace()
        trace.observe(output(events=[NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8)]))
        trace.observe(output(events=[]))
        trace.observe(output(events=[event('KV_LOAD_FINISHED', 13)]))
        trace.finish('success')
        self.assertIsNone(self.traces[-1].kv_seconds)
        self.assertIsNone(self.traces[-1].kv_loaded_tokens)
        self.assertFalse(self.traces[-1].engine_events_trusted)


    def test_missing_latest_token_timestamp_invalidates_decode(self):
        for metrics in (None, NS(token_timestamp=float('inf'), engine_events=[], cached_tokens=0)):
            trace = obs.RequestTrace()
            trace.observe(output([1], 20, [event('SCHEDULED', 18)]))
            trace.observe(NS(token_ids=[2], req_metrics=metrics, status='FINISH'))
            trace.finish('success')
            row = self.traces[-1]
            self.assertEqual(obs.interval(row.scheduled, row.first_token), 2)
            self.assertIsNone(obs.interval(row.first_token, row.last_token))
            self.assertIsNone(row.otel_span.attributes.get('lmdeploy.tpot_seconds'))
            self.assertIsNone(row.last_token)
            self.assertIsNone(row.output_tokens)


    def test_generator_cleanup_failure_is_not_logged_as_success(self):
        trace = obs.RequestTrace()
        @obs.trace_generate
        async def generate(self, request):
            try:
                yield NS(finish_reason='stop')
            finally:
                raise RuntimeError('cleanup failed')
        async def run():
            generator = generate(None, NS(trace=trace, consumed=False))
            await anext(generator)
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                await generator.aclose()
        asyncio.run(run())
        self.assertEqual(self.traces[-1].engine_status, 'error')


    def test_coalesced_sse_records_share_one_accepted_send_time(self):
        observer = obs.OutputObservation(0)
        body = (b'data: {"choices":[{"delta":{"reasoning":"x"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"y"}}]}\n\ndata: [DONE]\n\n')
        with patch.object(obs.time, 'monotonic', return_value=10.) as clock:
            observer.feed(body)
        clock.assert_called_once()
        row = observer.summary()
        self.assertEqual(row['reasoning_to_content_seconds'], 0)
        self.assertEqual(row['max_output_gap_seconds'], 0)
        self.assertEqual(row['first_output_seconds'], 10)


    def test_deeply_malformed_sse_recovers_without_retaining_payload(self):
        observer = obs.OutputObservation(0)
        observer.feed(b'data: ' + b'[' * 2000 + b']' * 2000 + b'\n\ndata: [DONE]\n\n')
        self.assertTrue(observer.terminal)
        self.assertFalse(observer.buffer)
        self.assertIsNone(observer.first_output)

    def test_nonstream_http_200_reflects_engine_outcome_across_tasks(self):
        async def run():
            for result in ('cancelled', 'error', 'success'):
                async def app(scope, receive, send):
                    async def branch():
                        obs.RequestTrace().finish(result)
                    await asyncio.create_task(branch())
                    await send({'type': 'http.response.start', 'status': 200,
                                'headers': [(b'content-type', b'application/json')]})
                    await send({'type': 'http.response.body', 'body': b'{"result":"unchanged"}',
                                'more_body': False})
                sent = []
                async def send(message):
                    sent.append(message)
                async def receive():
                    return {'type': 'http.request', 'body': b''}
                with patch.object(otel, 'initialize'):
                    middleware = obs.RequestTraceMiddleware(app)
                await middleware({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions'}, receive, send)
                self.assertEqual(self.http_spans[-1].attributes.get('lmdeploy.outcome'), result)
                self.assertEqual(sent[-1]['body'], b'{"result":"unchanged"}')
                self.assertIsNone(obs.request_outcome_context.get())
        asyncio.run(run())


class TargetedReviewTests(RecordTestCase):
    def test_load_cancellation_is_not_a_store_failure(self):
        trace = obs.RequestTrace()
        trace.observe(output(events=[NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8),
                                     event('KV_LOAD_CANCELLED', 12)]))
        trace.finish('cancelled')
        row = self.traces[-1]
        self.assertEqual(row.kv_status, 'cancelled')
        self.assertEqual(row.kv_seconds, 2)
        self.assertEqual(row.kv_loaded_tokens, 0)


    def test_shrinking_or_restarted_event_stream_never_replays_totals(self):
        for latest in ([event('KV_LOAD_FINISHED', 13)], [event('QUEUED', 11), event('SCHEDULED', 14)]):
            trace = obs.RequestTrace()
            trace.observe(output(events=[NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8),
                                         event('KV_LOAD_FINISHED', 13)]))
            trace.observe(output([1], 15, latest))
            trace.finish('success')
            row = self.traces[-1]
            self.assertFalse(row.engine_events_trusted)
            self.assertIsNone(row.kv_loaded_tokens)
            self.assertIsNone(row.kv_seconds)
            self.assertIsNone(row.output_tokens)


    def test_overlapping_loads_are_untrusted_instead_of_overwritten(self):
        trace = obs.RequestTrace()
        trace.observe(output(events=[NS(type=NS(name='KV_LOAD_STARTED'), timestamp=10, value=8),
                                     NS(type=NS(name='KV_LOAD_STARTED'), timestamp=11, value=4),
                                     event('KV_LOAD_FINISHED', 14)]))
        trace.finish('success')
        self.assertFalse(self.traces[-1].engine_events_trusted)
        self.assertIsNone(self.traces[-1].kv_seconds)
        self.assertIsNone(self.traces[-1].kv_loaded_tokens)


    def test_cancelled_lookup_preserves_elapsed_time_without_claiming_miss(self):
        trace = obs.RequestTrace()
        trace.observe(output(events=[event('KV_LOOKUP_STARTED', 10), event('KV_LOOKUP_CANCELLED', 12)]))
        trace.finish('cancelled')
        self.assertEqual(self.traces[-1].lookup_seconds, 2)
        self.assertIsNone(self.traces[-1].store_matched_tokens)


    def test_disabled_generate_returns_original_generator(self):
        async def original(self, request):
            yield NS(finish_reason=None)
        decorated = obs.trace_generate(original)
        async def run():
            generator = decorated(None, NS(trace=None, consumed=False))
            self.assertIs(generator.ag_code, original.__code__)
            await generator.aclose()
        asyncio.run(run())
        self.assertEqual(self.traces, [])


    def test_library_preprocessing_requires_explicit_opt_in(self):
        @obs.trace_preprocess
        async def prepare(engine):
            return 'unchanged'
        async def run():
            with patch.object(obs, 'RequestTrace') as factory:
                self.assertEqual(await prepare(NS(request_trace_enabled=False)), 'unchanged')
                factory.assert_not_called()
        asyncio.run(run())

    def test_only_otlp_endpoint_enables_cached_flag(self):
        source = (SOURCE / 'lmdeploy/trace.py').read_text()
        for env in ({}, {'LMDEPLOY_TRACE_ENABLED': '1', 'LMDEPLOY_TRACE_LOG_DIR': '/unused'}):
            module = {}
            with patch.dict('os.environ', env, clear=True):
                exec(compile(source, 'trace.py', 'exec'), module)
            self.assertFalse(module['ENABLED'])
        for endpoint in ('OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'):
            module = {}
            with patch.dict('os.environ', {endpoint: 'http://collector.example:4317'}, clear=True):
                exec(compile(source, 'trace.py', 'exec'), module)
            with patch.dict('os.environ', {}, clear=True):
                self.assertTrue(module['ENABLED'])

    def test_byte_filter_never_classifies_payload_strings_as_errors(self):
        observer = obs.OutputObservation(0)
        observer.feed(b'data: {"choices":[{"delta":{"content":"response.failed reasoning_content error"}}]}\n')
        self.assertIsNotNone(observer.content_first)
        self.assertIsNone(observer.reasoning_first)
        self.assertFalse(observer.error)
        escaped = obs.OutputObservation(0)
        escaped.feed(b'data: {"choices":[{"delta":{"\\u0063ontent":"ok"}}]}\n')
        self.assertIsNotNone(escaped.content_first)


class RealImportSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable; real runtime import pending')
    def test_real_import_graph_in_fresh_interpreter(self):
        import subprocess
        subprocess.run([sys.executable, '-c',
                        'import lmdeploy.serve.core.async_engine; '
                        'import lmdeploy.serve.openai.api_server; '
                        'import lmdeploy.pytorch.kv_connector.mooncake.store.scheduler; '
                        'import lmdeploy.pytorch.paging.kv_load_coordinator'],
                       cwd=SOURCE, check=True, capture_output=True, text=True)


class OptionalCORSRuntimeTests(RecordTestCase):
    @unittest.skipUnless(importlib.util.find_spec('starlette') and importlib.util.find_spec('httpx'),
                         'Starlette/httpx unavailable; real CORS test pending')
    def test_browser_can_read_outer_trace_response_header(self):
        from httpx import ASGITransport, AsyncClient
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def endpoint(request):
            return JSONResponse({'ok': True})

        app = Starlette(routes=[Route('/v1/completions', endpoint, methods=['POST'])])
        app.add_middleware(CORSMiddleware, allow_origins=['https://client.example'], expose_headers=[obs.HEADER])
        app = obs.RequestTraceMiddleware(app)
        async def request():
            async with AsyncClient(transport=ASGITransport(app),
                                   base_url='http://testserver', trust_env=False) as client:
                return await client.post('/v1/completions',
                                         headers={'Origin': 'https://client.example',
                                                  'x-request-id': 'req_browser_123456'})
        response = asyncio.run(request())
        self.assertEqual(response.status_code, 200)
        self.assertIn(obs.HEADER, response.headers)
        self.assertIn(obs.HEADER, response.headers['access-control-expose-headers'].lower())
        self.assertEqual(self.http_spans[0].attributes['request.id'], 'req_browser_123456')


class PromptIdentityTests(RecordTestCase):
    def run_app(self, app):
        async def run():
            sent = []
            async def send(message):
                sent.append(message)
            async def receive():
                return {'type': 'http.disconnect'}
            with patch.object(otel, 'initialize'):
                middleware = obs.RequestTraceMiddleware(app)
            await middleware({'type': 'http', 'method': 'POST', 'path': '/v1/completions'}, receive, send)
            self.assertIsNone(obs.request_outcome_context.get())
            return sent
        return asyncio.run(run())

    def test_lazy_preprocessing_reuses_the_header_id_for_the_actual_prompt(self):
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            trace = obs.RequestTrace()
            trace.finish('success')
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        sent = self.run_app(app)
        ids = {s.attributes.get('request.id') for s in self.exporter.get_finished_spans()}
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids, {None})
        self.assertFalse(any(k == obs.HEADER.encode() for k, _ in sent[0]['headers']))

    def test_sequential_batch_outputs_follow_explicit_wire_indices(self):
        @obs.trace_generate
        async def generate(engine, request):
            yield NS(finish_reason=None)
            yield NS(finish_reason='stop')
        async def app(scope, receive, send):
            first_trace = obs.RequestTrace()
            first_trace.start = 0
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            for index in range(2):
                # The second prompt can be prepared after headers were sent.
                trace = first_trace if index == 0 else obs.RequestTrace()
                trace.start = 0
                step = 0
                async for _ in generate(None, NS(trace=trace, consumed=False)):
                    key = 'reasoning_content' if index == 0 and step == 0 else 'content'
                    payload = {'choices': [{'index': index, 'delta': {key: 'private'}}]}
                    body = b'data: ' + json.dumps(payload).encode() + b'\n\n'
                    with patch.object(obs.time, 'monotonic', return_value=2 + index * 4 + step * 2):
                        await send({'type': 'http.response.body', 'body': body, 'more_body': True})
                    step += 1
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        self.run_app(app)
        outputs = self.prompt_spans
        self.assertEqual(len(outputs), 2)
        first, second = outputs
        self.assertNotEqual(first.context.span_id, second.context.span_id)
        self.assertIsNone(first.attributes.get('request.id'))
        self.assertEqual(first.attributes.get('lmdeploy.reasoning_first_seconds'), 2)
        self.assertEqual(first.attributes.get('lmdeploy.content_first_seconds'), 4)
        self.assertIsNone(second.attributes.get('lmdeploy.reasoning_first_seconds'))
        self.assertEqual(second.attributes.get('lmdeploy.content_first_seconds'), 6)
        self.assertEqual({t.request_id for t in self.traces},
                         {first.attributes.get('request.id'), second.attributes.get('request.id')})
        self.assertNotIn('private', json.dumps(self.exported_attributes()))

    def test_prepared_but_unconsumed_prompts_are_recorded_on_disconnect(self):
        async def app(scope, receive, send):
            obs.RequestTrace()
            obs.RequestTrace()
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            self.run_app(app)
        engines = self.traces
        self.assertEqual(len(engines), 2)
        self.assertTrue(all(r.engine_status == 'cancelled' and r.output_tokens is None for r in engines))
        self.assertTrue(all(not r.token_counts_complete for r in engines))
        self.assertIsNone(obs.request_outcome_context.get())

    def test_unmapped_or_ambiguous_choices_do_not_invent_prompt_observations(self):
        async def app(scope, receive, send):
            # Duplicate indices are ambiguous; unknown and malformed indices
            # must likewise contribute only to the HTTP observation.
            obs.RequestTrace(prompt_index=0)
            obs.RequestTrace(prompt_index=0)
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            for index in (0, 9, [0], None):
                body = b'data: ' + json.dumps({'choices': [
                    {'index': index, 'delta': {'reasoning_content': 'private'}}]}).encode() + b'\n\n'
                for offset in range(0, len(body), 7):
                    await send({'type': 'http.response.body', 'body': body[offset:offset + 7], 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        self.run_app(app)
        outputs = self.prompt_spans
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(r.attributes.get('lmdeploy.reasoning_first_seconds') is None for r in outputs))


class SSEFramingTests(unittest.TestCase):
    def test_invalid_choice_index_only_contributes_to_http_output(self):
        for index in (-1, True, 'private-index'):
            events = []
            observer = obs.OutputObservation(0, on_output=lambda *args: events.append(args))
            observer.feed(('data: ' + json.dumps({'choices': [{'index': index, 'text': 'private'}]})
                           + '\n\n').encode())
            self.assertIsNotNone(observer.first_output)
            self.assertEqual(events, [])

    def test_oversized_terminal_events_keep_outcome_without_parsing_content(self):
        for event in ('response.completed', 'response.incomplete', 'response.failed'):
            frame = (f'event: {event}\ndata: {{"type":"{event}","output":"'.encode()
                     + b'x' * (1024 * 1024) + b'"}\n\n')
            for bodies in ([frame], [frame[i:i + 4096] for i in range(0, len(frame), 4096)]):
                observer = obs.OutputObservation(0)
                for body in bodies:
                    observer.feed(body)
                self.assertTrue(observer.terminal)
                self.assertEqual(observer.error, event == 'response.failed')
                self.assertIsNone(observer.first_output)
                self.assertFalse(observer.buffer)

    def test_complete_fragmented_and_coalesced_frames_have_same_semantics(self):
        frames = [
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"\\u4e2d\\u6587"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"response.failed error"}}]}\n\n',
            b'data: {"choices":[{"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        variants = [frames, [b''.join(frames)],
                    [frame[i:i + 3] for frame in frames for i in range(0, len(frame), 3)],
                    [frame.replace(b'\n', b'\r\n') for frame in frames]]
        for bodies in variants:
            observer = obs.OutputObservation(0)
            with patch.object(obs.time, 'monotonic', return_value=10.):
                for body in bodies:
                    observer.feed(body)
            self.assertTrue(observer.terminal)
            self.assertFalse(observer.error)
            self.assertFalse(observer.cancelled)
            self.assertEqual(observer.first_output, 10.)
            self.assertEqual(observer.reasoning_first, 10.)
            self.assertEqual(observer.content_first, 10.)
            self.assertEqual(observer.max_output_gap, 0.)
            self.assertFalse(observer.buffer)

    def test_oversized_complete_frame_is_discarded_then_valid_frame_recovers(self):
        observer = obs.OutputObservation(0)
        frame = b'data: {"choices":[{"delta":{"content":"' + b'x' * (1024 * 1024) + b'"}}]}\n\n'
        observer.feed(frame)
        self.assertIsNone(observer.first_output)
        self.assertFalse(observer.buffer)
        with patch.object(obs.time, 'monotonic', return_value=10.):
            observer.feed(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        self.assertEqual(observer.first_output, 10.)


if __name__ == '__main__':
    unittest.main()


class ReviewCorrelationTests(RecordTestCase):
    def run_request(self, count=3, header=None, response_id=None, status=200, traceparent=None):
        async def app(scope, receive, send):
            for _ in range(count):
                trace = obs.RequestTrace()
                self.assertIsNone(trace.engine_status)
                trace.finish('success')
            headers = [(b'x-request-id', response_id)] if response_id is not None else []
            await send({'type': 'http.response.start', 'status': status, 'headers': headers})
            await send({'type': 'http.response.body', 'body': b'{}'})
        async def run():
            sent = []
            async def send(message):
                sent.append(message)
            async def receive():
                return {'type': 'http.request', 'body': b''}
            headers = [(b'x-request-id', header)] if header is not None else []
            if traceparent is not None:
                headers.append((b'traceparent', traceparent.encode()))
            with patch.object(otel, 'initialize'):
                await obs.RequestTraceMiddleware(app)(
                    {'type': 'http', 'method': 'POST', 'path': '/v1/completions',
                     'headers': headers}, receive, send)
            return sent
        return asyncio.run(run())


    def test_rejected_request_has_http_context_without_invented_request_id(self):
        sent = self.run_request(count=0, status=400)
        span, = self.http_spans
        self.assertTrue(span.context.is_valid)
        self.assertEqual(span.attributes['http.response.status_code'], 400)
        self.assertNotIn('request.id', span.attributes)
        self.assertEqual(sent[0]['headers'], [])

    def test_invalid_id_count_warn_once_and_existing_response_header_preserved(self):
        with patch.object(otel, '_invalid_request_ids', 0):
            with self.assertLogs(otel.__name__, level='WARNING') as logs:
                for header in (b'short', b'has.dot:123456789', b'x'*65):
                    sent = self.run_request(count=0, header=header, response_id=b'upstream-response')
                    self.assertIn((b'x-request-id', b'upstream-response'), sent[0]['headers'])
            self.assertEqual(len(logs.output), 1)
            self.assertNotIn('has.dot', str(logs.output))
            self.assertEqual(otel.invalid_request_id_count(), 3)
            self.run_request(count=0)
            self.assertEqual(otel.invalid_request_id_count(), 3)
        sent = self.run_request(header=b'req_valid_12345678', response_id=b'app-response')
        self.assertIn((b'x-request-id', b'app-response'), sent[0]['headers'])
        self.assertEqual(self.http_spans[-1].attributes['request.id'], 'req_valid_12345678')

    def test_explicit_prompt_positions_survive_reordered_preprocessing(self):
        @obs.trace_preprocess
        async def prepare():
            return NS(input_token_len=1, gen_config=None)
        async def run():
            token = obs.request_outcome_context.set({'traces': [], 'request_id': None})
            try:
                second = await prepare(trace_prompt_index=1)
                first = await prepare(trace_prompt_index=0)
                self.assertEqual((first.trace.prompt_index, second.trace.prompt_index), (0, 1))
            finally:
                obs.request_outcome_context.reset(token)
        asyncio.run(run())


    def test_generate_wrapper_preserves_async_iteration_contract(self):
        import inspect
        closed = []
        @obs.trace_generate
        async def generate(engine, request):
            try:
                yield NS(finish_reason=None)
            finally:
                closed.append(True)
        self.assertTrue(inspect.isasyncgenfunction(inspect.unwrap(generate)))
        async def run():
            trace = obs.RequestTrace()
            iterator = generate(None, NS(trace=trace, consumed=False))
            self.assertTrue(inspect.isasyncgen(iterator))
            await anext(iterator)
            await iterator.aclose()
            self.assertEqual(trace.engine_status, 'cancelled')
        asyncio.run(run())
        self.assertEqual(closed, [True])
        source = ast.parse((SOURCE / 'lmdeploy/serve/core/async_engine.py').read_text())
        engine = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncEngine')
        for name, decorator in (('preprocess', 'trace_preprocess'), ('generate', 'trace_generate')):
            method = next(n for n in engine.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
            self.assertIn(decorator, [ast.unparse(d) for d in method.decorator_list])
