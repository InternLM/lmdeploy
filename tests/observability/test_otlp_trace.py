"""Real SDK, protobuf transport, ASGI context and privacy tests; no GPU
required."""
import asyncio
import importlib.util
import json
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from test_request_trace import SOURCE, obs, otel

try:
    import grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as pb
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc as rpc
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
    from opentelemetry.trace import SpanKind, StatusCode
    SDK = True
except ImportError:
    SDK = False


@unittest.skipUnless(SDK, 'install lmdeploy[serve] to test SDK and OTLP wire transport')
class OTLPTests(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(sampler=ParentBased(ALWAYS_ON))
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.patch = patch.object(otel, '_tracer', self.provider.get_tracer('test'))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.provider.shutdown)

    def request(self, headers=(), status=200, count=2, crash=False, stream=False):
        records, sent = [], []
        async def app(scope, receive, send):
            for _ in range(count):
                t = obs.RequestTrace()
                t.model = 'test-model'
                t.backend = 'turbomind'
                t.terminal = True
                t.prompt_tokens, t.generated_tokens, t.output_tokens, t.cached_tokens = 10, 2, 2, 0
                t.preprocessed = t.start + .00001
                stamp = t.start_ns / 1e9
                t.queued, t.scheduled = stamp + .00002, stamp + .00003
                t.first_token, t.last_token = stamp + .00004, stamp + .00005
                time.sleep(.001)
                t.finish('error' if crash else 'success')
            if crash:
                raise ValueError('Bearer private-completion')
            response_headers = [(b'content-type', b'text/event-stream')] if stream else []
            await send({'type': 'http.response.start', 'status': status, 'headers': response_headers})
            body = b'data: [DONE]\n\n' if stream else b'private-completion'
            await send({'type': 'http.response.body', 'body': body})
        async def receive():
            return {'type': 'http.request', 'body': b'private-prompt'}
        async def send(message):
            sent.append(message)
        def record(event, rid, **kw):
            records.append(dict(event=event, request_id=rid, **kw))

        async def run():
            with patch.object(obs, 'enabled', return_value=True), patch.object(obs, 'get_writer'), \
                 patch.object(obs, 'emit', side_effect=record):
                await obs.RequestTraceMiddleware(app)(
                    {'type': 'http', 'method': 'POST', 'path': '/v1/completions', 'headers': headers}, receive, send)
        if crash:
            with self.assertRaisesRegex(ValueError, 'private-completion'):
                asyncio.run(run())
        else:
            asyncio.run(run())
        self.assertIsNone(obs.request_outcome_context.get())
        return records, sent

    def test_parent_prompt_stages_and_readonly_ids(self):
        trace_id, parent_id = 'a'*32, 'b'*16
        records, sent = self.request([(b'traceparent', f'00-{trace_id}-{parent_id}-01'.encode()),
            (b'x-request-id', b'req_gateway_123456'), (b'x-session-id', b'session.prod:42'),
            (b'x-session-affinity', b'private-affinity')])
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == 'llm_request')
        prompts = [s for s in spans if s.name == 'inference']
        self.assertEqual(len(spans), 7)
        self.assertEqual(root.parent.span_id, int(parent_id, 16))
        self.assertEqual(root.context.trace_id, int(trace_id, 16))
        self.assertEqual(root.attributes['gen_ai.usage.output_tokens'], 4)
        self.assertTrue(all(p.parent.span_id == root.context.span_id for p in prompts))
        self.assertEqual(len({p.context.span_id for p in prompts}), 2)
        phases = [s for s in spans if s.name in ('model.prefill', 'model.decode')]
        self.assertCountEqual([s.name for s in phases], ['model.prefill', 'model.decode'] * 2)
        for s in spans:
            self.assertLessEqual(s.start_time, s.end_time)
        for phase in phases:
            self.assertEqual(phase.kind, SpanKind.INTERNAL)
            self.assertEqual(phase.attributes['inference.engine'], 'lmdeploy')
            self.assertEqual(phase.attributes['inference.role'], 'full')
            self.assertNotIn('envoy.error', phase.attributes)
            self.assertEqual(phase.parent, root.parent)
            self.assertEqual(phase.context.trace_id, root.context.trace_id)
            self.assertIn(phase.attributes['lmdeploy.prompt_index'], (0, 1))
            self.assertEqual(phase.attributes['request.id'], 'req_gateway_123456')
            self.assertEqual(phase.attributes['session.id'], 'session.prod:42')
            self.assertFalse(any(k.startswith(('gen_ai.usage.', 'gen_ai.latency.')) for k in phase.attributes))
        self.assertTrue(all(r['request_id'] == 'req_gateway_123456' for r in records))
        self.assertEqual(len({r['span_id'] for r in records}), 2)
        self.assertEqual(sent[-1]['body'], b'private-completion')
        wire = str([dict(s.attributes) for s in spans])
        self.assertNotIn('private', wire)

    def test_first_correlation_header_wins(self):
        for first, expected in [(b'req_gateway_first', 'req_gateway_first'), (b'bad', None)]:
            with self.subTest(first=first):
                self.exporter.clear()
                self.request([(b'X-Request-ID', first), (b'x-request-id', b'req_gateway_later'),
                              (b'x-session-id', b'session.first'), (b'X-Session-ID', b'session.later')], count=0)
                root, = self.exporter.get_finished_spans()
                self.assertEqual(root.attributes.get('request.id'), expected)
                self.assertEqual(root.attributes['session.id'], 'session.first')

    def test_direct_request_phases_use_local_parent(self):
        self.request(count=1)
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == 'llm_request')
        self.assertIsNone(root.parent)
        self.assertCountEqual([s.name for s in spans], ['llm_request', 'inference', 'model.prefill', 'model.decode'])
        for child in (s for s in spans if s.name != 'llm_request'):
            self.assertEqual(child.parent.span_id, root.context.span_id)
            self.assertEqual(child.context.trace_id, root.context.trace_id)
            self.assertNotIn('session.id', child.attributes)

    def test_library_phases_have_request_router_parent(self):
        t = obs.RequestTrace()
        stamp = t.start_ns / 1e9
        t.queued = t.scheduled = stamp + .00001
        t.first_token, t.last_token = stamp + .00002, stamp + .00003
        time.sleep(.001)
        with patch.object(obs, 'emit'):
            t.finish('success')
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == 'llm_request')
        self.assertIsNone(root.parent)
        self.assertTrue(root.attributes['trace.context.missing'])
        self.assertEqual(len(spans), 3)
        for phase in (s for s in spans if s.name != 'llm_request'):
            self.assertEqual(phase.parent.span_id, root.context.span_id)
            self.assertNotIn('lmdeploy.prompt_index', phase.attributes)

    def test_explicit_session_id_preserved_without_request_id_format_restrictions(self):
        for value in ('session_prod_42', 'session.prod:42', 'tenant/session@example.com',
                      ' session.prod:42 ', 's' * 129, 's' * 1024):
            with self.subTest(value_length=len(value)):
                self.exporter.clear()
                _, sent = self.request([(b'x-session-id', value.encode()),
                                        (b'x-session-affinity', b'private-affinity'),
                                        (b'x-request-id', b'req_gateway_123456')], count=0)
                span = self.exporter.get_finished_spans()[0]
                self.assertEqual(span.attributes['session.id'], value)
                self.assertEqual(span.attributes['request.id'], 'req_gateway_123456')
                self.assertNotIn('session.affinity', span.attributes)
                self.assertNotIn('session.affinity_hashed', span.attributes)
                self.assertEqual(sent[-1]['body'], b'private-completion')

    def test_explicit_session_id_keeps_size_control_and_secret_protection(self):
        for value in ('', 's' * 1025, 'session\n42', 'session\x0042', 'session\x7f42',
                      'session\x8542', 'Authorization: private', 'Bearer private',
                      'sk-ant-private', 'session_sk-private'):
            with self.subTest(value_length=len(value)):
                self.exporter.clear()
                self.request([(b'x-session-id', value.encode('latin1'))], count=0)
                self.assertNotIn('session.id', self.exporter.get_finished_spans()[0].attributes)

    def test_session_fallback_respects_priority_and_affinity_privacy(self):
        cases = [
            ({'x-openclaw-session-key': 'session.prod:42'}, 'session.prod:42'),
            ({'x-session-id': 'explicit', 'x-openclaw-session-key': 'fallback'}, 'explicit'),
            ({'x-session-affinity': 'private-routing', 'x-openclaw-session-key': 'fallback'}, None),
            ({'x-openclaw-session-key': 'Bearer private'}, None),
            ({'x-session-id': 'Bearer private', 'x-openclaw-session-key': 'fallback'}, None),
        ]
        for headers, expected in cases:
            with self.subTest(headers=list(headers)):
                self.exporter.clear()
                self.request([(k.encode(), v.encode()) for k, v in headers.items()], count=0)
                attrs = self.exporter.get_finished_spans()[0].attributes
                self.assertEqual(attrs.get('session.id'), expected)
                self.assertNotIn('session.affinity', attrs)
                self.assertNotIn('session.affinity_hashed', attrs)
                self.assertNotIn('private', str(dict(attrs)))

    def test_configurable_service_keeps_real_engine_role_and_privacy(self):
        with patch.dict(os.environ, {'OTEL_SERVICE_NAME': 'registered-lmdeploy-service',
                'OTEL_RESOURCE_ATTRIBUTES': ('inference.engine=other,inference.role=decode,k8s.pod.name=test-pod,'
                    'service.namespace=inference,deployment.environment.name=prod,k8s.pod.uid=pod-uid,'
                    'authorization=private,prompt=private'),
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:1'}, clear=True), \
                patch.object(otel, '_provider', None), patch.object(otel, '_tracer', None), \
                patch.object(otel, '_pid', None):
            otel._initialize()
            try:
                attrs = otel._provider.resource.attributes
                self.assertEqual(attrs['service.name'], 'registered-lmdeploy-service')
                self.assertEqual(attrs['inference.engine'], 'lmdeploy')
                self.assertEqual(attrs['inference.role'], 'full')
                self.assertEqual(attrs['k8s.pod.name'], 'test-pod')
                for key in ('service.namespace', 'deployment.environment.name', 'k8s.pod.uid',
                            'telemetry.sdk.language'):
                    self.assertNotIn(key, attrs)
                self.assertNotIn('authorization', attrs)
                self.assertNotIn('prompt', attrs)
            finally:
                otel._provider.shutdown()
        for service in ('', 'bad\nservice', 'Bearer private', 's' * 1025):
            with self.subTest(service_length=len(service)), \
                    patch.dict(os.environ, {'OTEL_SERVICE_NAME': service}, clear=True):
                with self.assertRaises(ValueError):
                    otel._initialize()

    def test_resource_metadata_has_the_same_privacy_and_size_bounds(self):
        for value in ('node\nprivate', 'x' * 1025, 'Bearer private'):
            with self.subTest(value_length=len(value)), patch.dict(os.environ, {
                    'OTEL_RESOURCE_ATTRIBUTES': 'k8s.node.name=' + value,
                    'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://127.0.0.1:1'}, clear=True), \
                    patch.object(otel, '_provider', None), patch.object(otel, '_tracer', None), \
                    patch.object(otel, '_pid', None):
                otel._initialize()
                try:
                    self.assertNotIn('k8s.node.name', otel._provider.resource.attributes)
                finally:
                    otel._provider.shutdown()

    def test_exported_usage_matches_stop_filtered_generation_and_http_response(self):
        from test_request_trace import EngineIntegrationTests

        EngineIntegrationTests.setUpClass()
        fixture = EngineIntegrationTests()
        for include_stop in (False, True):
            self.exporter.clear()
            records, sent = [], []

            async def app(scope, receive, send):
                engine, prepared, _ = fixture.engine()
                prepared.gen_config.ignore_eos = False
                prepared.gen_config.stop_token_ids = [12]
                prepared.gen_config.stop_words = []
                prepared.gen_config.include_stop_str_in_output = include_stop
                engine.tokenizer.decode = lambda *a, **kw: '<stop>'
                results = [r async for r in engine.generate(prepared)]
                final = results[-1]
                payload = dict(input_tokens=final.input_token_len, output_tokens=final.generate_token_len,
                               cached_tokens=final.cached_tokens)
                await send(dict(type='http.response.start', status=200, headers=[]))
                await send(dict(type='http.response.body', body=json.dumps(payload).encode()))

            async def send(message):
                sent.append(message)

            async def receive():
                return dict(type='http.request', body=b'')

            async def run():
                with patch.object(obs, 'enabled', return_value=True), patch.object(obs, 'get_writer'), \
                        patch.object(obs, 'emit', side_effect=lambda event, rid, **kw: records.append(kw)):
                    await obs.RequestTraceMiddleware(app)(
                        dict(type='http', method='POST', path='/v1/completions'), receive, send)

            asyncio.run(run())
            usage = json.loads(sent[-1]['body'])
            self.assertEqual(usage['output_tokens'], 2 if include_stop else 1)
            for span in self.exporter.get_finished_spans():
                if span.name in ('inference', 'llm_request'):
                    for field, expected in usage.items():
                        self.assertEqual(span.attributes['gen_ai.usage.' + field], expected)
                    self.assertTrue(span.attributes['lmdeploy.token_counts_complete'])
            self.assertEqual(records[0]['output_tokens'], usage['output_tokens'])

    def test_secret_after_underscore_is_not_exported_as_an_id(self):
        records, sent = self.request([(b'x-request-id', b'req_sk-ant-private-value'),
                                      (b'x-session-id', b'session_sk-private-value')], count=1)
        self.assertTrue(all(r['request_id'] is None for r in records))
        self.assertNotIn(b'x-request-id', dict(sent[0]['headers']))
        for span in self.exporter.get_finished_spans():
            self.assertNotIn('request.id', span.attributes)
            self.assertNotIn('session.id', span.attributes)
        self.assertEqual(otel.request_id({'x-request-id': 'task-valid_12345678'}), 'task-valid_12345678')

    @unittest.skipUnless(importlib.util.find_spec('starlette') and importlib.util.find_spec('httpx'),
                         'Starlette/httpx unavailable')
    def test_framework_500_response_is_observed_without_swallowing_error(self):
        from httpx import ASGITransport, AsyncClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        async def endpoint(request):
            raise ValueError('private application detail')
        records = []
        async def run():
            with patch.object(obs, 'enabled', return_value=True), patch.object(obs, 'get_writer'), \
                 patch.object(obs, 'emit', side_effect=lambda *args, **kw: records.append(kw)):
                app = obs.RequestTraceMiddleware(Starlette(routes=[
                    Route('/v1/completions', endpoint, methods=['POST'])]))
                async with AsyncClient(transport=ASGITransport(app, raise_app_exceptions=False),
                                       base_url='http://testserver', trust_env=False) as client:
                    response = await client.post('/v1/completions')
                    self.assertEqual(response.status_code, 500)
                async with AsyncClient(transport=ASGITransport(app, raise_app_exceptions=True),
                                       base_url='http://testserver', trust_env=False) as client:
                    with self.assertRaisesRegex(ValueError, 'private application detail'):
                        await client.post('/v1/completions')
                self.assertIsNone(obs.request_outcome_context.get())
        asyncio.run(run())
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['http_status'] == 500 and r['status'] == 'error' for r in records))
        for span in self.exporter.get_finished_spans():
            self.assertEqual(span.attributes['http.response.status_code'], 500)
            self.assertEqual(span.status.status_code.name, 'ERROR')
            self.assertNotIn('private application detail', str(span.attributes))

    def test_optional_vendor_state_cannot_leak_values_into_otlp(self):
        self.request([(b'traceparent', ('00-' + 'a'*32 + '-' + 'b'*16 + '-01').encode()),
                      (b'tracestate', b'vendor=sk-ant-private_marker')], count=1)
        for span in self.exporter.get_finished_spans():
            self.assertEqual(span.context.trace_id, int('a'*32, 16))
            self.assertFalse(span.context.trace_state)
            self.assertFalse(span.parent.trace_state)

    def test_missing_invalid_context_and_ids(self):
        self.request([(b'traceparent', b'invalid'), (b'x-request-id', b'../invalid')], count=0, status=400)
        root = self.exporter.get_finished_spans()[0]
        self.assertIsNone(root.parent)
        self.assertTrue(root.attributes['trace.context.missing'])
        self.assertNotIn('request.id', root.attributes)
        self.assertEqual(root.status.status_code.name, 'ERROR')
        self.assertEqual(root.attributes['http.response.status_code'], 400)
        self.assertAlmostEqual(root.attributes['gen_ai.latency.e2e'], (root.end_time - root.start_time) / 1e9)
        self.assertNotIn('gen_ai.latency.time_to_first_token', root.attributes)
        self.assertFalse(any(key.startswith('gen_ai.usage.') for key in root.attributes))

    def test_http_5xx_after_successful_generation(self):
        self.request(count=1, status=503)
        spans = {s.name: s for s in self.exporter.get_finished_spans()}
        for name in ('llm_request', 'inference'):
            self.assertEqual(spans[name].attributes['http.response.status_code'], 503)
            self.assertEqual(spans[name].status.status_code.name, 'ERROR')
        self.assertEqual(spans['inference'].attributes['lmdeploy.outcome'], 'success')
        for name in ('model.prefill', 'model.decode'):
            self.assertEqual(spans[name].status.status_code.name, 'OK')
            self.assertNotIn('envoy.error', spans[name].attributes)

    def test_failed_response_start_omits_http_code(self):
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})

        async def send(message):
            raise BrokenPipeError()

        with patch.object(obs, 'enabled', return_value=True), patch.object(obs, 'get_writer'), \
                patch.object(obs, 'emit') as emit, self.assertRaises(BrokenPipeError):
            asyncio.run(obs.RequestTraceMiddleware(app)(
                {'type': 'http', 'method': 'POST', 'path': '/v1/completions'}, None, send))
        root, = self.exporter.get_finished_spans()
        self.assertEqual(root.status.status_code.name, 'ERROR')
        self.assertNotIn('http.response.status_code', root.attributes)
        self.assertIsNone(emit.call_args.kwargs['http_status'])

    def test_unsampled_parent_and_exception_privacy(self):
        self.request([(b'traceparent', ('00-'+'a'*32+'-'+'b'*16+'-00').encode())])
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)
        self.request(crash=True)
        spans = self.exporter.get_finished_spans()
        self.assertTrue(all(s.status.status_code.name == 'ERROR' for s in spans))
        self.assertNotIn('private-completion', str(spans))

    def test_instrumentation_failure_does_not_replace_result(self):
        with patch.object(otel._tracer, 'start_span', side_effect=RuntimeError('private')):
            _, sent = self.request()
        self.assertEqual(sent[-1]['body'], b'private-completion')

    def test_batch_prompt_ends_before_next_prompt_and_error_events_in_bounds(self):
        self.request()
        spans = self.exporter.get_finished_spans()
        prompts = sorted((s for s in spans if s.name == 'inference'), key=lambda s: s.start_time)
        self.assertLessEqual(prompts[0].end_time, prompts[1].start_time)
        root = next(s for s in spans if s.name == 'llm_request')
        self.assertGreater(root.attributes['gen_ai.latency.time_to_first_token'], 0)
        self.exporter.clear()
        self.request(crash=True)
        for span in self.exporter.get_finished_spans():
            for event in span.events:
                self.assertLessEqual(span.start_time, event.timestamp)
                self.assertLessEqual(event.timestamp, span.end_time)

    def test_wall_clock_adjustment_does_not_corrupt_spans_or_events(self):
        real_time_ns = time.time_ns
        for jump in (-10_000_000_000, 10_000_000_000):
            for crash in (False, True):
                with self.subTest(jump=jump, crash=crash):
                    self.exporter.clear()
                    calls = 0
                    def adjusted_clock():
                        nonlocal calls
                        calls += 1
                        return real_time_ns() + (jump if calls > 2 else 0)
                    with patch.object(obs.time, 'time_ns', side_effect=adjusted_clock):
                        records, _ = self.request(count=1, crash=crash, stream=True)
                    spans = self.exporter.get_finished_spans()
                    root = next(s for s in spans if s.name == 'llm_request')
                    prompt = next(s for s in spans if s.name == 'inference')
                    duration = prompt.attributes['gen_ai.latency.e2e']
                    self.assertGreaterEqual(duration, .001)
                    self.assertAlmostEqual((prompt.end_time - prompt.start_time) / 1e9, duration, places=8)
                    self.assertEqual(records[0]['total_seconds'], duration)
                    self.assertGreaterEqual(root.end_time, prompt.end_time)
                    if not crash:
                        self.assertIn('llm.stream.end', {event.name for event in root.events})
                    for span in spans:
                        self.assertLessEqual(span.start_time, span.end_time)
                        for event in span.events:
                            self.assertLessEqual(span.start_time, event.timestamp)
                            self.assertLessEqual(event.timestamp, span.end_time)

    def test_prompt_starts_share_http_clock_anchor(self):
        real_time_ns = time.time_ns
        for jump in (-10_000_000_000, 10_000_000_000):
            with self.subTest(jump=jump):
                self.exporter.clear()
                calls = 0
                def adjusted_clock():
                    nonlocal calls
                    calls += 1
                    return real_time_ns() + (jump if calls > 1 else 0)
                with patch.object(obs.time, 'time_ns', side_effect=adjusted_clock):
                    self.request(count=2)
                spans = self.exporter.get_finished_spans()
                root = next(s for s in spans if s.name == 'llm_request')
                self.assertEqual(len(spans), 7)
                for span in spans:
                    self.assertLessEqual(root.start_time, span.start_time)
                    self.assertLessEqual(span.end_time, root.end_time)

    def test_failed_http_span_does_not_create_prompt_roots(self):
        original = otel._tracer.start_span
        def start(name, **kwargs):
            if name == 'llm_request':
                raise RuntimeError('failure')
            return original(name, **kwargs)
        with patch.object(otel._tracer, 'start_span', side_effect=start):
            self.request()
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

    def test_scalar_attributes_and_context_cleanup_after_bad_metadata(self):
        span = otel._tracer.start_span('inference')
        otel.attributes(span, {'safe': 1, 'array': ['private'], 'object': {'private': 1}, 'nan': float('nan')})
        otel.attributes(span, {'control': 'bad\rvalue', 'session.id': 'session.prod:42'})
        otel.finish(span, 'success')
        attrs = self.exporter.get_finished_spans()[0].attributes
        self.assertNotIn('control', attrs)
        self.assertEqual(attrs['session.id'], 'session.prod:42')
        self.assertNotIn('array', self.exporter.get_finished_spans()[0].attributes)
        self.assertNotIn('object', self.exporter.get_finished_spans()[0].attributes)
        original = obs.RequestTrace.finish
        def finish(t, status):
            original(t, status)
            t.model = []  # Deliberately break HTTP model aggregation.
        with patch.object(obs.RequestTrace, 'finish', finish):
            _, sent = self.request()
        self.assertEqual(sent[-1]['body'], b'private-completion')
        self.assertIsNone(obs.request_outcome_context.get())

    def test_emitted_fields_match_platform_extension_contract(self):
        contract = json.loads((SOURCE / 'docs/en/serving/request_trace_contract.json').read_text())
        self.request([(b'x-request-id', b'req_contract_123456')])
        for span in self.exporter.get_finished_spans():
            self.assertEqual(contract['span_mapping'][span.name], span.name)
            self.assertTrue(set(span.attributes) <= set(contract['allowed_attributes']))
            self.assertTrue(all(isinstance(v, (str, int, bool, float)) for v in span.attributes.values()))
            self.assertTrue(all(e.name in contract['events'] for e in span.events))
        prompt = next(s for s in self.exporter.get_finished_spans() if s.name == 'inference')
        self.assertGreaterEqual(prompt.attributes['lmdeploy.tpot_seconds'], 0)
        self.assertGreaterEqual(prompt.attributes['lmdeploy.preprocess_seconds'], 0)
        self.assertFalse(any(s.name == 'tokenize' for s in self.exporter.get_finished_spans()))

    def test_concurrent_initialization_creates_one_provider(self):
        calls = []
        def initialize():
            time.sleep(.01)
            calls.append(1)
            otel._tracer = object()
            otel._pid = os.getpid()
        with patch.object(otel, 'ENDPOINT', 'http://unused'), patch.object(otel, '_pid', None), \
                patch.object(otel, '_tracer', None), patch.object(otel, '_initialize', side_effect=initialize):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: otel.initialize(), range(32)))
        self.assertEqual(len(calls), 1)

    def test_token_completeness_is_explicit_for_success_and_failure(self):
        self.request()
        for span in self.exporter.get_finished_spans():
            if span.name in ('inference', 'llm_request'):
                self.assertTrue(span.attributes['lmdeploy.token_counts_complete'])
        self.exporter.clear()
        self.request(crash=True)
        phases = [s for s in self.exporter.get_finished_spans() if s.name in ('model.prefill', 'model.decode')]
        self.assertEqual(len(phases), 4)
        self.assertTrue(all('envoy.error' not in s.attributes for s in phases))
        self.assertTrue(all(s.status.status_code is StatusCode.ERROR for s in phases))
        for span in self.exporter.get_finished_spans():
            if span.name in ('inference', 'llm_request'):
                self.assertFalse(span.attributes['lmdeploy.token_counts_complete'])

    def test_export_queue_does_not_wait_for_exporter(self):
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
        entered, release = threading.Event(), threading.Event()
        class BlockedExporter(SpanExporter):
            def export(self, spans):
                entered.set()
                release.wait(5)
                return SpanExportResult.FAILURE
            def shutdown(self):
                pass
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(BlockedExporter(),
            max_queue_size=8, max_export_batch_size=1, schedule_delay_millis=1))
        try:
            with patch.object(otel, '_tracer', provider.get_tracer('test')):
                self.request(count=1)
                self.assertTrue(entered.wait(1))
                start = time.monotonic()
                for _ in range(4):
                    self.request(count=1)
                self.assertLess(time.monotonic() - start, 1)
        finally:
            release.set()
            provider.shutdown()

    def test_sensitive_metadata_is_not_exported(self):
        self.request([(b'x-request-id', b'sk-private_12345678'),
                      (b'x-session-id', b'sk-ant-private')], count=0)
        root = self.exporter.get_finished_spans()[0]
        self.assertNotIn('request.id', root.attributes)
        self.assertNotIn('session.id', root.attributes)

    def test_real_grpc_and_http_protobuf_batch_export(self):
        for protocol in ('grpc', 'http/protobuf'):
            received = []
            if protocol == 'grpc':
                class Receiver(rpc.TraceServiceServicer):
                    def Export(self, request, context):
                        received.append(request)
                        return pb.ExportTraceServiceResponse()
                server = grpc.server(ThreadPoolExecutor(max_workers=1))
                rpc.add_TraceServiceServicer_to_server(Receiver(), server)
                port = server.add_insecure_port('127.0.0.1:0')
                server.start()
                def stop():
                    server.stop(0).wait()
            else:
                class Handler(BaseHTTPRequestHandler):
                    def do_POST(self):
                        self.server.test.assertEqual(self.path, '/v1/traces')
                        self.server.test.assertEqual(self.headers['Content-Type'], 'application/x-protobuf')
                        received.append(pb.ExportTraceServiceRequest.FromString(self.rfile.read(int(self.headers['Content-Length']))))
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-protobuf')
                        self.end_headers()
                        self.wfile.write(pb.ExportTraceServiceResponse().SerializeToString())
                    def log_message(self, *args):
                        pass
                server = HTTPServer(('127.0.0.1', 0), Handler)
                server.test = self
                port = server.server_port
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                def stop():
                    server.shutdown()
                    server.server_close()
                    thread.join()
            try:
                endpoint = f'http://127.0.0.1:{port}'
                with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': endpoint,
                     'OTEL_EXPORTER_OTLP_PROTOCOL': protocol, 'OTEL_SERVICE_NAME': 'lmdeploy-full',
                     'OTEL_RESOURCE_ATTRIBUTES': (
                         'deployment.environment=test,k8s.cluster.name=cluster,k8s.namespace.name=namespace,'
                         'k8s.pod.name=pod,k8s.node.name=node,session.affinity=private-affinity-resource,'
                         'SESSION.AFFINITY=private-affinity-uppercase,'
                         'session.affinity_hashed=untrusted-affinity-hash,session.id=untrusted-resource-session,'
                         'request.id=untrusted-resource-request,custom.metadata=unapproved-resource'),
                     'NO_PROXY': '*', 'no_proxy': '*'}, clear=True), \
                     patch.object(otel, 'ENDPOINT', endpoint), patch.object(otel, '_pid', None):
                    otel.initialize()
                    provider = otel._provider
                    try:
                        self.request([(b'x-session-id', b'session.prod:42'),
                                      (b'x-request-id', b'req_gateway_123456'),
                                      (b'x-session-affinity', b'private-affinity-header')], count=1)
                        self.assertTrue(provider.force_flush(5000))
                        contract = json.loads((SOURCE / 'docs/en/serving/request_trace_contract.json')
                                              .read_text())
                        for req in received:
                            for rs in req.resource_spans:
                                resource = {a.key: a.value for a in rs.resource.attributes}
                                self.assertEqual(set(resource), set(contract['resource_allowed_attributes']))
                                self.assertEqual(resource['deployment.environment'].string_value, 'test')
                                self.assertEqual(resource['k8s.node.name'].string_value, 'node')
                            wire = req.SerializeToString()
                            for forbidden in (b'private-affinity', b'untrusted-', b'unapproved-resource'):
                                self.assertNotIn(forbidden, wire)
                        spans = [s for req in received for rs in req.resource_spans
                                 for ss in rs.scope_spans for s in ss.spans]
                        self.assertCountEqual([s.name for s in spans],
                                              ['llm_request', 'inference', 'model.prefill', 'model.decode'])
                        for phase in (s for s in spans if s.name.startswith('model.')):
                            attrs = {a.key: a.value for a in phase.attributes}
                            self.assertEqual(attrs['inference.role'].string_value, 'full')
                            self.assertEqual(attrs['inference.engine'].string_value, 'lmdeploy')
                            self.assertNotIn('envoy.error', attrs)
                        self.assertTrue(all(len(s.trace_id) == 16 and len(s.span_id) == 8 for s in spans))
                        root = next(s for s in spans if s.name == 'llm_request')
                        attrs = {a.key: a.value.string_value for a in root.attributes}
                        self.assertEqual(attrs['session.id'], 'session.prod:42')
                        self.assertEqual(attrs['request.id'], 'req_gateway_123456')
                    finally:
                        provider.shutdown()
                        otel._tracer = self.provider.get_tracer('test')
            finally:
                stop()
