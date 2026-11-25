"""Instrumentation faults and request failures must not poison later traces."""
import asyncio
from unittest.mock import patch

import test_request_trace as fixtures
from test_request_trace import RecordTestCase, obs, otel


class IsolationTests(RecordTestCase):
    def request(self, error=None):
        sent = []
        payload = b'data: {"choices":[{"delta":{"content":"private-output"}}]}\n\n'

        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            await send({'type': 'http.response.body', 'body': payload, 'more_body': True})
            if error is not None:
                raise error
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})

        async def send(message):
            sent.append(message)

        async def run():
            try:
                await obs.RequestTraceMiddleware(app)(
                    {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions'}, None, send)
            except BaseException as caught:
                self.assertIs(caught, error)
                raise
            finally:
                self.assertIsNone(obs.request_outcome_context.get())

        if error is None:
            asyncio.run(run())
        else:
            with self.assertRaises(type(error)):
                asyncio.run(run())
        return b''.join(message.get('body', b'') for message in sent)

    def test_failed_and_cancelled_requests_keep_later_traces_working(self):
        for error, expected in ((ValueError('private-error'), 'error'),
                                (asyncio.CancelledError(), 'cancelled'), (None, 'success')):
            self.request(error)
            span = self.http_spans[-1]
            self.assertEqual(span.attributes['lmdeploy.outcome'], expected)
            self.assertEqual(span.attributes['http.response.status_code'], 200)
            self.assertEqual(span.status.status_code.name, 'OK' if error is None else 'ERROR')
        self.assertEqual(len(self.http_spans), 3)
        self.assertNotIn('private', str(self.exported_attributes()))

    def test_output_observation_fault_preserves_response_and_original_exception(self):
        normal = self.request()
        with patch.object(obs, '_decode_json', side_effect=RuntimeError('observer failed')):
            self.assertEqual(self.request(), normal)
            self.request(ValueError('original application error'))
        self.assertEqual(self.request(), normal)
        self.assertEqual([s.attributes['lmdeploy.outcome'] for s in self.http_spans],
                         ['success', 'success', 'error', 'success'])

    def test_span_creation_failure_does_not_disable_later_requests(self):
        with patch.object(otel._tracer, 'start_span', side_effect=RuntimeError('SDK failure')):
            body = self.request()
            self.request(ValueError('original application error'))
        self.assertEqual(self.http_spans, [])
        self.assertEqual(self.request(), body)
        self.assertEqual(self.http_spans[-1].attributes['lmdeploy.outcome'], 'success')

    def test_finish_fault_does_not_mask_error_or_leak_context(self):
        from opentelemetry.sdk.trace import Span
        with patch.object(Span, 'end', side_effect=RuntimeError('processor failed')):
            body = self.request()
            self.request(ValueError('original application error'))
            self.request(asyncio.CancelledError())
        self.assertEqual(self.request(), body)
        self.assertEqual(self.http_spans[-1].attributes['lmdeploy.outcome'], 'success')

    def test_trace_creation_fault_preserves_preprocessing(self):
        marker = object()

        @obs.trace_preprocess
        async def prepare():
            return marker

        with patch.object(obs, 'RequestTrace', side_effect=RuntimeError('metadata failed')):
            self.assertIs(asyncio.run(prepare()), marker)
        self.request()
        self.assertEqual(self.http_spans[-1].attributes['lmdeploy.outcome'], 'success')

    def test_native_metadata_fault_preserves_generation_and_next_request(self):
        fixtures.EngineIntegrationTests.setUpClass()
        fixture = fixtures.EngineIntegrationTests()

        async def run():
            engine, prepared, handle = fixture.engine()
            result = [out async for out in engine.generate(prepared, stream_response=False)]
            self.assertFalse(handle.stream_output)
            self.assertEqual(engine.session_mgr.sessions, {})
            return result, prepared.trace

        # The diagnostic path and even its warning handler may fail; neither
        # owns the backend iterator, output bytes, token usage or session cleanup.
        with patch.object(obs.RequestTrace, '_observe_events', side_effect=ValueError('bad metadata')), \
                patch.object(otel, '_failed', False), \
                patch.object(otel.logging.Logger, 'warning', side_effect=OSError('broken handler')):
            broken, first = asyncio.run(run())
        healthy, second = asyncio.run(run())
        self.assertEqual(broken, healthy)
        self.assertTrue(first._native_failed)
        self.assertFalse(second._native_failed)
        self.assertEqual(first.engine_status, 'success')
        self.assertEqual(first.output_tokens, healthy[-1].generate_token_len)
        self.assertNotIn('lmdeploy.queue_seconds', first.otel_span.attributes)
        self.assertEqual(second.otel_span.attributes['lmdeploy.queue_seconds'], 2)
