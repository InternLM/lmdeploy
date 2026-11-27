# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in request phase timing.

No model inputs or generated text are recorded.
"""

import asyncio
import contextvars
import functools
import json
import math
import time
from dataclasses import asdict

from lmdeploy.serve.observability import otel_trace, output_trace
from lmdeploy.trace import ENABLED

request_outcome_context = contextvars.ContextVar('observability_request_outcome', default=None)
HEADER = 'x-request-id'
_decode_json = json.JSONDecoder().decode


def enabled():
    return ENABLED


@otel_trace.fail_open
def _annotate_frame(frame, payload, terminal):
    return output_trace.ObservedString(frame, output_trace.summarize(payload, terminal=terminal))


def trace_frame(frame, payload=None, *, terminal=False):
    """Attach flags to a serialized frame only when this request is
    observed."""
    if not enabled():
        return frame
    outcome = request_outcome_context.get()
    if outcome is None or not outcome.get('observe', True):
        return frame
    annotated = _annotate_frame(frame, payload, terminal)
    return frame if annotated is None else annotated


def supported_engine(backend, config):
    """This observer describes full-engine serving, including explicit library
    opt-in."""
    role = getattr(config, 'role', None)
    return backend != 'pytorch' or role is None or getattr(role, 'name', role) == 'Hybrid'


def interval(start, end):
    if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end < start:
        return None
    return end - start


def _event_key(event):
    timestamp = event.timestamp
    return (event.type.name, timestamp if math.isfinite(timestamp) else None,
            getattr(event, 'value', None))


class RequestTrace:
    """Fixed-size request state; output chunks are observed but never
    retained."""

    def __init__(self, prompt_index=None, choice_index=None):
        self.outcome = request_outcome_context.get()
        if self.outcome is not None and self.outcome.get('closed'):
            self.outcome = None
        self.request_id = self.outcome.get('request_id') if self.outcome is not None else None
        self.start = time.monotonic()
        http_start_ns = self.outcome.get('start_ns') if self.outcome is not None else None
        self.start_ns = (http_start_ns + int((self.start - self.outcome['start']) * 1e9)
                         if http_start_ns is not None else time.time_ns())
        self.end_ns = None
        self.elapsed_seconds = None
        self.model = None
        self.base_model = None
        self.otel_span = None
        self.preprocessed = self.generate_start = self.handle_ready = None
        self.queued = self.scheduled = self.first_token = self.last_token = None
        self.lookup_start = self.lookup_seconds = self.store_matched_tokens = None
        self.kv_start = self.kv_seconds = self.kv_loaded_tokens = None
        self.kv_pending_tokens = None
        self.kv_status = None
        self.kv_failed = False
        self.kv_active = False
        self.prompt_tokens = self.cached_tokens = None
        self.prefill_metrics = None
        self.generated_tokens = 0  # Native count, including stop tokens; used for timing.
        self.output_tokens = None  # API count from GenOut, after stop-token filtering.
        self.finish_reason = None
        self.finished = False
        self.engine_status = None
        self.backend = None
        self.terminal = False
        self.event_cursor = 0
        self.last_event_key = None
        self.engine_events_trusted = True
        self._native_failed = False
        self.prompt_index = None
        self.choice_index = choice_index if self.outcome is not None else None
        self.output_observation = None
        if self.outcome is not None:
            traces = self.outcome['traces']
            if prompt_index is None:
                prompt_index = 0 if self.choice_index is not None else len(traces)
            self.prompt_index = prompt_index
            traces.append(self)
            # Protocol indices, never the last generator to yield, own output.
            index = self.choice_index if self.choice_index is not None else self.prompt_index
            owners = self.outcome.setdefault('output_traces', {})
            owners[index] = self if index not in owners else None
        self.otel_span = otel_trace.begin_prompt(
            self.outcome.get('otel_span') if self.outcome is not None else None,
            self.start_ns, self.prompt_index, self.request_id)
        otel_trace.attributes(self.otel_span, {'lmdeploy.choice_index': self.choice_index})

    @property
    def token_counts_complete(self):
        return (self.terminal and self.engine_status == 'success'
                and all(value is not None for value in (self.prompt_tokens, self.output_tokens, self.cached_tokens)))

    def observe(self, output):
        if self._native_failed:
            return
        try:
            metrics = output.req_metrics
            status = getattr(output, 'status', None)
            status = getattr(status, 'name', status)
            token_count = len(output.token_ids)
            self.terminal |= status == 'FINISH'
            if metrics is not None:
                self._observe_events(metrics.engine_events)
                prefill = getattr(metrics, 'prefill', None)
                if prefill is not None and self.prefill_metrics is None:
                    self.prefill_metrics = asdict(prefill)
            if token_count:
                # A missing timestamp on the newest batch invalidates the last-token
                # boundary; an older valid timestamp must not understate Decode.
                self.last_token = None
                if (metrics is not None and math.isfinite(metrics.token_timestamp)
                        and metrics.token_timestamp > 0):
                    # A final-only multi-token batch supplies no first-token boundary.
                    # Preserve backend mode and leave the split unknown in that case.
                    if self.generated_tokens == 0:
                        first = getattr(metrics, 'first_token_timestamp', 0)
                        if math.isfinite(first) and 0 < first <= metrics.token_timestamp:
                            self.first_token = first
                        elif token_count == 1:
                            self.first_token = metrics.token_timestamp
                    self.last_token = metrics.token_timestamp
            self.generated_tokens += token_count
        except Exception:
            # This input's native metrics are incomplete. Preserve API output
            # and counters, and keep later requests fully observable.
            self._native_failed = True
            self._invalidate_events()
            self.first_token = self.last_token = None
            self.prefill_metrics = None
            otel_trace.report_failure()

    def _invalidate_events(self):
        self.engine_events_trusted = False
        self.queued = self.scheduled = None
        self.lookup_start = self.lookup_seconds = self.store_matched_tokens = None
        self.kv_start = self.kv_seconds = self.kv_loaded_tokens = self.kv_pending_tokens = None
        self.kv_status = None
        self.kv_active = False

    def _observe_events(self, events):
        count = len(events)
        if not self.engine_events_trusted or not count:
            return
        if self.event_cursor:
            if count < self.event_cursor or _event_key(events[self.event_cursor - 1]) != self.last_event_key:
                self._invalidate_events()
                return
        start = self.event_cursor
        for index in range(start, count):
            event = events[index]
            name = event.type.name
            value = getattr(event, 'value', None)
            stamp = event.timestamp if math.isfinite(event.timestamp) and event.timestamp > 0 else None
            if name == 'QUEUED' and self.queued is None:
                self.queued = stamp
            elif name == 'SCHEDULED' and self.scheduled is None:
                self.scheduled = stamp
            elif name == 'KV_LOOKUP_STARTED':
                self.lookup_start = stamp
            elif name in ('KV_LOOKUP_FINISHED', 'KV_LOOKUP_CANCELLED'):
                duration = interval(self.lookup_start, stamp)
                if duration is not None:
                    self.lookup_seconds = (self.lookup_seconds or 0) + duration
                self.lookup_start = None
                if value is not None and name == 'KV_LOOKUP_FINISHED':
                    self.store_matched_tokens = max(self.store_matched_tokens or 0, value)
            elif name == 'KV_LOAD_STARTED':
                if self.kv_active:
                    self._invalidate_events()
                    return
                self.kv_active = True
                self.kv_start = stamp
                self.kv_pending_tokens = value
                if not self.kv_failed:
                    self.kv_status = 'pending'
            elif name in ('KV_LOAD_FINISHED', 'KV_LOAD_FAILED', 'KV_LOAD_CANCELLED'):
                duration = interval(self.kv_start, stamp)
                if duration is not None:
                    self.kv_seconds = (self.kv_seconds or 0) + duration
                self.kv_start = None
                if name == 'KV_LOAD_FAILED':
                    self.kv_status = 'failed'
                    self.kv_failed = True
                    self.kv_loaded_tokens = self.kv_loaded_tokens or 0
                elif name == 'KV_LOAD_CANCELLED':
                    if not self.kv_failed:
                        self.kv_status = 'cancelled'
                    self.kv_loaded_tokens = self.kv_loaded_tokens or 0
                else:
                    if not self.kv_failed:
                        self.kv_status = 'success'
                    if self.kv_pending_tokens is not None:
                        self.kv_loaded_tokens = (self.kv_loaded_tokens or 0) + self.kv_pending_tokens
                self.kv_pending_tokens = None
                self.kv_active = False
        if count > start:
            self.event_cursor = count
            self.last_event_key = _event_key(events[count - 1])

    @otel_trace.fail_open
    def finish(self, status):
        if self.finished:
            return
        self.finished = True
        self.engine_status = status
        if self.outcome is not None and status in ('error', 'cancelled'):
            self.outcome[status] = True
        self.elapsed_seconds = interval(self.start, time.monotonic())
        self.end_ns = (self.start_ns + int(self.elapsed_seconds * 1e9)
                       if self.otel_span is not None and self.elapsed_seconds is not None else None)
        otel_trace.finish_engine(self, status)


@otel_trace.fail_open
def _create_trace(engine, prompt_index, choice_index, model):
    trace = RequestTrace(prompt_index=prompt_index, choice_index=choice_index)
    trace.backend = getattr(engine, 'backend', None)
    trace.base_model = otel_trace.metadata_string(getattr(engine, 'model_name', None))
    trace.model = otel_trace.metadata_string(model or trace.base_model)
    return trace


@otel_trace.fail_open
def _finish_prompt(trace, status, http_status):
    if not trace.finished:
        trace.finish('error' if status == 'error' else 'cancelled')
    observed = trace.output_observation
    otel_trace.finish(trace.otel_span, trace.engine_status, end_ns=trace.end_ns,
                      http_status=http_status, output=observed.summary() if observed is not None else None)


def trace_preprocess(func):
    @functools.wraps(func)
    async def wrapped(*args, **kwargs):
        prompt_index = kwargs.pop('trace_prompt_index', None)
        choice_index = kwargs.pop('trace_choice_index', None)
        model = kwargs.pop('trace_model', None)
        outcome = request_outcome_context.get()
        if (not enabled() or (args and not getattr(args[0], 'request_trace_enabled', True))
                or (outcome is not None and outcome.get('observe') is False)):
            return await func(*args, **kwargs)
        engine = args[0] if args else None
        trace = _create_trace(engine, prompt_index, choice_index, model or kwargs.get('adapter_name'))
        if trace is None:
            return await func(*args, **kwargs)
        try:
            request = await func(*args, **kwargs)
        except BaseException as exc:
            trace.finish('cancelled' if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else 'error')
            raise
        trace.preprocessed = time.monotonic()
        if model is None:
            trace.model = otel_trace.metadata_string(getattr(request, 'adapter_name', None) or trace.model)
        trace.prompt_tokens = request.input_token_len
        request.trace = trace
        return request
    return wrapped


def trace_generate(func):
    async def traced(self, request, trace, *args, **kwargs):
        trace.generate_start = time.monotonic()
        status = 'cancelled'
        generator = func(self, request, *args, **kwargs)
        try:
            async for output in generator:
                # Use the same cumulative counters consumed by all API adapters.
                trace.prompt_tokens = getattr(output, 'input_token_len', trace.prompt_tokens)
                trace.output_tokens = getattr(output, 'generate_token_len', trace.output_tokens)
                trace.cached_tokens = getattr(output, 'cached_tokens', trace.cached_tokens)
                if output.finish_reason is not None:
                    trace.finish_reason = output.finish_reason
                    status = ('cancelled' if output.finish_reason == 'abort' else
                              'error' if output.finish_reason == 'error' else 'success')
                yield output
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException:
            status = 'error'
            raise
        finally:
            try:
                await generator.aclose()
            except (asyncio.CancelledError, GeneratorExit):
                if status != 'success':
                    status = 'cancelled'
                raise
            except BaseException:
                status = 'error'
                raise
            finally:
                trace.finish(status)
    @functools.wraps(func)
    def wrapped(self, request, *args, **kwargs):
        trace = getattr(request, 'trace', None)
        if trace is None or getattr(request, 'consumed', False):
            return func(self, request, *args, **kwargs)
        return traced(self, request, trace, *args, **kwargs)
    return wrapped


class RequestTraceMiddleware:
    """Pure ASGI wrapper.

    Passes bodies unchanged; observes server send times.
    """

    def __init__(self, app):
        self.app = app
        otel_trace.initialize()

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        root = scope.get('root_path', '')
        if root and (path == root or path.startswith(root + '/')):
            path = path[len(root):]
        if (not enabled() or scope['type'] != 'http' or scope.get('method') != 'POST'
                or path not in ('/v1/chat/completions', '/v1/completions',
                                            '/v1/responses', '/v1/messages', '/generate')):
            return await self.app(scope, receive, send)
        # HTTP correlation is read-only; prompt state and spans are independent.
        outcome = {'error': False, 'cancelled': False, 'traces': [], 'output_traces': {}, 'closed': False}
        start = time.monotonic()
        received_ns = time.time_ns()
        outcome['start'], outcome['start_ns'] = start, received_ns
        wanted = {b'traceparent', b'x-request-id', b'x-session-id', b'x-session-affinity', b'x-openclaw-session-key'}
        # Match Router's HeaderMap::get: the first field wins, even if it is invalid.
        headers = {}
        for key, value in scope.get('headers', []):
            key = key.lower()
            if key in wanted:
                headers.setdefault(key.decode('ascii'), value.decode('latin1'))
        outcome['operation'] = ('chat' if path in ('/v1/chat/completions', '/v1/messages')
                                else 'text_completion')
        outcome['request_id'] = otel_trace.request_id(headers)
        outcome['otel_span'] = otel_trace.begin_http(headers, path, received_ns, outcome)
        span = outcome['otel_span']
        outcome['observe'] = span is not None and span.is_recording()
        if not outcome['observe']:
            # Keep correlation and the decision in child tasks without allocating prompt
            # observers or inspecting any response body for an unsampled request.
            token = request_outcome_context.set(outcome)

            async def correlated_send(message):
                if message['type'] == 'http.response.start' and outcome['request_id'] is not None:
                    response_headers = message.get('headers', [])
                    if not any(k.lower() == b'x-request-id' for k, _ in response_headers):
                        message = dict(message, headers=[*response_headers,
                            (b'x-request-id', outcome['request_id'].encode())])
                await send(message)

            try:
                return await self.app(scope, receive, correlated_send)
            finally:
                outcome['closed'] = True
                request_outcome_context.reset(token)
        outcome_token = request_outcome_context.set(outcome)
        status_code = None
        complete = False
        def observe_output(index, reasoning, content, tool, observed_at):
            if index is None:
                trace = outcome['traces'][0] if len(outcome['traces']) == 1 else None
            else:
                trace = outcome['output_traces'].get(index)
            if trace is not None:
                if trace.output_observation is None:
                    trace.output_observation = OutputObservation(trace.start)
                trace.output_observation.observe(reasoning, content, tool, observed_at)

        observation = OutputObservation(start, observe_output)
        status = 'cancelled'

        async def traced_send(message):
            nonlocal status_code, complete
            if message['type'] == 'http.response.start':
                message = dict(message)
                message['headers'] = list(message.get('headers', []))
                if (outcome['request_id'] is not None
                        and not any(k.lower() == HEADER.encode() for k, _ in message['headers'])):
                    message['headers'].append((HEADER.encode(), outcome['request_id'].encode()))
                await send(message)
                status_code = message['status']
                observation.sse = any(k.lower() == b'content-type' and b'text/event-stream' in v
                                      for k, v in message['headers'])
                return
            await send(message)
            if message['type'] == 'http.response.body':
                if observation.sse:
                    observation.feed(message.get('body', b''))
                if not message.get('more_body', False):
                    complete = True

        def result():
            if (observation.error or outcome['error']
                    or (status_code is not None and status_code >= 400)):
                return 'error'
            if observation.cancelled or outcome['cancelled']:
                return 'cancelled'
            if (complete and not observation.sse) or observation.terminal:
                return 'success'
            return 'cancelled'

        try:
            await self.app(scope, receive, traced_send)
            status = result()
        except (asyncio.CancelledError, GeneratorExit, OSError):
            status = result()
            raise
        except BaseException:
            status = 'error'
            raise
        finally:
            try:
                for trace in outcome['traces']:
                    _finish_prompt(trace, status, status_code)
            finally:
                otel_trace.finish_http(outcome, status, status_code, time.monotonic() - start, observation)
                outcome['closed'] = True
                outcome['output_traces'].clear()
                outcome['traces'].clear()
                request_outcome_context.reset(outcome_token)


class OutputObservation:
    """Bounded SSE observation, including fragments split across ASGI
    messages."""

    def __init__(self, start, on_output=None):
        self.start = start
        self.on_output = on_output
        self.sse = self.terminal = self.error = self.cancelled = False
        self.buffer = bytearray()
        self.skipping = False
        self.event_failed = self.event_terminal = self.event_data = False
        self.pending_terminal = self.pending_error = self.pending_cancelled = False
        self.first_output = self.reasoning_first = self.reasoning_last = self.content_first = None
        self.previous_output = self.max_output_gap = None

    @otel_trace.fail_open
    def feed(self, body):
        # One accepted ASGI body has one observation time, even if it carries
        # many SSE records. Avoid splitlines() allocations and repeated copying
        # of a partial line split over many small body messages.
        observed_at = time.monotonic()
        metadata = getattr(body, 'trace_metadata', None)
        if isinstance(metadata, output_trace.OutputMetadata):
            self.observe_metadata(metadata, observed_at)
            return
        # The server normally sends one complete SSE record per body. Keep
        # split, coalesced, CRLF and oversized records on the bounded slow path.
        if (not self.buffer and not self.skipping and len(body) <= 1024 * 1024
                and body.startswith(b'data:') and body.endswith(b'\n\n')
                and body.find(b'\n') == len(body) - 2):
            self._line(body[:-2], observed_at)
            self._line(b'', observed_at)
            return
        offset = 0
        while offset < len(body):
            newline = body.find(b'\n', offset)
            end = len(body) if newline < 0 else newline + 1
            size = end - offset
            if len(self.buffer) + size > 1024 * 1024:
                # Retain only the presence of data for an oversized event. SSE
                # dispatch requires data and a blank line, not just event:.
                prefix = (self.buffer[:5] + body[offset:offset + 5])[:5]
                self.event_data |= prefix == b'data:'
                self.buffer.clear()
                self.skipping = True
            if not self.skipping:
                if not self.buffer and newline >= 0:
                    self._line(body[offset:end].rstrip(b'\r\n'), observed_at)
                else:
                    self.buffer.extend(memoryview(body)[offset:end])
                    if newline >= 0:
                        self._line(self.buffer.rstrip(b'\r\n'), observed_at)
            if newline >= 0:
                self.buffer.clear()
                self.skipping = False
            offset = end

    def _line(self, line, observed_at):
        if not line:
            if self.event_data:
                self.terminal |= self.event_terminal or self.pending_terminal
                self.error |= self.pending_error or (self.event_failed and not self.pending_cancelled)
                self.cancelled |= self.pending_cancelled
            self.event_failed = self.event_terminal = self.event_data = False
            self.pending_terminal = self.pending_error = self.pending_cancelled = False
            return
        if line.startswith(b'event:'):
            event = line[6:].strip()
            self.event_terminal = event in (b'response.completed', b'response.incomplete',
                                            b'response.failed', b'message_stop')
            self.event_failed = event in (b'response.failed', b'error')
            return
        if not line.startswith(b'data:'):
            return
        self.event_data = True
        data = line[5:].strip()
        if data == b'[DONE]':
            self.pending_terminal = True
            return
        try:
            # SSE is UTF-8; avoid json.loads(bytes)' repeated encoding detection.
            obj = _decode_json(data.decode('utf-8'))
        except (ValueError, UnicodeError, RecursionError):
            return
        if not isinstance(obj, dict):
            return
        self.observe_metadata(output_trace.summarize(obj), observed_at, defer_outcome=True)

    def observe_metadata(self, metadata, observed_at, defer_outcome=False):
        if defer_outcome:
            # Raw SSE may end between a data line and the dispatch delimiter.
            # Adapter metadata already describes complete accepted frames.
            self.pending_terminal |= metadata.terminal
            self.pending_error |= metadata.error
            self.pending_cancelled |= metadata.cancelled
        else:
            self.terminal |= metadata.terminal
            self.error |= metadata.error
            self.cancelled |= metadata.cancelled
        reasoning = content = tool = False
        for index, has_reasoning, has_content, has_tool in metadata.outputs:
            reasoning |= has_reasoning
            content |= has_content
            tool |= has_tool
            if self.on_output is not None and (index is None or index >= 0):
                self.on_output(index, has_reasoning, has_content, has_tool, observed_at)
        self.observe(reasoning, content, tool, observed_at)

    def observe(self, reasoning, content, tool, now):
        """Accumulate flags only; never retain or reparse output payloads."""
        if reasoning:
            self.reasoning_first = now if self.reasoning_first is None else self.reasoning_first
            self.reasoning_last = now
        if content and self.content_first is None:
            self.content_first = now
        if reasoning or content or tool:
            if self.first_output is None:
                self.first_output = now
            if self.previous_output is not None:
                gap = now - self.previous_output
                if self.max_output_gap is None or gap > self.max_output_gap:
                    self.max_output_gap = gap
            self.previous_output = now

    @otel_trace.fail_open
    def summary(self):
        return dict(first_output_seconds=interval(self.start, self.first_output),
                    max_output_gap_seconds=self.max_output_gap,
                    reasoning_first_seconds=interval(self.start, self.reasoning_first),
                    reasoning_last_seconds=interval(self.start, self.reasoning_last),
                    content_first_seconds=interval(self.start, self.content_first),
                    reasoning_to_content_seconds=interval(self.reasoning_first, self.content_first))
