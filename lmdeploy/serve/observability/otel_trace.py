# Copyright (c) OpenMMLab. All rights reserved.
"""Optional OTLP traces.

SDK imports and exporter threads are startup-only.
"""
import functools
import logging
import math
import os
import re
import socket
import threading
import time

ENDPOINT = os.getenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT') or os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
_provider = _tracer = None
_pid = None
_init_lock = threading.Lock()
_SECRET = re.compile(r'authorization:|bearer\s+|(?<![A-Za-z0-9])sk-(?:ant-)?[A-Za-z0-9_-]+', re.I)


_failed = False


def fail_open(func):
    @functools.wraps(func)
    def call(*args, **kwargs):
        global _failed
        try:
            return func(*args, **kwargs)
        except Exception:
            if not _failed:
                logging.getLogger(__name__).warning('Trace instrumentation failed; inference continues')
                _failed = True
            return None
    return call


def initialize():
    if not ENDPOINT or (_tracer is not None and _pid == os.getpid()):
        return
    with _init_lock:
        if _tracer is None or _pid != os.getpid():
            _initialize()


def _initialize():
    global _provider, _tracer, _pid
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
    protocol = os.getenv('OTEL_EXPORTER_OTLP_TRACES_PROTOCOL',
                         os.getenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc'))
    if protocol == 'grpc':
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    elif protocol == 'http/protobuf':
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    else:
        raise ValueError('Trace protocol must be grpc or http/protobuf')
    service = metadata_string(os.getenv('OTEL_SERVICE_NAME', 'lmdeploy-full'))
    if service is None or not service.strip():
        raise ValueError('Trace service.name must be nonempty, printable and at most 1024 characters')
    resource = Resource.create({'service.name': service,
                                'service.instance.id': f'{socket.gethostname()}:{os.getpid()}'})
    # This implementation is the non-P/D engine, regardless of external env overrides.
    resource = resource.merge(Resource({'inference.engine': 'lmdeploy', 'inference.role': 'full'}))
    allowed = {'service.name', 'service.instance.id', 'deployment.environment',
               'k8s.cluster.name', 'k8s.namespace.name', 'k8s.pod.name', 'k8s.node.name',
               'inference.engine', 'inference.role'}
    resource = Resource({k: v for k, v in resource.attributes.items()
                         if k in allowed and metadata_string(v) is not None})
    capacity = int(os.getenv('OTEL_BSP_MAX_QUEUE_SIZE', '2048'))
    batch = int(os.getenv('OTEL_BSP_MAX_EXPORT_BATCH_SIZE', '512'))
    delay = int(os.getenv('OTEL_BSP_SCHEDULE_DELAY', '1000'))
    if not 0 < batch <= capacity or delay <= 0:
        raise ValueError('Trace queue/batch sizes and delay must be positive; batch must fit queue')
    provider = TracerProvider(resource=resource, sampler=ParentBased(ALWAYS_ON))
    processor = BatchSpanProcessor(OTLPSpanExporter(),
        max_queue_size=capacity, max_export_batch_size=batch, schedule_delay_millis=delay)
    provider.add_span_processor(processor)
    _provider, _tracer, _pid = provider, provider.get_tracer('lmdeploy.request_trace'), os.getpid()


_invalid_request_ids = 0
_invalid_id_lock = threading.Lock()


@fail_open
def request_id(headers):
    global _invalid_request_ids
    value = headers.get('x-request-id')
    if value is None:
        return None
    if re.fullmatch(r'[A-Za-z0-9_-]{16,64}', value) and not _SECRET.search(value):
        return value
    with _invalid_id_lock:
        _invalid_request_ids += 1
        first = _invalid_request_ids == 1
    if first:
        logging.getLogger(__name__).warning(
            'Invalid x-request-id omitted from trace metadata; count available via invalid_request_id_count()')
    return None


def invalid_request_id_count():
    """Process-local rejected header count; never retains header values."""
    return _invalid_request_ids


def metadata_string(value):
    """Bound model/session metadata without exporting controls or
    credentials."""
    if isinstance(value, str) and value and len(value) <= 1024 and value.isprintable() and not _SECRET.search(value):
        return value
    return None


@fail_open
def begin_http(headers, path, start_ns, outcome):
    if _tracer is None:
        return None
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    # Vendor state is optional and can carry unvalidated values into OTLP trace_state.
    context = TraceContextTextMapPropagator().extract(
        {'traceparent': headers.get('traceparent', '')}, context=Context())
    parent = trace.get_current_span(context).get_span_context()
    attrs = {'http.request.method': 'POST', 'url.path': path, 'http.route': path,
             'trace.context.missing': not parent.is_valid}
    if parent.is_valid:
        outcome['upstream_context'] = context
    rid = outcome['request_id']
    if rid is not None:
        attrs['request.id'] = rid
    # The platform owns affinity normalization (§5.5): never export or hash it.
    # Its presence blocks a lower-priority OpenClaw fallback. The gateway must
    # inject its normalized session as x-session-id for this case.
    session = headers.get('x-session-id')
    if not session and not headers.get('x-session-affinity'):
        session = headers.get('x-openclaw-session-key')
    # Preserve explicit IDs, including punctuation; apply only metadata safety bounds.
    session = metadata_string(session)
    if session is not None:
        outcome['session_id'] = session
        attrs['session.id'] = session
    span = _tracer.start_span('llm_request', context=context, kind=trace.SpanKind.SERVER,
                              start_time=start_ns, attributes=attrs)
    span.add_event('request.received', timestamp=start_ns)
    return span


@fail_open
def begin_prompt(parent, start_ns, index, rid):
    if _tracer is None or (index is not None and parent is None):
        return None
    from opentelemetry import trace
    from opentelemetry.context import Context
    attrs = {'lmdeploy.prompt_index': index} if index is not None else {'trace.context.missing': True}
    if rid is not None:
        attrs['request.id'] = rid
    name = 'llm_request' if index is None else 'inference'
    return _tracer.start_span(name, context=trace.set_span_in_context(parent, Context())
                             if parent is not None else Context(), start_time=start_ns, attributes=attrs)


@fail_open
def correlation(span):
    if span is None:
        return {}
    ctx = span.get_span_context()
    return dict(trace_id=f'{ctx.trace_id:032x}', span_id=f'{ctx.span_id:016x}')


@fail_open
def attributes(span, values):
    if span is not None and span.is_recording():
        span.set_attributes({k: v for k, v in values.items() if isinstance(v, (str, int, bool, float))
                             and (not isinstance(v, float) or math.isfinite(v))
                             and (not isinstance(v, str) or metadata_string(v) is not None)})


@fail_open
def finish_engine(t, status):
    span = t.otel_span
    if span is None or not span.is_recording():
        return
    end = t.end_ns
    attrs = {'gen_ai.provider.name': 'lmdeploy',
             'gen_ai.operation.name': (t.outcome or {}).get('operation', 'text_completion'),
             'gen_ai.request.model': t.model, 'gen_ai.response.model': t.model,
             'lmdeploy.base_model': t.base_model if t.base_model != t.model else None,
             'lmdeploy.choice_index': t.choice_index,
             'gen_ai.usage.input_tokens': t.prompt_tokens, 'gen_ai.usage.output_tokens': t.output_tokens,
             'gen_ai.usage.cached_tokens': t.cached_tokens, 'lmdeploy.backend': t.backend,
             'lmdeploy.finish_reason': t.finish_reason,
             'lmdeploy.token_counts_complete': t.token_counts_complete,
             'gen_ai.latency.e2e': t.elapsed_seconds,
             'gen_ai.latency.time_to_first_token':
                 t.first_token - t.queued if t.first_token is not None and t.queued is not None
                 and t.first_token >= t.queued else None,
             'lmdeploy.queue_seconds': t.scheduled - t.queued
                 if t.scheduled is not None and t.queued is not None and t.scheduled >= t.queued else None,
             'lmdeploy.preprocess_seconds': t.preprocessed - t.start
                 if t.preprocessed is not None and t.preprocessed >= t.start else None,
             'lmdeploy.handle_wait_seconds': t.handle_ready - t.generate_start
                 if t.handle_ready is not None and t.generate_start is not None
                 and t.handle_ready >= t.generate_start else None,
             'lmdeploy.tpot_seconds': (t.last_token - t.first_token) / (t.generated_tokens - 1)
                 if t.first_token is not None and t.last_token is not None
                 and t.last_token >= t.first_token and t.generated_tokens > 1 else None,
             'lmdeploy.store_lookup_seconds': t.lookup_seconds,
             'lmdeploy.store_matched_tokens': t.store_matched_tokens,
             'lmdeploy.kv_load_seconds': t.kv_seconds, 'lmdeploy.kv_loaded_tokens': t.kv_loaded_tokens,
             'lmdeploy.kv_load_status': t.kv_status, 'lmdeploy.outcome': status}
    attributes(span, attrs)
    from opentelemetry import trace
    # Phases share the propagated router/Envoy parent when available. Direct
    # requests and library calls have a local request span to parent them.
    outcome = t.outcome or {}
    ctx = outcome.get('upstream_context')
    if ctx is None:
        ctx = trace.set_span_in_context(outcome.get('otel_span') or span)
    for name, start, stop in (
            ('model.prefill', int(t.scheduled*1e9) if t.scheduled is not None else None,
             int(t.first_token*1e9) if t.first_token is not None else None),
            ('model.decode', int(t.first_token*1e9) if t.first_token is not None else None,
             int(t.last_token*1e9) if t.last_token is not None else None)):
        if start is not None and stop is not None and t.start_ns <= start <= stop <= end:
            child_attrs = {'inference.engine': 'lmdeploy', 'inference.role': 'full'}
            if t.request_id is not None:
                child_attrs['request.id'] = t.request_id
            if 'session_id' in outcome:
                child_attrs['session.id'] = outcome['session_id']
            if t.prompt_index is not None:
                child_attrs['lmdeploy.prompt_index'] = t.prompt_index
            if t.choice_index is not None:
                child_attrs['lmdeploy.choice_index'] = t.choice_index
            child = _tracer.start_span(name, context=ctx, start_time=start, attributes=child_attrs)
            finish(child, status, stop)
    if t.first_token is not None and t.start_ns <= int(t.first_token*1e9) <= end:
        span.add_event('llm.first_token', timestamp=int(t.first_token*1e9))
    if t.outcome is None:
        finish(span, status, end)


@fail_open
def finish(span, status, end_ns=None, http_status=None, output=None):
    if span is None:
        return
    from opentelemetry.trace import Status, StatusCode
    end_ns = end_ns or time.time_ns()
    attributes(span, {'http.response.status_code': http_status, 'lmdeploy.outcome': status})
    if output:
        attributes(span, {f'lmdeploy.{key}': value for key, value in output.items()})
    failed = status != 'success' or (http_status is not None and http_status >= 500)
    span.set_status(Status(StatusCode.ERROR if failed else StatusCode.OK))
    if status != 'success':
        span.add_event('error', timestamp=end_ns)  # No exception messages or payloads.
    span.end(end_time=end_ns)


@fail_open
def finish_http(outcome, status, http_status, elapsed, observation):
    span = outcome['otel_span']
    if span is None or not span.is_recording():
        return
    start_ns = outcome['start_ns']
    end_ns = start_ns + int(elapsed * 1e9)
    attrs = {'gen_ai.latency.e2e': elapsed, 'lmdeploy.http_total_seconds': elapsed}
    try:
        traces = outcome['traces']
        if traces:
            attrs['lmdeploy.token_counts_complete'] = status == 'success' and all(
                t.token_counts_complete for t in traces)
            # API choices share an input but each contributes its own output.
            inputs = [t for t in traces if t.choice_index in (None, 0)]
            for key, field, selected in (('input_tokens', 'prompt_tokens', inputs),
                                         ('cached_tokens', 'cached_tokens', inputs),
                                         ('output_tokens', 'output_tokens', traces)):
                values = [getattr(t, field) for t in selected]
                if values and all(v is not None for v in values):
                    attrs['gen_ai.usage.' + key] = sum(values)
            models = {t.model for t in traces if t.model}
            if len(models) == 1:
                attrs['gen_ai.request.model'] = attrs['gen_ai.response.model'] = models.pop()
            start = start_ns / 1e9
            end = end_ns / 1e9
            first = min((t.first_token for t in traces if t.first_token is not None and start <= t.first_token <= end),
                        default=None)
            if first is not None:
                attrs['gen_ai.latency.time_to_first_token'] = first - start
        if observation.sse and observation.terminal:
            span.add_event('llm.stream.end', timestamp=end_ns)
        attrs.update({f'lmdeploy.{k}': v for k, v in observation.summary().items()})
    finally:
        attributes(span, attrs)
        finish(span, status, end_ns=end_ns, http_status=http_status)


def _after_fork():
    global _init_lock, _invalid_id_lock, _invalid_request_ids
    _invalid_request_ids = 0
    _init_lock = threading.Lock()
    _invalid_id_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)
