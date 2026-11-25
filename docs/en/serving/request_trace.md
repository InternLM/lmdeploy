# OpenTelemetry request tracing

Request phase timing follows vLLM request-level metrics: queue, Prefill to first token, and Decode
from first to last token. Completed phase spans also expose these intervals in the platform
waterfall. The first release covers ordinary generation and the native PyTorch Mooncake Store
connector. PyTorch tracing is enabled only for the Hybrid (full-engine) role; Prefill and Decode
roles do not install this observer. P/D and Proxy tracing are outside this release.
TurboMind supports
ordinary request timing; this change does not add a new Store connector to TurboMind.

## Purpose and implementation

This is the repository-maintained feature usage and implementation guide. It explains the request
flow, source entry points, configuration, exported fields, timing boundaries and operational limits.
It is intended to be read with the feature commit without separate delivery notes.

The feature answers where an inference request spent its time, using existing engine observations
and request-completion summaries. A span is one timed operation in a distributed trace: `trace_id`
identifies the trace, `span_id` identifies an operation, and `parent_span_id` links it to its caller.
These are tracing identifiers; `request_id` remains the read-only gateway business identifier.

1. The API server installs `RequestTraceMiddleware`. It reads the incoming W3C context and allowed
   identifiers, starts the HTTP span, and observes response status and accepted sends.
   The application retains the FastAPI interface; tracing wraps its built middleware stack,
   including framework-generated error responses. With tracing disabled, it uses plain FastAPI.
2. Preprocessing creates one `RequestTrace` per prompt, attached to `PreprocessedRequest.trace`.
   Batch positions and concurrent choice indices are passed explicitly. Generation observes existing
   engine output and events to accumulate token counts, queue/token boundaries and optional
   Store timings.
3. At prompt completion, the observer emits the prompt summary and completed Prefill/Decode spans.
   HTTP completion supplies transport and visible-output summaries. The prompt index links
   each phase to the prompt that produced it.
4. OpenTelemetry's bounded background processor exports completed spans over OTLP. There are no
   per-token spans, synchronous request-path disk/network I/O, extra Store RPCs, or internal
   streaming for nonstream requests. Unknown timing boundaries remain unknown.

| Source entry point                                                                                               | Responsibility                                                                               |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `lmdeploy/trace.py`, `lmdeploy/serve/openai/api_server.py`                                                       | Read startup enable flags and connect tracing to the API server                              |
| `lmdeploy/serve/core/async_engine.py`, `lmdeploy/serve/openai/endpoints/completions.py`                          | Attach per-prompt state, assign batch positions and observe the existing generation path     |
| `lmdeploy/serve/core/chat_runner.py`, `lmdeploy/serve/openai/chat_completions/`                                  | Pass the resolved model and explicit choice identity through existing preprocessing metadata |
| `lmdeploy/serve/observability/request_trace.py`                                                                  | HTTP middleware, timing aggregation, bounded SSE observation and request outcome tracking    |
| `lmdeploy/serve/observability/output_trace.py`                                                                   | Payload-free metadata shared by streaming adapters and the accepted-send observer            |
| `lmdeploy/serve/observability/otel_trace.py`                                                                     | SDK initialization, parent propagation, metadata filtering, spans and OTLP export            |
| `lmdeploy/messages.py`, `lmdeploy/pytorch/messages.py`                                                           | Carry optional Store event boundaries and counts with ordinary engine output                 |
| `lmdeploy/pytorch/kv_connector/mooncake/store/scheduler.py`, `lmdeploy/pytorch/paging/kv_load_coordinator.py`    | Observe existing Store lookup and load lifecycle                                             |
| `lmdeploy/pytorch/trace.py`, `lmdeploy/pytorch/engine/inputs_maker.py`, `lmdeploy/pytorch/engine/engine_loop.py` | Accumulate bounded prefill diagnostics at host dispatch/completion boundaries                |
| `requirements/serve.txt`, `tests/observability/`, `tests/test_lmdeploy/serve/test_request_trace_*.py`            | Serving dependencies, behavioral/transport regression tests and real-package checks          |

## Export traces to a Collector

Install `lmdeploy[serve]` and set the endpoint before starting the API server.
The serving dependencies include the OpenTelemetry SDK and both OTLP exporters;
no separate trace extra is needed. Installing dependencies does not enable tracing.
Set `OTEL_EXPORTER_OTLP_ENDPOINT` in the deployment environment to the selected Collector's
receiver URL. Use the protocol, TLS settings and resource values supplied for that deployment;
the API server does not supply a fixed Collector address.

```bash
pip install 'lmdeploy[serve]'

export OTEL_SERVICE_NAME=lmdeploy-full
: "${OTEL_EXPORTER_OTLP_ENDPOINT:?Set the Collector endpoint in the deployment environment}"
export OTEL_EXPORTER_OTLP_ENDPOINT
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=prod,k8s.cluster.name=cluster,k8s.namespace.name=ns,k8s.pod.name=pod,k8s.node.name=node"
export OTEL_BSP_MAX_QUEUE_SIZE=2048
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
export OTEL_BSP_SCHEDULE_DELAY=5000
```

Tracing is disabled unless an OTLP endpoint is configured before startup.
`OTEL_SERVICE_NAME` is configurable and defaults to `lmdeploy-full`; register the configured name
in the platform's exact service allowlist before deployment. The engine and role remain
`lmdeploy` and `full`, regardless of resource overrides. A service name does not enable P/D tracing.
OTLP/gRPC is the default; `http/protobuf` supports a Collector HTTP endpoint on port 4318.
Signal-specific `OTEL_EXPORTER_OTLP_TRACES_*` settings take precedence. The HTTP generic endpoint
gets `/v1/traces` appended by the exporter; a signal-specific endpoint must include the path.
The SDK handles TLS, certificates, export timeout and transport retries using its standard settings.
Collector persistent buffering is deployment owned.

The three `OTEL_BSP_*` settings above are optional and show the defaults, matching vLLM's use
of the OpenTelemetry SDK: a 5-second scheduled delay, batches of up to 512 completed spans and
a queue of 2,048 spans. A full batch can be exported earlier. Ended spans are pushed
asynchronously; an unfinished request is not exported periodically. The delay affects when
data becomes visible, not the recorded span timestamps or request duration. Collector batching
is separate and may add another delay. Existing deployment overrides still take precedence.
Five seconds is not an upper bound on delivery time or data loss on forced termination;
queued spans, export retries and shutdown behavior also matter.

The private provider uses ParentBased(ALWAYS_ON), a bounded BatchSpanProcessor and asynchronous
export. It respects an upstream unsampled flag; there is no additional random source sampling.
Unsampled HTTP requests skip prompt observers and output inspection while
preserving ordinary response behavior and valid request-ID correlation.
Invalid startup settings fail startup; runtime instrumentation/export failures do not replace an
inference result. No SDK work or per-token span creation is added to the engine output loop.

LMDeploy emits `llm_request` (HTTP SERVER), `inference` (one per prompt), and the INTERNAL
`model.prefill` and `model.decode` phases when their complete boundaries are known.
Choose either HTTP or prompt usage when aggregating tokens. GenAI latency attributes use seconds.
OTLP timestamps are epoch nanoseconds. Store and visible output timings are scalar attributes on
the prompt span.
Input preparation is the scalar `lmdeploy.preprocess_seconds`, with no separate span.
Stage durations are not pure GPU kernel times.

Prompt spans are children of the HTTP request span `llm_request`. Phase spans use the propagated
upstream parent (for example, the Router's `model.router` or Envoy's `envoy.upstream`). Without a valid incoming
parent, they use the local request span. `lmdeploy.prompt_index` links a batch prompt to its
phases. Concurrent choices also carry `lmdeploy.choice_index` on inference and phase spans.
HTTP input/cache usage counts shared input once (choice 0, matching the API). Output usage
sums all choices. Distinct batch prompts still contribute their individual inputs.
Request/response model fields use the resolved API model,
including a selected LoRA adapter; `lmdeploy.base_model` identifies its base model when different.
Library calls without HTTP create one `llm_request` root with phases below it.
Each prompt span ends at its own inference completion, not at the end of the enclosing HTTP batch.
Output observations remain separate attributes; `llm.stream.end` belongs to the HTTP span.
HTTP TTFT starts at HTTP receipt. Prompt TTFT starts at engine queue admission.
The HTTP span also records E2E latency for requests rejected before inference. It records the
HTTP status code once the ASGI server accepts the response headers. A 5xx marks spans carrying
that HTTP code as ERROR. Phase spans and the prompt's `lmdeploy.outcome` describe engine work.
Token completeness is explicit through `lmdeploy.token_counts_complete`; failed/cancelled counts
are partial. Provider initialization is serialized once per process, and the SDK owns shutdown.
Handle waiting and mean TPOT use `lmdeploy.handle_wait_seconds` and `lmdeploy.tpot_seconds`.

For raw SSE fallback observation, a terminal event is accepted only after its blank-line
delimiter is sent. A truncated terminal event does not mark the request successful. Event
headers still identify oversized terminal records without retaining their payloads. Normal
server adapters continue to use payload-free metadata attached to complete output frames.

LMDeploy exports both phase names directly with `inference.engine=lmdeploy`,
`inference.role=full`, a valid inbound `request.id` and optional `session.id` when available.
Read phase durations from their timestamps and failures from their OpenTelemetry status.
Keep the names `llm_request` and `inference` for request and prompt usage, E2E and TTFT.
When the Router forwards `model.router` in W3C `traceparent`, both phases and
`llm_request` are its direct children. Each `inference` is a child of `llm_request`.

Register engine `lmdeploy` and role `full` and retain the platform's error-trace tail-sampling
policy. The gateway must propagate the valid
`x-request-id`, `traceparent` and explicit `x-session-id` and export the referenced upstream span.

The lists below describe the platform integration fields. The companion
`request_trace_contract.json` contains them in machine-readable form. SDK and environment resource
attributes are restricted to the list below, with scalar, size and credential checks. Other fields
are omitted before export, including raw affinity and request/session IDs injected through
`OTEL_RESOURCE_ATTRIBUTES`. Request/session IDs come only from the accepted request headers.
Additional resource fields require an agreed platform contract update before being allowed here.

Resource attributes:

```text
service.name
service.instance.id
deployment.environment
k8s.cluster.name
k8s.namespace.name
k8s.pod.name
k8s.node.name
inference.engine
inference.role
```

Span attributes:

```text
inference.engine
inference.role
request.id
session.id
trace.context.missing
http.request.method
url.path
http.route
http.response.status_code
gen_ai.provider.name
gen_ai.operation.name
gen_ai.request.model
gen_ai.response.model
gen_ai.usage.input_tokens
gen_ai.usage.output_tokens
gen_ai.usage.cached_tokens
gen_ai.usage.cache_read.input_tokens
gen_ai.latency.time_to_first_token
gen_ai.latency.e2e
lmdeploy.prompt_index
lmdeploy.choice_index
lmdeploy.base_model
lmdeploy.backend
lmdeploy.finish_reason
lmdeploy.queue_seconds
lmdeploy.store_lookup_seconds
lmdeploy.store_matched_tokens
lmdeploy.kv_load_seconds
lmdeploy.kv_loaded_tokens
lmdeploy.kv_load_status
lmdeploy.outcome
lmdeploy.http_total_seconds
lmdeploy.first_output_seconds
lmdeploy.reasoning_first_seconds
lmdeploy.reasoning_last_seconds
lmdeploy.content_first_seconds
lmdeploy.reasoning_to_content_seconds
lmdeploy.max_output_gap_seconds
lmdeploy.handle_wait_seconds
lmdeploy.tpot_seconds
lmdeploy.token_counts_complete
lmdeploy.preprocess_seconds
```

`gen_ai.usage.cache_read.input_tokens` is the standard cache-read count. The existing
`gen_ai.usage.cached_tokens` carries the same value for platform compatibility; do not add them.
The platform accepts scalar attributes only, so finish reasons remain `lmdeploy.finish_reason`
rather than introducing the array-valued `gen_ai.response.finish_reasons`.
`gen_ai.usage.input_tokens` includes cached input. The cache-read count is a subset, not an
additional input count. The compatibility cache alias and `gen_ai.latency.*` fields are part of
the platform contract, not additional standard GenAI attributes. Token counts describe usage;
they do not calculate monetary cost or GPU work. The Prefill summary below distinguishes cache
sources, computation and recomputation where the PyTorch engine exposes them.

### Request-level TTFT diagnostics

With tracing enabled, PyTorch AR and ARSpec (including MTP) export this fixed-size summary on
the corresponding `model.prefill` span. There is one collection mode, with no additional switch or
per-chunk spans, events, arrays or token text. The summary is frozen when the engine accepts
the final prefill output, then travels only with the terminal engine output, including
nonstreaming and coalesced output. It is not copied on each decode token. It does not change
sampling, chunking, cache policy or output behavior.
If complete Prefill timing boundaries are unavailable, no `model.prefill` span or diagnostic
summary is exported; the values are not moved to `inference` as a fallback.
Host accounting follows the OTLP endpoint setting; it has no separate switch.
If decode prefetch preempts a prefill whose output is then discarded, its executed work still
counts. Accounting continues through rescheduling and recomputation until a prefill output is
accepted; a completed forward alone does not freeze the summary.

All names below have the prefix `lmdeploy.prefill.`:

| Attribute                                                  | Meaning and TTFT use                                                                                                                                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunk_count`                                              | Number of completed target-model prefill forwards before the first output; a normal unsplit prefill counts as one.                                                         |
| `computed_tokens`                                          | Total target-model token positions executed by these forwards, including retained history and replay work.                                                                 |
| `computed_input_tokens`                                    | Executed positions within the current input; repeated execution is included.                                                                                               |
| `recomputed_tokens`                                        | Executed positions below a previously completed prefill boundary, including positions previously skipped by cache reuse. Reveals replay work.                              |
| `min_chunk_tokens`, `max_chunk_tokens`                     | Smallest and largest executed chunks; distinguish small scheduling slices from large forwards.                                                                             |
| `local_cached_tokens`, `external_cached_tokens`            | Accepted reuse within the current input at its initial dispatch, split between local cache and successfully loaded external KV. Lookup hits and failed loads do not count. |
| `inter_chunk_wait_seconds`, `max_inter_chunk_wait_seconds` | Total and longest host-observed gap from completion of one chunk to dispatch of the next when no chunk of this request is in flight.                                       |
| `preemption_count`, `reschedule_count`                     | Preemption episodes and subsequent actual dispatches before the first output. Deactivation and reclamation in one episode are counted once.                                |

Cache counts exclude previous-turn history and tentative matches rolled back before dispatch.
They describe initial reuse; they are not cumulative cache-lookup counts. Replay may make
`computed_input_tokens + local_cached_tokens + external_cached_tokens` exceed the input size.
The coordinator publishes one accepted external interval after aggregating all TP ranks;
duplicate progress and failed loads do not add intervals. Loads after initial dispatch are
outside this initial-reuse summary and remain observable through the Store/load metrics.
The summary belongs to one input turn, so resumed sessions and batched prompts do not share it.

The waiting values are a **lower bound on scheduling gaps**, measured with the host monotonic
clock. A prefetched chunk is already in flight and contributes no host gap, even if it later
waits in the executor. Completion notification may also arrive after GPU completion. Do not
interpret the remaining prefill interval as GPU compute time, or add Store lookup/load time
to these intervals: those operations can overlap existing queue/prefill timing.

Use input size, cache reuse and computed tokens to identify extra work; use chunk sizes,
host gaps and preemption/replay counters to identify scheduling pressure. These summaries do
not identify a slow GPU kernel. Existing initial queue, preprocessing, handle wait and Store
lookup/load measurements remain available and retain their timing boundaries.

DLLM and TurboMind do not provide this additional summary; their existing tracing is unchanged.
Unsupported paths, missing native metrics
and requests without a terminal native output leave the summary absent, rather than reporting zeros.
An accounting error invalidates only affected inputs and clears their pending accounting;
later inputs continue normally. Failed/cancelled submission and output waits also invalidate
affected summaries. A successful executor completion counts even without generated tokens;
a missing completion does not establish executed work. Diagnostics report the operation,
exception type and a process-local omitted-summary count, with warnings at powers of two.
Exception text and request data are not logged. No GPU synchronization is introduced.
Platform attribute allowlists must include the new scalar keys; span-name allowlists and
parent propagation do not change.

Span events:

```text
request.received
llm.first_token
llm.stream.end
error
```

The API server enables the observer when an OTLP endpoint is configured. Library
`pipeline()`/`AsyncEngine` users additionally pass `enable_request_trace=True`.
`PreprocessedRequest.trace` is explicitly typed and owned by the existing request, with no side table.
Tracing does not write local request files.

## Correlation and outcomes

For repeated correlation headers, the first occurrence is used, including when it fails validation.
`request_id` is the validated incoming `x-request-id`, never generated or rewritten by the engine.
Missing/invalid IDs are omitted from spans. The gateway contract requires
16–64 ASCII letters, digits, underscores or hyphens. Invalid values increment the process-local
`otel_trace.invalid_request_id_count()` and emit one warning per process without the original value.
Missing headers are not counted as invalid. No fallback business request ID is generated.
An existing response `x-request-id` is preserved; a validated inbound ID is added only if that
response header is absent. CORS exposes the header. Request/response bodies are unchanged.
All prompts in one HTTP request share the gateway request ID and trace ID. Each prompt has a
separate span ID, a zero-based `lmdeploy.prompt_index`, and the HTTP span as its parent.
Rejected requests retain an HTTP span without an invented prompt.
The completions endpoint supplies prompt positions explicitly when preprocessing a batch.
Chat fan-out passes explicit choice indices through preprocessing metadata. The indices distinguish
separate engine executions of a shared prompt without adding business request IDs.
Library calls without HTTP context do not invent a gateway request ID.

A valid W3C `traceparent` is inherited; a missing/invalid context creates one HTTP root marked
`trace.context.missing=true`. Prompt and phase spans are descendants, not extra roots.
Optional `tracestate` is not retained or exported; unvalidated vendor values cannot bypass the
metadata privacy filter through OTLP trace state.
Session selection first uses a nonempty `x-session-id`. When both it and `x-session-affinity`
are empty or absent, a safe `x-openclaw-session-key` can supply `session.id`. With a nonempty
affinity header and no explicit session ID, the engine omits `session.id`. The platform requires
affinity processing at the gateway and forbids engines from exporting or hashing its raw value.
The gateway
must apply the documented priority (explicit ID, affinity normalization, OpenClaw) and inject the
approved result as `x-session-id`. This gateway step still requires integration acceptance.
Affinity headers remain untouched. Exceptions, bodies and credentials are not attributes.
Explicit session IDs retain punctuation and spaces without rewriting. Nonempty printable values
up to 1024 characters are accepted, matching the metadata string-size bound; control characters,
oversized values and credential-like values are omitted. The request ID format is unchanged.
Focused SDK tests cover these session ID boundaries and actual gRPC/HTTP protobuf export.

HTTP, prompt and completed phase spans carry `lmdeploy.outcome`: `success`, `error` or `cancelled`.
Phase outcomes reflect the engine result; they are not separate per-phase exception diagnoses.
HTTP status is recorded only when headers were accepted, so an SSE stream can retain HTTP 200
while its span records an error or cancellation. Failed/cancelled requests keep available partial
usage with `lmdeploy.token_counts_complete=false`; unknown phase boundaries are omitted.

The generation wrapper returns an async generator and retains the original callable through
`__wrapped__`; introspection should use `inspect.unwrap`. Subclasses overriding preprocessing or
generation must preserve the tracing wrapper or call the base implementation.
All tracing enable flags are process-start snapshots; environment changes require a restart.

Output token counts use the same cumulative `GenOut` counters consumed by the API adapters,
after stop-token filtering and the configured stop-token inclusion policy. No text is retokenized.
Native token counts are kept separately for first-token detection and TPOT; a hidden stop token
can affect native timing without contributing to API output usage. Missing API output counts
remain unknown, not zero. Failed/cancelled results report available partial API counters and mark
`token_counts_complete=false`.
Input usage includes cached tokens. OpenAI input usage is directly comparable; Anthropic Messages
reports uncached `input_tokens` and separate `cache_read_input_tokens`, whose sum is comparable.

These fields help locate long prompts, KV waits and visible-output stalls without claiming a
network or GPU root cause. Visible-output gaps are not per-token ITL, and TPOT is an average.

## Timing boundaries and unknown values

Prefill/Decode include scheduling, preemption and host processing; they are not GPU kernel times.
KV lookup/load may overlap queue time, so stage durations must not be added together. API output
observations are accepted server sends, not client network receipt times.

PyTorch carries the earliest generated-token timestamp on the request before output coalescing;
later batches retain it without additional engine output. A multi-token batch without an explicit
first-token timestamp supplies only a last-token observation, leaving Prefill/Decode, TTFT and TPOT
unknown. A backend that already provides incremental
outputs retains the available boundaries even when the HTTP response is aggregated. Scoring-only
requests do not invent a generated token. A known single-token Decode interval is zero.
If the newest token batch lacks timing, last-token and Decode/TPOT remain unknown rather than
reusing an earlier timestamp.

Local elapsed times use monotonic clocks. HTTP and prompt span ends use their starting epoch plus
monotonic elapsed time, so wall-clock adjustments during execution cannot reverse their durations.
Prompt starts share their HTTP request's clock anchor, keeping parent/child ordering consistent.
Native events and token times keep the existing wall clock for PyTorch/TurboMind interoperability. Invalid, missing or regressing intervals
are omitted from spans.
Cross-host comparisons and native-to-frontend delay require synchronized clocks. TurboMind token
time is a Python callback observation, while PyTorch supplies model-agent timing. Buffered output
can limit boundary precision; missing boundaries cannot diagnose an unobserved interval.

Visible stages are observed from Chat/Completions, Responses, Messages and native Generate SSE.
Nonstream JSON and hidden reasoning do not infer visible stage times. The fallback parser bounds SSE
lines to 1 MiB; oversized or malformed lines can leave stages unknown without changing the response
body. Responses terminal event headers preserve completion for oversized terminal payloads
only after a data line and the terminating blank line. Raw SSE fields must start at the line
boundary; an event without data or without its final blank line is not a completed event,
even at clean EOF. This matches the API Server adapters' framing.
Annotated frames are observed without this parser limit.
Records accepted in one ASGI body share its observation timestamp; parsing time is not counted as
a visible output gap. Frames use JSON semantics, including escaped keys; words inside generated
text are never classified as events.
UTF-8 decoding is explicit to avoid repeated encoding detection. Responses function-call starts
count as visible output. Engine error and
cancellation outcomes are shared with the HTTP summary, including nonstream responses with HTTP 200.

## Mooncake Store

Lookup timing starts at speculative prefetch submission when present, or at the first direct
lookup otherwise. Prefetch and admission polling share the same start marker.
The scheduler records result delivery, and the paging coordinator records
load admission through terminal progress aggregated across TP ranks. Repeated polling does not
create repeated start records. A fixed-size event cursor also deduplicates PyTorch cumulative
event snapshots, so repeated outputs cannot multiply lookup/load times or loaded-token counts.
Nonempty histories that shrink or change their processed prefix are marked untrusted, not treated
as incremental batches. Untrusted event metrics remain null for the rest of the request; token
counts and independent token timing continue. Overlapping loads receive the same treatment.
Query token matches and successfully published load tokens are
separate; a positive lookup does not establish successful cache reuse.

The native lookup client can fall back to zero on errors. `store_matched_tokens=0` therefore means
no reusable match was returned, not proof of a cache miss. Consult native Store logs for the cause.
User stop/end during a load is recorded as cancelled rather than failed. Lookup cancellation emits
a closing boundary and retains its observed duration without claiming a cache miss. Cancellation
events still depend on normal native output delivery and may be unavailable after engine loss.
A failed load keeps the existing rollback/recompute behavior and is marked failed even if later
work succeeds. Loaded-token totals count actual published ranges and may include repeated ranges.

Existing asynchronous saves remain unchanged and do not delay request completion. There is no
request-level save completion or save latency field, since background work can outlive generation.
No checksum, payload copy, forced transfer or extra Store RPC is introduced.

KV events travel with ordinary engine outputs. An engine that dies before returning output can
lose pending events; missing load results are unknown. Request cancellation, lookup cancellation,
request cleanup and worker-drain behavior retain their native ownership rules.

## Lightweight observation and failure behavior

Each request maintains bounded timing and counters. Only completed spans enter the SDK's bounded
queue for asynchronous OTLP export. No trace-specific local files, disk writer thread or local-log
configuration are used. Native request logging remains a separate feature.

Instrumentation exceptions are isolated from generation, preprocessing and ASGI output observation.
Malformed native diagnostic metadata invalidates that request's affected timings without changing
its API output or token usage. Failures in span creation/finalization or output observation do not
replace the application's original exception or cancellation. Context is cleared when each request
exits; a failed request or observation does not disable tracing for later requests. Runtime faults
produce a bounded warning without exception text or request content. Invalid explicit startup
configuration still fails validation before serving.

Request success, failure and cancellation are recorded when observation is available. Exporter
outages, queue overflow, process termination and instrumentation faults can lose spans; this is
best-effort telemetry, not a durable audit trail. Exporting waits outside the request path. Prompt
text, output text, reasoning text, token arrays, hashes and KV contents are not exported.

CPU tests run with:

```bash
python -m unittest discover -s tests/observability -v
```

They exercise actual production methods with simulated backends and no GPU dependencies. Real
model execution, Mooncake Store end-to-end behavior and timing overhead require machine testing.

Real import and browser CORS smoke checks are also provided:

```bash
pytest tests/test_lmdeploy/serve/test_request_trace_runtime.py
```

The import smoke skips when torch is unavailable and otherwise imports the real package graph.

## Machine validation

### PyTorch with Router and Mooncake Store

The following GPU measurements predate the request-level Prefill diagnostics and OTLP-only
cleanup. They are a historical baseline, not an overhead measurement for the current code.

GLM-5.3-Flash (revision `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`) ran on one
eight-H200 node with PyTorch DP8/EP8, MTP5 and Mooncake Store. Settings included batch size
32, context length 524,288, prefill chunk limit 8,192, cache fraction 0.7 and prefix caching.
The runtime used Python 3.12.3, PyTorch 2.12.1/CUDA 13.0, Transformers 5.17.0 and
OpenTelemetry SDK/exporters 1.45.2, checked against the declared serving dependencies.

AgentX used 70 concurrent trajectories from `semianalysis_cc_traces_weka_062126_256k`,
streaming requests, a 256,000-token workload context limit and dataset seed 20260910.
This seed fixes workload selection, not model generation. Each mode had a warmup and
900-second measurement, followed by a 30-second drain and 10-second cancellation settlement.
The reported throughput uses the approximately 940-second window. The same separate CPU
client ran both modes. Model and client startup were outside the measurement window.

The comparison covers tracing disabled in both Router and LMDeploy versus their full
sampling and asynchronous OTLP export through a local Collector to the gateway. Collector file export is also disabled during measurement.

| Metric (completed requests) |  Trace off | Full sampling | Change |
| --------------------------- | ---------: | ------------: | -----: |
| Successful requests         |      1,064 |         1,064 |  0.00% |
| Mean TTFT (ms)              |  1,373.116 |     1,325.361 | -3.48% |
| Median TTFT (ms)            |    903.918 |       911.211 | +0.81% |
| p99 TTFT (ms)               | 10,339.701 |     9,642.086 | -6.75% |
| Mean TPOT/ITL (ms/token)    |     24.549 |        25.125 | +2.35% |
| Output tokens/s             |    886.763 |       864.199 | -2.54% |
| Requests/s                  |      1.132 |         1.132 |  0.00% |
| Goodput (requests/s)        |      0.790 |         0.749 | -5.25% |
| Mean input tokens           | 93,967.212 |    93,855.014 | -0.12% |
| Mean output tokens          |    783.420 |       763.485 | -2.54% |

Goodput requires TTFT at most 3,000 ms and per-user output throughput at least 40 tokens/s.
Latency and token statistics describe completed requests; they do not include unfinished
requests. Matching 1,059 requests by trajectory and turn gives identical mean output length,
with mean TTFT changing by -3.91% and mean ITL by +2.04%. These are sequential runs, not a
universal overhead bound: generation, cache state and scheduling can differ. Lower TTFT does
not establish a tracing speedup, and the throughput change accompanies different output lengths.

The Collector accepted and exported 5,712 spans during the full-sampling run, with zero
refused or failed exports and an empty queue at completion. The off run exported no new spans.
These counters establish OTLP transport delivery, not final gateway filtering or query acceptance.
The deployment's five business span names, parent relationships and timing boundaries are
unchanged by the streaming metadata optimization.
