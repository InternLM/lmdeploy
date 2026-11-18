# OpenTelemetry request tracing and optional JSONL logs

Request phase timing follows vLLM request-level metrics: queue, Prefill to first token, and Decode
from first to last token. Completed phase spans also expose these intervals in the platform
waterfall. The first release covers ordinary generation and the native PyTorch Mooncake Store
connector. P/D and Proxy tracing are outside this release. TurboMind supports
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
   HTTP completion supplies transport and visible-output summaries. Phase spans are direct
   children of the HTTP/router span, associated with their prompt through `lmdeploy.prompt_index`.
4. OpenTelemetry's bounded background processor exports OTLP. Independently, an optional bounded
   writer queue serializes JSONL summaries to disk. Neither output requires per-token spans,
   synchronous request-path disk/network I/O, extra Store RPCs, or internal streaming for nonstream
   requests. Unknown timing boundaries remain unknown.

| Source entry point                                                                                            | Responsibility                                                                               |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `lmdeploy/trace.py`, `lmdeploy/serve/openai/api_server.py`                                                    | Read startup enable flags and connect tracing to the API server                              |
| `lmdeploy/serve/core/async_engine.py`, `lmdeploy/serve/openai/endpoints/completions.py`                       | Attach per-prompt state, assign batch positions and observe the existing generation path     |
| `lmdeploy/serve/core/chat_runner.py`, `lmdeploy/serve/openai/chat_completions/`                               | Pass the resolved model and explicit choice identity through existing preprocessing metadata |
| `lmdeploy/serve/observability/request_trace.py`                                                               | HTTP middleware, timing aggregation, bounded SSE observation and asynchronous JSONL writing  |
| `lmdeploy/serve/observability/otel_trace.py`                                                                  | SDK initialization, parent propagation, metadata filtering, spans and OTLP export            |
| `lmdeploy/messages.py`, `lmdeploy/pytorch/messages.py`                                                        | Carry optional Store event boundaries and counts with ordinary engine output                 |
| `lmdeploy/pytorch/kv_connector/mooncake/store/scheduler.py`, `lmdeploy/pytorch/paging/kv_load_coordinator.py` | Observe existing Store lookup and load lifecycle                                             |
| `setup.py`, `tests/observability/`, `tests/test_lmdeploy/serve/test_request_trace_*.py`                       | Optional trace dependencies, behavioral/transport regression tests and real-package checks   |

## Export traces to a local Collector

Install `lmdeploy[trace]` and set the endpoint before starting the API server:

```bash
export OTEL_SERVICE_NAME=lmdeploy-full
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.namespace.svc:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=prod,k8s.cluster.name=cluster,k8s.namespace.name=ns,k8s.pod.name=pod,k8s.node.name=node"
export OTEL_BSP_MAX_QUEUE_SIZE=2048
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
export OTEL_BSP_SCHEDULE_DELAY=1000
```

The endpoint enables tracing independently of JSONL. Both outputs can be enabled together.
`OTEL_SERVICE_NAME` is configurable and defaults to `lmdeploy-full`; register the configured name
in the platform's exact service allowlist before deployment. The engine and role remain
`lmdeploy` and `full`, regardless of resource overrides. A service name does not enable P/D tracing.
OTLP/gRPC is the default; `http/protobuf` supports a Collector HTTP endpoint on port 4318.
Signal-specific `OTEL_EXPORTER_OTLP_TRACES_*` settings take precedence. The HTTP generic endpoint
gets `/v1/traces` appended by the exporter; a signal-specific endpoint must include the path.
The SDK handles TLS, certificates, export timeout and transport retries using its standard settings.
Never send JSONL to the trace pipeline. Collector persistent buffering is deployment owned.

The private provider uses ParentBased(ALWAYS_ON), a bounded BatchSpanProcessor and asynchronous
export. It respects an upstream unsampled flag; there is no additional random source sampling.
Invalid startup settings fail startup; runtime instrumentation/export failures do not replace an
inference result. No SDK work or per-token span creation is added to the engine output loop.

Native spans are `llm_request` (HTTP SERVER), `inference` (one per prompt), and `prefill`, `decode`
phases when their complete boundaries are known. HTTP and prompt usage overlap: choose
one level when aggregating tokens. GenAI latency attributes are seconds; OTLP timestamps are epoch
nanoseconds. Store and visible output timings are scalar attributes on the prompt span.
Input preparation is the scalar `lmdeploy.preprocess_seconds`, with no separate span.
Stage durations are not pure GPU kernel times.

All prompt and phase spans are children of the HTTP/router `llm_request`, as required by the
platform parent contract. `lmdeploy.prompt_index` associates a batch prompt with its phases.
Concurrent choices of the same prompt additionally carry `lmdeploy.choice_index` on inference
and phase spans. HTTP input/cache usage counts the shared input once (choice 0, matching the API);
output usage sums all choices. Distinct batch prompts still contribute their individual inputs.
Request/response model fields use the resolved API model,
including a selected LoRA adapter; `lmdeploy.base_model` identifies its base model when different.
Library calls without HTTP create one `llm_request` root with phases below it.
Each prompt span ends at its own inference completion, not at the end of the enclosing HTTP batch.
Output observations remain separate attributes; `llm.stream.end` belongs to the HTTP span.
HTTP TTFT starts at HTTP receipt; prompt TTFT starts at engine queue admission.
Token completeness is explicit through `lmdeploy.token_counts_complete`; failed/cancelled counts
are partial. Provider initialization is serialized once per process, and the SDK owns shutdown.
Handle waiting and mean TPOT use `lmdeploy.handle_wait_seconds` and `lmdeploy.tpot_seconds`.

The platform must register the configured service (default `lmdeploy-full`) with role `full`,
map `llm_request` to `model.router`,
`inference` to the new `model.inference`, and map phase names to their corresponding `model.*`
names. The `lmdeploy.*` metadata and timing attributes require explicit platform allowlisting.
Existing vLLM/SGLang-only gateway contracts will discard these spans until extended.
This implementation does not deploy or modify a Collector or Gateway.

Platform normalization must run before filtering. Register engine `lmdeploy` and role `full`,
copy `inference.engine` and `inference.role` from resources into query attributes where required,
and retain the platform's error-trace tail-sampling policy. The gateway must propagate the valid
`x-request-id`, `traceparent` and explicit `x-session-id` and export the referenced upstream span.

The complete allowlists below make the integration contract explicit. The companion
`request_trace_contract.json` contains the same lists in machine-readable form.

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

Span events:

```text
request.received
llm.first_token
llm.stream.end
error
```

## Enable logging

JSONL logging is disabled by default, including when OTLP tracing is enabled.
Only set `LMDEPLOY_TRACE_ENABLED=1` when local files are required. Before starting the API server, set:

```bash
export LMDEPLOY_TRACE_ENABLED=1
export LMDEPLOY_TRACE_LOG_DIR=/absolute/writable/logs/instance
# Optional: 256 MiB by default, with additional UTC date rotation.
export LMDEPLOY_TRACE_LOG_MAX_MB=256
```

The directory is expanded with `expanduser`, validated as writable and absolute, and created if
missing. Flags are cached at process startup; settings take effect after restart.

Logging preserves generation settings and streaming mode. Unavailable engine timings stay null;
nonstream backend execution is never switched to internal streaming for logging.

JSONL files are written to:

```text
<root>/lmdeploy/<hostname>/<YYYYMMDDTHHMMSS.ffffffZ>.<PID>.jsonl
```

The observer handles ordinary complete SSE frames directly, with a shared bounded parser for
fragmented/coalesced/CRLF frames. It decodes JSON semantically without scanning for each possible
field name first. Each accepted ASGI body is parsed once; per-choice semantic flags are routed by
explicit wire indices, even when one frame contains multiple choices. Unknown or ambiguous owners
only contribute to HTTP observations, never to a guessed prompt. No shared last-yield state is used.
Per-request engine observers are bound once before generation; token counts and
status are read once per output. None of these paths retain generated text.

Files are created exclusively; collisions use numeric suffixes. Rotation follows UTC dates and
configured size. Old files remain until manually removed. JSONL bypasses native logging levels.
Native stdout/stderr and its redirection remain deployment owned. Rotation does not bound total
disk use. For example, assuming 2 KiB total JSONL per request at 2,000 requests/s gives about 354 GB
per day before filesystem overhead. This is a sizing example, not measured log volume; provision
capacity and an external cleanup policy according to actual traffic.

The API server opts in when enabled. Library `pipeline()`/`AsyncEngine` users additionally pass
`enable_request_trace=True`; setting the environment alone does not create their writer thread.
`PreprocessedRequest.trace` is explicitly typed and owned by the existing request, with no side table.

## Correlation and records

For repeated correlation headers, the first occurrence is used, including when it fails validation.
`request_id` is the validated incoming `x-request-id`, never generated or rewritten by the engine.
Missing/invalid IDs remain null in JSONL and are omitted from spans. The gateway contract requires
16–64 ASCII letters, digits, underscores or hyphens. Invalid values increment the process-local
`otel_trace.invalid_request_id_count()` and emit one warning per process without the original value.
Missing headers are not counted as invalid. No fallback business request ID is generated.
An existing response `x-request-id` is preserved; a validated inbound ID is added only if that
response header is absent. CORS exposes the header. Request/response bodies are unchanged.
All prompts in one HTTP request share the gateway request ID. Each prompt has a distinct span ID
and a zero-based `prompt_index`. Engine/output JSONL records share that prompt's `span_id` and
its `trace_id`. In local-only mode a valid platform v00 `traceparent` supplies the trace ID and
upstream parent; missing/invalid context generates a local trace ID. Each HTTP request gets a new
HTTP span ID. All prompt records inherit the trace ID, use distinct span IDs, and reference the
HTTP span through `parent_span_id`. Rejected requests still log their HTTP trace/span IDs and any
valid upstream parent. This path needs no OpenTelemetry dependency, does not overwrite the gateway
request ID, and does not create exported OTel spans.
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

`engine_request_finished` and `request_finished` remain per-prompt summaries, including cancelled
prepared prompts. Rejected HTTP requests have an HTTP summary without an invented prompt.

The generation wrapper returns an async generator and retains the original callable through
`__wrapped__`; introspection should use `inspect.unwrap`. Subclasses overriding preprocessing or
generation must preserve the tracing wrapper or call the base implementation.
All tracing enable flags are process-start snapshots; environment changes require a restart.

| Fields                                                                                                       | Purpose and interpretation                                                                                                                |
| ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `timestamp`, `hostname`, `pid`, `backend`                                                                    | Locate the instance and corresponding native logs                                                                                         |
| `request_id`, `prompt_index`, `choice_index`, `event`, `component`, `path`                                   | Associate an engine execution's records and locate its batch prompt and chat choice                                                       |
| `model`, `base_model`                                                                                        | Identify the resolved API model/LoRA and the underlying model                                                                             |
| `status`, `http_status`, `disconnected`, `finish_reason`                                                     | Distinguish success, errors, cancellation and length limits; HTTP 200 alone is insufficient                                               |
| `input_tokens`, `output_tokens`, `cached_tokens`, `max_output_tokens`                                        | Distinguish long prompts, long generation, cache reuse and configured output limits                                                       |
| `engine_events_trusted`                                                                                      | False if native event history shrinks, resets or contains overlapping loads; affected event metrics become null instead of being replayed |
| `token_counts_complete`                                                                                      | True only with API counters and successful terminal engine output; failed/cancelled counts are partial                                    |
| `preprocess_seconds`                                                                                         | Preprocess entry through rendering, tokenization, multimodal processing and validation                                                    |
| `handle_wait_seconds`                                                                                        | Generate entry through acquisition of an engine request handle                                                                            |
| `queue_seconds`                                                                                              | Native QUEUED through first SCHEDULED                                                                                                     |
| `prefill_seconds`                                                                                            | First SCHEDULED through first observed native token boundary                                                                              |
| `decode_seconds`                                                                                             | First through last native token boundary                                                                                                  |
| `engine_ttft_seconds`                                                                                        | Native QUEUED through first token                                                                                                         |
| `tpot_seconds`                                                                                               | Decode interval divided by native generated token count minus one (including stop tokens); null without usable boundaries                 |
| `store_lookup_seconds`                                                                                       | Sum of observed lookup submission-to-result intervals, including asynchronous polling waits                                               |
| `store_matched_tokens`                                                                                       | Maximum newly reusable token count returned by observed Store lookups                                                                     |
| `kv_load_seconds`, `kv_loaded_tokens`, `kv_load_status`                                                      | Observed external-load duration, cumulative successfully published token ranges, and pending/success/failed/cancelled outcome             |
| `token_delivery_delay_seconds`                                                                               | First native token timestamp through its receipt by frontend generation code                                                              |
| `max_engine_output_gap_seconds`                                                                              | Largest monotonic interval between frontend receipts of nonempty engine output batches                                                    |
| `engine_timestamps`                                                                                          | Native wall-clock queue/schedule/first-token/last-token times for correlation with delivery and native logs                               |
| `http_received_unix_seconds`                                                                                 | HTTP entry wall clock; shared by prompts within a batch                                                                                   |
| `total_seconds`                                                                                              | This prompt's preprocessing entry through generation exit                                                                                 |
| `http_first_body_seconds`, `http_total_seconds`                                                              | HTTP entry through first accepted body / response exit; shared transport measurements for batch prompts                                   |
| `first_output_seconds`                                                                                       | This prompt's preprocessing entry through first visible reasoning/content/tool output                                                     |
| `max_output_gap_seconds`                                                                                     | Largest gap between visible SSE observations; metadata and terminal markers are excluded                                                  |
| `reasoning_first_seconds`, `reasoning_last_seconds`, `content_first_seconds`, `reasoning_to_content_seconds` | Locate visible reasoning and content stages; null when unobserved                                                                         |

Output token counts use the same cumulative `GenOut` counters consumed by the API adapters,
after stop-token filtering and the configured stop-token inclusion policy. No text is retokenized.
Native token counts are kept separately for first-token detection and TPOT; a hidden stop token
can affect native timing without contributing to API output usage. Missing API output counts
remain unknown, not zero. Failed/cancelled results report available partial API counters and mark
`token_counts_complete=false`.
Input usage includes cached tokens. OpenAI input usage is directly comparable; Anthropic Messages
reports uncached `input_tokens` and separate `cache_read_input_tokens`, whose sum is comparable.

These fields address observed slow-request patterns: long prompts, KV waits, engine-to-frontend
first-output gaps, and stalls after generation starts. They locate intervals without claiming a
network or GPU root cause. Output-batch gaps are not per-token ITL, and TPOT is an average.

## Timing boundaries and unknown values

Prefill/Decode include scheduling, preemption and host processing; they are not GPU kernel times.
KV lookup/load may overlap queue time, so stage durations must not be added together. API output
observations are accepted server sends, not client network receipt times.

A final-only batch containing several tokens supplies a last-token observation but no first-token
boundary. Its Prefill/Decode, TTFT and TPOT are null. A backend that already provides incremental
outputs retains the available boundaries even when the HTTP response is aggregated. Scoring-only
requests do not invent a generated token. A known single-token Decode interval is zero.
If the newest token batch lacks timing, last-token and Decode/TPOT remain unknown rather than
reusing an earlier timestamp.

Local elapsed times use monotonic clocks. HTTP and prompt span ends use their starting epoch plus
monotonic elapsed time, so wall-clock adjustments during execution cannot reverse their durations.
Prompt starts share their HTTP request's clock anchor, keeping parent/child ordering consistent.
Prompt JSONL and span E2E values share the same measurement. Native events and token times keep the
existing wall clock for PyTorch/TurboMind interoperability. Invalid, missing or regressing intervals
are null.
Cross-host comparisons and native-to-frontend delay require synchronized clocks. TurboMind token
time is a Python callback observation, while PyTorch supplies model-agent timing. Buffered output
can limit boundary precision; missing boundaries cannot diagnose an unobserved interval.

Visible stages are observed from Chat/Completions, Responses, Messages and native Generate SSE.
Nonstream JSON and hidden reasoning do not infer visible stage times. SSE lines are bounded to
1 MiB; oversized or malformed lines can leave stages unknown without changing the response body.
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

## Lightweight writing and failure behavior

Each request maintains fixed-size timing and counters. End summaries enter an 8192-record bounded
queue; a background thread serializes and writes up to 64 queued records per flush. Overflow and
I/O failures are counted and reported from the writer with a 60-second warning limit, keeping disk
I/O and fault-reporting logs off the request's submit path. Runtime `emit` errors are counted and
suppressed, preserving successful responses and original application exceptions. Configuration
errors still fail explicit startup validation. Write/open/flush failures trigger exponential retry
backoff from 1 to 60 seconds; queued records are dropped during backoff rather than creating one
file per record. Shutdown attempts a five-second drain.

There is no per-token disk record, per-record fsync, log retention service, query utility, OTLP
collector or alert webhook. Prompt text, generated text, reasoning text, token arrays, hashes and
KV contents are not retained in JSONL. Native request logging remains a separate feature.
Queue overflow, storage faults, blocked disks and forced termination can lose records. This is a
request-diagnosis log, not an audit trail or guaranteed durable delivery mechanism.

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

The API-usage and resource-bound corrections passed 619 current CPU/interface checks and
56 subtests. SDK 1.27 ran 113 tests (three dependency skips); the full runtime exercised the
real-package tests and both protobuf transports. These corrections have not been remeasured
with GPU inference. The machine measurements below are a prior baseline, not a measurement
of the current correction's performance overhead.

Validation passed 2,560 CPU checks, 212 existing Store scheduling/loading checks and
three TurboMind native grammar checks (plus 41 subtests). The trace suite also ran
with the minimum supported OpenTelemetry SDK/exporters 1.27.0: 105 passed and three
runtime-dependent tests skipped in that isolated environment. Current SDK/exporters
1.45.2 were tested with the full runtime. Real gRPC and HTTP/protobuf export ran on both
SDK versions. The unit-test workflow installs the optional trace dependencies.

Both backends were tested with Qwen3-Coder-480B-A35B-Instruct-FP8 on four H200 GPUs,
TP4/EP1, Python 3.12, PyTorch 2.12.1/CUDA 13.0 and Transformers 5.17.0. Runtime and
build dependencies were checked against the official declarations. TurboMind native
libraries were built from the same source; PyTorch used eager mode and Ray. Both used
max batch size 64, session length 32,768, prefill chunk limit 2,048, cache fraction 0.5
and disabled local prefix caching. The API used `reasoning_parser=default` and abort handling.
Only this 480B checkpoint was used.

The four modes (off, JSONL, OTLP, both) passed 120 functional cases and 48 additional
Responses/Anthropic/Generate protocol cases, including streaming and nonstreaming.
Checks cover concurrent choices, batch prompts, usage/model metadata, output ownership,
parent propagation, session priorities, sampling, rejection and cancellation. Deterministic
output hashes matched across modes. Each backend also passed a 96-request burst above
the engine batch limit and exposed measurable request waiting.

### Steady-state performance

Thirty rounds completed 18,241 measured requests without errors. Normal-load request
IDs were reconciled against JSONL records and exported request/prompt spans with no missing
records. Mode order was off/OTLP/JSONL/both/off for each workload. Each mode had a warmup
batch; throughput includes the measurement window and draining its outstanding requests.
The off reference pools the two bracketing rounds. The same loaded engine was reused;
off retained idle SDK/writer threads. These are incremental steady-state measurements
with a loopback protobuf receiver, not cold-start or production Collector measurements.

For 128 input/128 output tokens, streaming, concurrency 32, each window was 180 seconds:

| Backend   | Mode  | Output tokens/s | Mean/p99 TTFT (ms) | Mean/p99 TPOT (ms) |
| --------- | ----- | --------------- | ------------------ | ------------------ |
| pytorch   | off   | 386.08          | 280.43/811.66      | 81.19/87.53        |
| pytorch   | otlp  | 388.77          | 288.03/1148.12     | 80.51/82.37        |
| pytorch   | jsonl | 386.83          | 237.64/427.30      | 81.32/83.59        |
| pytorch   | both  | 384.78          | 283.24/613.41      | 81.35/83.78        |
| turbomind | off   | 1569.13         | 98.38/218.39       | 19.70/20.60        |
| turbomind | otlp  | 1484.83         | 93.41/225.82       | 20.93/21.89        |
| turbomind | jsonl | 1543.13         | 97.24/224.76       | 20.08/21.55        |
| turbomind | both  | 1529.32         | 95.66/232.08       | 20.28/21.08        |

The other workloads were chat with three choices (136 actual input tokens, 128 output tokens
per choice, concurrency 16, 90-second windows) and nonstreaming completions (2,048 input,
128 output tokens, concurrency 16, 60-second windows). All used temperature 0, seed 23 and
ignore-EOS with identical effective sampling settings across modes. TPOT is per choice;
nonstreaming client TTFT/TPOT remain unobserved. Throughput changes versus pooled off:

| Backend   | Workload               | OTLP   | JSONL  | Both   | Off-round spread |
| --------- | ---------------------- | ------ | ------ | ------ | ---------------- |
| pytorch   | stream 128/128 C32     | +0.70% | +0.19% | -0.34% | 0.21%            |
| pytorch   | chat n=3 C16           | +1.29% | +0.53% | +0.08% | 3.40%            |
| pytorch   | nonstream 2048/128 C16 | +0.28% | +0.45% | -0.27% | 0.05%            |
| turbomind | stream 128/128 C32     | -5.37% | -1.66% | -2.54% | 0.63%            |
| turbomind | chat n=3 C16           | -0.70% | +1.62% | +1.39% | 0.70%            |
| turbomind | nonstream 2048/128 C16 | -1.43% | -1.41% | -1.08% | 2.26%            |

Positive/negative changes include runtime variation; a small positive result is not evidence
that tracing speeds up inference. These measurements do not establish a universal overhead
bound or compare absolute backend performance.

An independent same-configuration TurboMind run used four 180-second streaming windows
in OTLP/off/OTLP/off order, reusing the loaded model. All 8,480 measured requests
succeeded and their expected records/spans were present. Pooled off throughput was
1529.72 tokens/s versus 1473.53 with OTLP
(-3.67%); the two off rounds differed by
1.25%. Both groups measured lower throughput with OTLP for this
workload. Include this observed cost in capacity planning; it is not a universal bound.

### Store, adapter identity and exporter failure

The existing PyTorch Mooncake Store connector passed eight cases on the same 480B checkpoint,
including repeated cache loads and two-choice requests with both the base model and a real
rank-4 zero-valued LoRA adapter. Request/response model names and shared-input usage matched
the API; the adapter's base model was recorded separately. The engine started with JSONL off,
and two-choice OTLP-only cases confirmed Store tracing works independently of JSONL.
Four warm requests each matched and loaded 1,984 tokens; observed lookup took 3.0–6.1 ms
and successful loads took 3.14–6.23 seconds. These are Store intervals, not logging overhead.
This validates adapter routing and metadata; trained-LoRA quality remains unverified.
Store used loopback TCP, 4 GB global segments and 256 MB
local buffers per worker to fit the 78.1 GiB Pod memory limit. It is not an RDMA or Store-overhead
benchmark. No extra lookup RPC or save synchronization was introduced.

With the receiver deliberately unavailable, all 96 inference requests succeeded and
all 192 JSONL records were written. After recovery, 16 HTTP spans remained missing,
consistent with bounded, non-durable export. This validates failure isolation, not guaranteed
delivery or zero latency impact.

The receiver and upstream parent spans were explicit local test fixtures. Production
Collector/Gateway contract deployment, TLS, cross-host clock synchronization,
ClickHouse/Grafana queries and verify-component acceptance remain unverified. Complete
that production-chain acceptance before deployment.
