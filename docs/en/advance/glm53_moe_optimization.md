# Experimental GLM-5.3-Flash decode optimizations

These opt-in PyTorch backend optimizations target GLM-5.3-Flash on SM90 GPUs. They are disabled by default.

## Available optimizations

| Environment variable                     | Effect                                                                                                 | Requirements                                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `LMDEPLOY_ENABLE_SKINNY_GEMM=1`          | Select measured Triton BF16 schedules for small KDA and KPool projections.                             | SM90, supported matrix and token-row shapes, no bias, deterministic mode disabled. Other shapes use the existing linear backend. |
| `LMDEPLOY_GLM53_SHARED_EXPERT_OVERLAP=1` | Run the shared expert on an auxiliary CUDA stream while routed experts execute.                        | DP8/EP8 with TP1, deterministic mode disabled, all DP ranks decoding.                                                            |
| `LMDEPLOY_GLM53_FUSED_MOE_PREPARE=1`     | Fuse routing, FP8 quantization and preparation of DeepEP's symmetric send buffer.                      | DP8/EP8 with TP1, deterministic mode disabled, all DP ranks decoding, matching DeepEP prepared-send build.                       |
| `LMDEPLOY_GLM53_FUSE_ROUTE_PREPARE=1`    | Fuse valid-token route masking and combine-weight scaling without changing DeepEP's quantization path. | GLM-5.3 low-latency DeepEP path. This is an alternative to full preparation and does not require the DeepEP patch.               |

The full preparation kernel specializes the GLM-5.3 geometry: BF16 hidden states of width 4096, 288 experts, top-8 sigmoid routing with correction bias, normalized routing weights scaled by 2.5, and E4M3 activations quantized in groups of 128. It preserves routing tie-breaking and the original FP32 scale calculation. It writes the 16-byte header, 4096 FP8 values and 32 FP32 scales directly into each 4240-byte DeepEP message.

Prefill and globally mixed prefill/decode steps retain the original MoE path. Explicit routed-expert collection also retains the original routing path. Shared-expert overlap preserves the original BF16 addition order and joins both streams before the result is consumed. No SM reservation is required by this implementation.

Skinny GEMM uses FP32 accumulation and BF16 output. It preserves parameter ownership and checkpoint names, but its reduction order can differ from the default GEMM. It does not establish token-identical full-model generation.

## Build the optional DeepEP extension

The full preparation path requires the bundled experimental patch against DeepEP `c997a4ece7831f617ae70d10696f419e9bd33f38`. It is not available in an unpatched DeepEP installation. The other optimization switches can be used independently.

Build against the same PyTorch, CUDA and NVSHMEM environment used for serving. Set `NVSHMEM_DIR` if DeepEP cannot discover the installed NVSHMEM package. CUDA 12.8 was used for the measured build.

```bash
export DEEPEP_SOURCE_DIR=/path/to/deepep-prepared-send
PYTHON=python3 CUDA_HOME=/usr/local/cuda-12.8 \
  bash benchmark/glm53_moe/build_deepep_prepare.sh "$DEEPEP_SOURCE_DIR"
export PYTHONPATH="$DEEPEP_SOURCE_DIR:${PYTHONPATH:-}"
```

The script builds in place and leaves the installed DeepEP package unchanged. The patched Python wrapper and compiled extension must be used together. Enabling full preparation with an unpatched provider raises an explicit error.

The patch adds a non-owning view of the next symmetric send buffer and a `prepared_send=False` argument to low-latency dispatch. With prepared sends, dispatch skips activation loading, quantization and message packing, while retaining the existing communication, receive and combine implementations. The producer and dispatch must run consecutively on the same stream, with no intervening low-latency buffer operation. The new path requires hidden width 4096, FP8 enabled, FP32 scales, no scale rounding and no UE8M0 scales. Existing Python calls use the original path by default.

## Enable the stack

Configure NCCL and NVSHMEM for the deployment network as usual. For example, after building the optional provider:

```bash
export LMDEPLOY_ENABLE_SKINNY_GEMM=1
export LMDEPLOY_GLM53_SHARED_EXPERT_OVERLAP=1
export LMDEPLOY_GLM53_FUSED_MOE_PREPARE=1
export LMDEPLOY_GLM53_FUSE_ROUTE_PREPARE=0

lmdeploy serve api_server "$MODEL_PATH" \
  --proxy-url "$ROUTER_URL" \
  --backend pytorch --quant-policy fp8 --dp 8 --ep 8 \
  --model-name glm53-flash --max-batch-size 32 \
  --cache-max-entry-count 0.7 --session-len 1048576 \
  --distributed-executor-backend ray \
  --speculative-algorithm deepseek_mtp --speculative-num-draft-tokens 5 \
  --reasoning-parser default --tool-call-parser glm47 --trust-remote-code \
  --prefill-save-alignment 64 --prefill-save-last-only \
  --prefix-cache-decode-save-first --prefix-cache-decode-state-interval 1024 \
  --enable-prefix-caching --prefix-cache-state-budget 32
```

MTP is optional; the example matches the AgentX measurement below. For the control arm, set all four optimization variables to `0`. Restart the model and router before each arm, keeping the same provider, model, network, cache policy and routing policy.

## Validation and performance

The kernel and model-hook tests cover exact routing/FP8 message bytes, changing valid-token counts during CUDA Graph replay, stream ordering, serial fallbacks, projection fallback behavior and FP64 projection references. The eight-rank microbenchmark checks bitwise equality against the unfused MoE layer with synthetic weights; it is not a full-model accuracy evaluation.

```bash
pytest -q tests/pytorch/kernel/test_glm53_fused_prepare.py \
  tests/pytorch/kernel/test_glm53_moe_prepare.py \
  tests/pytorch/kernel/test_skinny_gemm.py \
  tests/pytorch/test_glm53_shared_expert_overlap.py

torchrun --standalone --nproc-per-node=8 benchmark/glm53_moe/bench_layer.py \
  --model "$MODEL_PATH" --output layer-results.json
```

AgentX was measured with AIPerf 0.12.0 at commit `89b21867872a5bbc4b0676bf5005c404da5e9f94`, `inferencex-agentx-mvp` and `semianalysis_cc_traces_weka_062126_256k`. The baseline was LMDeploy `33c75413b91ecb5085f0913dc5d0e90e9f29c1c5`. Both arms used eight H200 GPUs, DP8/EP8, MTP with five draft tokens, consistent-hash routing, concurrency 80, identical decode checkpoint settings and separate 1800-second profiling windows. The model revision was `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`.

Run the same client against each freshly started serving arm:

```bash
AIPERF_URL="$ROUTER_URL/v1" TOKENIZER="$MODEL_PATH" \
  bash benchmark/glm53_moe/bench_agentx.sh artifacts/baseline
# Restart with the optimization stack enabled, then:
AIPERF_URL="$ROUTER_URL/v1" TOKENIZER="$MODEL_PATH" \
  bash benchmark/glm53_moe/bench_agentx.sh artifacts/candidate
```

Use a stable `X-Session-ID` with consistent-hash routing so conversation prefixes remain local. The client script enables the AIPerf session-affinity header. With this AIPerf revision, leave global `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` unset in the client when using a local tokenizer path. Serving can remain offline; `HF_DATASETS_OFFLINE=1` can separately reuse a prepared dataset cache.

| Metric                                            | Baseline | Optimization stack |                  Change |
| ------------------------------------------------- | -------: | -----------------: | ----------------------: |
| Output throughput, tokens/s                       |  1096.27 |            1144.50 |                  +4.40% |
| Goodput, requests/s                               |    0.526 |              0.605 |                 +15.10% |
| Mean TTFT, ms                                     |  1487.16 |            1427.69 |                  -4.00% |
| Mean whole-response inter-token latency, ms       |    34.77 |              31.97 |                  -8.05% |
| Mean whole-response per-user throughput, tokens/s |    45.12 |              48.27 |                  +6.99% |
| Prompt-cache token hit rate                       |   89.96% |             90.57% | +0.61 percentage points |

Changes use unrounded exported values. Goodput uses the requested SLOs: TTFT at most 3000 ms and output throughput per user at least 40 tokens/s. Whole-response metrics include reasoning and avoid an outlier in the default content-window TPS metric. The machine-readable results are in `benchmark/glm53_moe/results_h200_agentx.json`.

Both runs reported `submission_valid=true`. The baseline had 2262 successful requests and no request errors. The optimization stack had 2345 successful requests and one empty-visible-content response error for a request limited to two output tokens, about 0.043% of completed records; its cause is unresolved. This was one run per arm, with no replicated confidence interval. Full-model numerical equivalence, model accuracy and agent task success rates remain unvalidated. The measured gains belong to the complete stack and are not an isolated attribution to the DeepEP patch.
