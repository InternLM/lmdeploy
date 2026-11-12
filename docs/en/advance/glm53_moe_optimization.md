# Experimental GLM-5.3-Flash decode optimizations

These opt-in PyTorch backend optimizations target GLM-5.3-Flash on SM90 GPUs. All switches default to disabled. They use the existing DeepEP Python API and require no DeepEP patch or custom build.

## Available optimizations

| Environment variable                     | Effect                                                                                       | Requirements                                                                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `LMDEPLOY_ENABLE_SKINNY_GEMM=1`          | Select measured Triton BF16 schedules for small KDA and KPool projections.                   | SM90, supported matrix/token-row shapes, no bias, deterministic mode disabled. Other shapes use the existing linear backend. |
| `LMDEPLOY_GLM53_SHARED_EXPERT_OVERLAP=1` | Run the shared expert on an auxiliary CUDA stream while routed experts execute.              | DP8/EP8 with TP1, deterministic mode disabled, all DP ranks decoding.                                                        |
| `LMDEPLOY_GLM53_FUSE_ROUTE_PREPARE=1`    | Fuse sigmoid/top-8 routing, valid-token masking, int64 ID output and combine-weight scaling. | GLM-5.3 DP8/EP8 with TP1, deterministic mode disabled, all DP ranks decoding.                                                |

The fused route kernel follows the official GLM-5.3 contract: 288 experts, sigmoid scoring with correction bias, top-8 selection with the original tie-breaking, normalized FP32 weights scaled by 2.5, and int64 IDs. It reads the graph-owned valid-token count on device, so padding and idle DP ranks remain masked when CUDA Graphs replay with a different batch.

The decode pipeline is:

```text
router GEMM
  -> fused routing + masking + combine-weight scaling
  -> native DeepEP dispatch(BF16 input, built-in FP8 quantization/packing/communication)
  -> routed experts and shared expert
  -> native DeepEP combine + original BF16 shared-expert addition
```

The original BF16 hidden states are passed to `low_latency_dispatch(..., use_fp8=True)` with its existing arguments. This implementation neither accesses DeepEP's private send buffers nor bypasses its quantization or packing. Routing, quantization and staging are not combined into a single kernel: the native interface accepts BF16 activations and owns the latter two operations.

Prefill and globally mixed steps retain the original routing path. Explicit routed-expert collection also retains the original router, with masking and combine scaling fused in the low-latency dispatcher. Shared-expert overlap joins the streams before the original BF16 addition. No SM reservation is introduced.

Skinny GEMM preserves parameter ownership and checkpoint names and uses FP32 accumulation with BF16 output. Its reduction order can differ from the default GEMM, so it does not establish token-identical full-model generation.

## Enable the stack

Use the normal DeepEP installation and configure NCCL/NVSHMEM for the deployment network as usual. No provider overlay or custom extension is needed.

```bash
export LMDEPLOY_ENABLE_SKINNY_GEMM=1
export LMDEPLOY_GLM53_SHARED_EXPERT_OVERLAP=1
export LMDEPLOY_GLM53_FUSE_ROUTE_PREPARE=1

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

MTP is optional. For a control arm, set all three optimization variables to `0`; keep the model, provider, routing and cache policies identical and restart the service before each arm. Multi-turn tests through a consistent-hash router should preserve a stable session identifier.

## Validation and performance scope

```bash
pytest -q tests/pytorch/kernel/test_glm53_fused_prepare.py \
  tests/pytorch/kernel/test_glm53_moe_prepare.py \
  tests/pytorch/kernel/test_skinny_gemm.py \
  tests/pytorch/test_glm53_shared_expert_overlap.py \
  tests/pytorch/test_glm53_native_deepep.py
```

Tests cover routing weights/IDs against the original router, ties and extreme scores, live valid-token counts during CUDA Graph replay, stream dependencies, serial fallbacks, FP64 projection references and the original BF16 DeepEP dispatch signature. The native-interface test deliberately supplies no prepared-send argument or send-buffer getter.

Full-duration AgentX performance, full-model numerical equivalence, model accuracy and agent success rates require separate evaluation for this implementation.
