# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os


def env_to_bool(
    env_var: str,
    default: bool = False,
    *,
    true_values: set | list = {'true', '1', 'yes', 'on'},
    false_values: set | list = {'false', '0', 'no', 'off'},
):
    """Env to bool."""
    value = os.getenv(env_var)
    if value is None:
        return default
    value = value.lower().strip()
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert environment variable '{env_var}={value}' to boolean. "
                         f'Allowed true values: {true_values}, false values: {false_values}')


def env_to_int(
    env_var: str,
    default: int = 0,
):
    """Env to int."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        value = int(value)
    except Exception:
        value = default
    return value


def env_to_list_int(
    env_var: str,
    default: list[int] = None,
):
    """Env to list of int."""
    default_ = default if default is not None else []
    value = os.getenv(env_var)
    if value is None:
        return default_
    try:
        value = [int(x) for x in value.split(',')]
    except Exception:
        value = default_
    return value


def env_to_float(
    env_var: str,
    default: float = 0,
):
    """Env to float."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        value = default
    return value


def env_to_choice(
    env_var: str,
    default: str,
    choices: set | list,
):
    """Env to selected string."""
    value = os.getenv(env_var)
    if value is None:
        return default
    value = value.lower().strip()
    if value not in choices:
        raise ValueError(f"Invalid environment variable '{env_var}={value}'. Allowed values: {choices}")
    return value


_ENVS = dict()


@contextlib.contextmanager
def set_envs():
    _origin_get_env = os.getenv

    def _patched_get_env(
        env_var: str,
        default: str | None = None,
    ):
        """Patched get_env."""
        if env_var in os.environ:
            _ENVS[env_var] = os.environ[env_var]

        return _origin_get_env(env_var, default)

    os.getenv = _patched_get_env
    yield
    os.getenv = _origin_get_env


with set_envs():
    # loader
    random_load_weight = env_to_bool('LMDEPLOY_RANDOM_LOAD_WEIGHT', True)

    # profile
    ray_nsys_enable = env_to_bool('LMDEPLOY_RAY_NSYS_ENABLE', False)
    ray_nsys_output_prefix = os.getenv('LMDEPLOY_RAY_NSYS_OUT_PREFIX', None)

    # ascend
    ascend_set_rt_visable_devices_by_ray = env_to_bool('ASCEND_SET_RT_VISIBLE_DEVICES_BY_RAY', False)

    # dp
    dp_master_addr = os.getenv('LMDEPLOY_DP_MASTER_ADDR', None)
    dp_master_port = os.getenv('LMDEPLOY_DP_MASTER_PORT', None)

    # executor
    executor_backend = os.getenv('LMDEPLOY_EXECUTOR_BACKEND', None)

    # torch profiler
    torch_profile_cpu = env_to_bool('LMDEPLOY_PROFILE_CPU', False)
    torch_profile_cuda = env_to_bool('LMDEPLOY_PROFILE_CUDA', False)
    torch_profile_delay = env_to_int('LMDEPLOY_PROFILE_DELAY', 0)
    torch_profile_duration = env_to_int('LMDEPLOY_PROFILE_DURATION', -1)
    torch_profile_output_prefix = os.getenv('LMDEPLOY_PROFILE_OUT_PREFIX', 'lmdeploy_profile_')

    # v4 ascend npu profiler (torch_npu.profiler == msprof for PyTorch).
    # Activated on rank0 only, scoped between two sentinel files so it wraps
    # exactly the measured workload (post graph-capture warmup).
    os.getenv('V4_PROFILE', '0')
    os.getenv('V4_PROFILE_DIR', '/deeplink/swang/claude_kfj/profile_out')
    os.getenv('V4_PROFILE_STEPS', '400')
    os.getenv('V4_PROFILE_START', '/tmp/v4_prof_start')
    os.getenv('V4_PROFILE_STOP', '/tmp/v4_prof_stop')
    os.getenv('V4_DEBUG_ALLOC', '0')
    os.getenv('V4_DEBUG_MOE', '0')
    # Phase1: precompute main RoPE cos/sin as persistent [session_len,...] table
    # (see v4_dsa.py build_v4_dsa_inputs). Must be registered here so the
    # driver-set value is captured into _ENVS and propagated to ray workers
    # (workers run build_v4_dsa_inputs, not the driver).
    os.getenv('V4_PRECOMPUTE_ROPE', '0')
    os.getenv('V4_COMP_DUMP', '0')
    # Phase1 step2: source seq_len/kv_len/total_q from CPU-side scheduler ints
    # in build_v4_dsa_inputs instead of .item() host syncs on NPU seqlens.
    os.getenv('V4_CPU_SEQLENS', '0')
    # FULL-graph C1 step1: bake max_seqlen_kv to a session_len ceiling in
    # build_v4_dsa_inputs (decode only). Verifies max_seqlen_kv is a pure
    # sizing scalar (vllm-ascend bakes it at capture, no-op update_graph_params)
    # so it can be captured as a fixed host arg in a FULL aclgraph.
    os.getenv('V4_META_CEILING', '0')
    # FULL-graph C1 step2b: defer the metadata op calls to the captured
    # attention forward (build them in-graph -> 0 host dispatch). build_v4_dsa
    # passes None sentinels + the baked ceiling; DeepseekV4Attention.forward
    # builds the op in-graph when it sees None.
    os.getenv('V4_META_IN_GRAPH', '0')
    # FULL-graph C1 step2b: set to '1' by the model-agent decode-warmup loop
    # (in-process, per worker) so build_v4_dsa_inputs inflates the SAS op's
    # seqused_kv to the session_len ceiling during graph capture (mirrors
    # vllm-ascend SEQ_LEN_WITH_MAX_PA_WORKSPACE=6144). Not driver-set; no ray
    # propagation needed (warmup + build_v4_dsa_inputs run in the same worker).
    os.getenv('V4_GRAPH_CAPTURING', '0')
    # Isolation gate for the lightning-indexer metadata op (default = follow
    # V4_META_IN_GRAPH). =0 keeps QLI eager while SAS can be in-graph, to
    # isolate which aclnn metadata op causes a 2nd-replay 507011.
    os.getenv('V4_QLI_IN_GRAPH', '0')
    # One device-wide torch.npu.synchronize() before the decode graph replay
    # (model_forward, right before model(**input_dict)). Tests whether a SINGLE
    # pre-replay sync masks the nondeterministic 507018 eager-op/replay timing
    # race (V4_DEBUG_ALLOC=1 masks it via many .item() syncs). Cheap workaround
    # candidate + confirms cross-stream/serialization race if it stabilizes.
    os.getenv('V4_PRE_REPLAY_SYNC', '0')
    # V4_DEVICE_SYNC: switch the per-replay stream drain (ascend_cudagraph
    # AscendSingleGraphRunner.forward, mirrors vllm-ascend acl_graph.py:199)
    # from current_stream().synchronize() to torch.npu.synchronize() (device-
    # wide, drains the HCCL internal stream too). This is a DETECTOR, not a
    # fixer: it surfaces the residual in-graph vector/HCCL racer (507035
    # "cross-device memory access timeout" on the EP rank) at ~req9 instead of
    # q56 -- a fast-crash ORACLE for fix iteration (a real fix keeps this
    # stable). Registered so the driver-set value reaches ray workers.
    os.getenv('V4_DEVICE_SYNC', '0')
    # Force the dlinfer per-step graph-update path. The torch_npu>=2.8 path
    # (replay -> _graph.update) orders the actual_seq_lengths_kv CPU-param
    # write AFTER replay with no sync -- races the next replay (507018, see
    # vllm-ascend acl_graph.py:243). =0 forces the legacy update_attn_params
    # (separate update_stream -> replay) path = vllm-ascend's stable ordering.
    os.getenv('V4_FORCE_ACL_UPDATE_PATH', '')
    # V4_SKIP_ACL_UPDATE: skip the vestigial per-step
    # _graph.update(actual_seq_lengths_kv) in the torch_npu_update replay
    # path. V4's attention op (sparse_attn_sharedkv) reads kv length from a
    # device seqused_kv tensor, NOT the standard op's actual_seq_lengths_kv
    # CPU param -- so V4's captured graph has no such param and the update is
    # a no-op-with-side-effect (mirrors vllm-ascend dsa_v1 update_graph_params
    # no-op). Registered so the driver-set value reaches ray workers.
    os.getenv('V4_SKIP_ACL_UPDATE', '0')
    # FULL-graph decode metadata (V4_FULL_GRAPH_DECODE=1): compute the SHARED
    # decode metadata (cos gather / slot_mapping / swa_block_table / start_pos
    # / _pin_meta copies -- ~26 per-step eager aclnn ops) IN-GRAPH inside the
    # captured DeepseekV4Model.forward (v4_dsa.build_v4_decode_meta_in_graph)
    # from the device graph-input buffers (kv_seqlens / block_offsets /
    # position_ids), instead of eagerly in build_v4_dsa_inputs. Eliminates the
    # eager metadata racers that overlap the graph replay CANN workspace
    # (507018 timing race) -> 0 eager metadata ops between replays (mirrors
    # vllm-ascend FULL graph) -> stable. build_v4_dsa_inputs passes None
    # sentinels for the shared fields once a graph is captured (has_captured);
    # the model forward fills them in-graph (gated on kv_seqlens being a device
    # tensor = capture/replay path). Requires V4_PRECOMPUTE_ROPE=1 +
    # V4_META_CALLONCE=1. Read in v4_dsa.build_v4_dsa_inputs +
    # deepseek_v4.DeepseekV4Model.forward + ascend_cudagraph (device kv_seqlens).
    os.getenv('V4_FULL_GRAPH_DECODE', '0')
    # one-shot diagnostic: dump the build_v4_dsa_inputs _full_graph gating
    # decision (is_decoding / global_decoding / has_captured / full_graph /
    # kv device) for the first decode build_context calls on rank0, to settle
    # whether a single-request DP2 step replays (global_decoding=True) or
    # falls back to eager-decode (False). Read in v4_dsa.build_v4_dsa_inputs.
    os.getenv('V4_GATING_PROBE', '0')
    # Diagnostic: feed the SAS op's seqused_kv = ceiling at EVERY replay (not
    # just capture) to test whether the op sizes internal UB/temps at the first
    # replay. See v4_dsa.py build_v4_dsa_inputs.
    os.getenv('V4_SAS_FREEZE', '0')
    # Prefill MoE via fused_moe_all2all (A3 ALLTOALL steps). Read in
    # deepseek_v4.py MoE.forward; registered so the driver export is captured
    # into _ENVS and propagated to ray workers (=mc2 decode routing path).
    os.getenv('V4_PREFILL_ALL2ALL', '0')
    # Paged per-request state block table (compressor state_bt). The default
    # contiguous arange scheme caps columns at max_prefill_token_num//state_bs
    # (2432-token ceiling) -> aicore MTE OOB past that. Paged sizing uses
    # session_len//state_bs columns + a shared-budget pool (no HBM growth) so
    # multi-chunk prefill / long decode works. Read in v4_dsa.build_v4_dsa_inputs.
    os.getenv('V4_STATE_PAGED', '0')
    # State pool budget (decouple pool HBM from max_batch) + sliding-window
    # reclamation (V4_STATE_WINDOW: keep last W state blocks/req live, null
    # older -> free-list, so max_batch is decoupled from decode length too).
    # All read via os.environ.get in v4_dsa.py; registered here so the
    # driver-set values are captured into _ENVS and propagated to ray workers
    # (envs.py's set_envs patch only captures os.getenv reads in THIS block,
    # NOT os.environ.get reads in other modules).
    os.getenv('V4_STATE_DECODE_MARGIN', '0')
    os.getenv('V4_STATE_POOL_BLOCKS_C4', '0')
    os.getenv('V4_STATE_POOL_BLOCKS_C128', '0')
    os.getenv('V4_STATE_WINDOW', '0')
    # V4_DSA_KV_POOL_BLOCKS: fixed budget (in MLA_BS=128 blocks) for the
    # compress_kv / indexer_k / indexer_scale POOL TENSORS (separate from the
    # swa pool). Read via os.environ.get in v4_dsa.allocate_v4_caches +
    # _dsa_kv_pool_size on the WORKER; registered here so the driver-set value
    # is captured into _ENVS and propagated to ray workers (same pattern as the
    # V4_STATE_POOL_BLOCKS_* vars above). 0 -> auto = max_num_seqs *
    # cdiv(session_len, MLA_BS).
    os.getenv('V4_DSA_KV_POOL_BLOCKS', '0')
    # Diagnostic: log per-rank DP forward meta (local/all_num_tokens,
    # all_batch_sizes) in agent._prepare_dp_v1, to see whether internal DP
    # actually splits the batch 1/1 across DP groups or leaves dp_rank>0 dummy.
    os.getenv('V4_DP_DEBUG', '0')
    # Option A: route decode batch-change steps eager (no replay race, 507035).
    # =1 (default) enables the eager routing in ascend_cudagraph.__call__;
    # =0 disables (ablation / revert). Registered so it reaches ray workers.
    os.getenv('V4_BATCH_CHANGE_EAGER', '1')
    # Diagnostic: print [V4-BCE] when a batch-change step is routed eager.
    os.getenv('V4_BC_EAGER_DEBUG', '0')
    # Diagnostic: per-step decode timing (eager build_context host dispatch vs
    # sync'd forward/replay wall). Read in agent.model_forward (rank0 print).
    os.getenv('V4_STEP_TIME', '0')
    # Diagnostic: one-shot CPU-activity torch.profiler op-count over a single
    # decode forward (step 6) to attribute per-fwd aten op counts.
    os.getenv('V4_OPCOUNT', '0')
    # Diagnostic: fingerprint the eager SAS metadata at decode step1 vs step2
    # (kv_len 6 vs 7) to test whether it is kv_len-invariant (decides whether
    # in-graph SAS capture is even possible). Read in v4_dsa._build_sas_metadata.
    os.getenv('V4_SAS_INVARIANCE', '0')
    # Cross-step call-once cache for the SAS + QLI metadata OPs (mirrors
    # vllm-ascend's decode_ratio_to_sas_metadata: compute both ops ONCE per
    # (cr, num_reqs) [SAS] / num_reqs [QLI], reuse across decode steps -- the
    # op outputs are sizing/layout descriptors, kv_len-invariant; the real
    # per-step kv_len flows through separate KV tensors at the indexer call).
    # Cuts the 3 SAS + 1 QLI op calls/step to 0 after the first decode step of
    # a stable batch. Read in v4_dsa.build_v4_dsa_inputs.
    os.getenv('V4_META_CALLONCE', '0')
    # Diagnostic: one-shot pure-Python op-count over a single DECODE
    # build_v4_dsa_inputs call (the build_host ~5ms eager phase, NOT the
    # captured forward). Attributes the remaining build_host torch ops.
    # Read in op_backend._maybe_build_v4_dsa.
    os.getenv('V4_BUILD_OPCOUNT', '0')

    # ray timeline
    ray_timeline_enable = env_to_bool('LMDEPLOY_RAY_TIMELINE_ENABLE', False)
    ray_timeline_output_path = os.getenv('LMDEPLOY_RAY_TIMELINE_OUT_PATH', 'ray_timeline.json')

    # ray external placement group bundles
    # only used when lmdeploy is initialized inside a Ray Actor with pg allocated
    ray_external_pg_bundles = env_to_list_int('LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES', [])

    # enable ray zero-copy tensors
    os.getenv('RAY_ENABLE_ZERO_COPY_TORCH_TENSORS', '1')

    # dist
    dist_master_addr = os.getenv('LMDEPLOY_DIST_MASTER_ADDR', None)
    dist_master_port = os.getenv('LMDEPLOY_DIST_MASTER_PORT', None)

    # logging
    log_file = os.getenv('LMDEPLOY_LOG_FILE', None)
    os.getenv('LMDEPLOY_LOG_PID', '0')

    # check env
    enable_check_env = env_to_bool('LMDEPLOY_ENABLE_CHECK_ENV', True)

    # hccl / ascend - passed to ray workers
    os.getenv('HCCL_BUFFSIZE', None)
    os.getenv('HCCL_CONNECT_TIMEOUT', None)
    os.getenv('HCCL_OP_EXPANSION_MODE', None)
    os.getenv('HCCL_IF_IP', None)

    # deepep
    os.getenv('DEEPEP_ENABLE_MNNVL', None)
    os.getenv('DEEPEP_MODE', 'auto')
    deep_ep_buffer_num_sms = env_to_int('DEEPEP_BUFFER_NUM_SMS', 20)

    # eplb
    eplb_num_groups = env_to_int('LMDEPLOY_EPLB_NUM_GROUPS', 4)
    eplb_experts_statistic_file = os.getenv('LMDEPLOY_EPLB_EXPERTS_STATISTIC_FILE', None)
    eplb_ranks_per_node = env_to_int('LMDEPLOY_EPLB_RANKS_PER_NODE', 8)
    eplb_num_redundant_experts = env_to_int('LMDEPLOY_EPLB_NUM_REDUNDANT_EXPERTS', 32)

    # deepgemm
    os.getenv('DG_JIT_DEBUG', '0')
    os.getenv('DG_JIT_PRINT_COMPILER_COMMAND', '0')

    # model agent
    skip_warmup = env_to_bool('LMDEPLOY_SKIP_WARMUP', False)

    # memory trim
    multimodal_session_trim_count = env_to_int('LMDEPLOY_MULTIMODAL_SESSION_TRIM_COUNT', 128)

    # model format
    scale_fmt = os.getenv('LMDEPLOY_SCALE_FMT', None)

    # repetition check
    repetition_window_size = env_to_int('LMDEPLOY_REPETITION_WINDOW_SIZE', 1024)

    # qwen3.5 recurrent_state dtype
    fp32_mamba_ssm_dtype = env_to_bool('LMDEPLOY_FP32_MAMBA_SSM_DTYPE', False)

    # cudagraph
    # fake capture flag for debug cudagraph padding behavior
    fake_capture = env_to_bool('LMDEPLOY_FAKE_CUDA_GRAPH_CAPTURE', False)

    # opt-ttft
    opt_ttft_policy = env_to_choice('LMDEPLOY_PT_TTFT_POLICY', 'size', {'fifo', 'size'})
    opt_ttft_short_turns = max(1, env_to_int('LMDEPLOY_PT_TTFT_SHORT_TURNS', 3))
    opt_ttft_aging_sec = env_to_float('LMDEPLOY_PT_TTFT_AGING_SEC', 2.0)


def get_all_envs():
    """Get all environment variables."""
    return _ENVS
