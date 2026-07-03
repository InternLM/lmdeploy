# Copyright (c) OpenMMLab. All rights reserved.
import ast
import builtins
import importlib
from pathlib import Path

import pytest
import torch


def test_dist_checker_requires_deepep_and_deepgemm(monkeypatch):
    from lmdeploy.pytorch.check_env import dist as dist_check

    failures = []
    monkeypatch.setattr(dist_check, 'is_deep_ep_installed', lambda: False)
    monkeypatch.setattr(dist_check, 'is_deep_gemm_installed', lambda: True)

    checker = dist_check.DistChecker(tp=1, dp=1, ep=2, distributed_executor_backend='mp', device_type='cuda')
    monkeypatch.setattr(checker, 'log_and_exit', lambda **kwargs: failures.append(kwargs))

    checker.check()

    assert failures
    assert 'DeepEP' in failures[0]['message']
    assert 'DeepGEMM' in failures[0]['message']
    assert 'dl' + 'blas' not in failures[0]['message'].lower()


def test_eplb_metadata_and_dispatch_mapping(monkeypatch):
    from lmdeploy.pytorch.nn import eplb

    physical_to_logical = torch.tensor([[0, 1, 1]])
    logical_to_all_physical = torch.tensor([[[0, -1], [1, 2]]])
    metadata = eplb.EPLBMetadata._init_raw(
        ep_size=1,
        physical_to_logical_map=physical_to_logical,
        logical_to_all_physical_map=logical_to_all_physical,
    )
    monkeypatch.setattr(eplb, '_global_eplb_metadata', metadata)

    assert eplb.EPLBManager.num_physical_experts() == 3
    assert eplb.get_eplb_phy2log_metadata_by_layer(0).tolist() == [0, 1, 1]

    info = eplb.EPLBManager.get_dispatch_info(ep_rank=0, layer_idx=0)
    topk_ids = torch.tensor([[0, 1, 1]])
    physical = eplb.EPLBManager.topk_ids_logical_to_physical(topk_ids, info)

    assert physical[0, 0].item() == 0
    assert physical[0, 1].item() in (1, 2)
    assert physical[0, 2].item() in (1, 2)


def test_deepep_buffer_uses_internal_default_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None
        build_count = 0

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False, **kwargs):
            FakeBuffer.build_count += 1
            self.group = group
            self.num_nvl_bytes = num_nvl_bytes
            self.num_rdma_bytes = num_rdma_bytes
            self.low_latency_mode = low_latency_mode
            self.kwargs = kwargs
            self.destroyed = False
            self.group_size = group.size()

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

        def destroy(self):
            self.destroyed = True

        def clean_low_latency_buffer(self, *args):
            self.clean_args = args

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setenv('DEEPEP_MAX_TOKENS' + '_PER_RANK', '999')
    monkeypatch.setenv('DEEPEP_BUFFER_NUM_SMS', '13')
    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._buffer_normal = None
    td.DeepEPBuffer._buffer_low_latency = None
    td.DeepEPBuffer._explicitly_destroy = False
    td.DeepEPBuffer._deepep_sms = 20
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128
    FakeBuffer.build_count = 0

    assert td.DeepEPBuffer.set_explicitly_destroy() is True
    buffer = td.DeepEPBuffer.get_buffer_common(FakeGroup(), 128, hidden=16, num_experts=4, hidden_bytes=32)
    reused_buffer = td.DeepEPBuffer.get_buffer_common(FakeGroup(), 256, hidden=32, num_experts=4, hidden_bytes=1024)

    assert FakeBuffer.low_latency_size_hint_args[0] == 128
    assert FakeBuffer.build_count == 1
    assert reused_buffer is buffer
    assert buffer.kwargs['explicitly_destroy'] is True
    assert buffer.kwargs['num_qps_per_rank'] == 13
    assert buffer.num_sms == 13
    assert td.DeepEPBuffer.destroy() is True
    assert buffer.destroyed is True
    assert td.DeepEPBuffer.destroy() is False


def test_disposible_tensor_dispose_is_best_effort_with_extra_refs():
    from lmdeploy.pytorch.backends.cuda.token_dispatcher import DisposibleTensor

    tensor = torch.empty(1)
    wrapped = DisposibleTensor(tensor)
    extra_refs = [tensor]

    wrapped.dispose()

    assert wrapped.value is tensor
    assert extra_refs[0] is tensor


def test_low_latency_dispatcher_accepts_explicit_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, *args, **kwargs):
            self.group = group
            self.num_nvl_bytes = kwargs.get('num_nvl_bytes', args[0] if len(args) > 0 else 0)
            self.num_rdma_bytes = kwargs.get('num_rdma_bytes', args[1] if len(args) > 1 else 0)
            self.low_latency_mode = kwargs.get('low_latency_mode', False)
            self.group_size = group.size()

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128

    dispatcher = td.DeepEPTokenDispatcherLowLatency(
        group=FakeGroup(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=256,
    )

    assert dispatcher.num_max_dispatch_tokens_per_rank == 256
    assert FakeBuffer.low_latency_size_hint_args[0] == 256


def test_normal_dispatcher_accepts_explicit_token_limit_for_common_buffer(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, *args, **kwargs):
            self.group = group
            self.num_nvl_bytes = kwargs.get('num_nvl_bytes', args[0] if len(args) > 0 else 0)
            self.num_rdma_bytes = kwargs.get('num_rdma_bytes', args[1] if len(args) > 1 else 0)
            self.low_latency_mode = kwargs.get('low_latency_mode', False)

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128

    dispatcher = td.DeepEPTokenDispatcherNormal(
        group=FakeGroup(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=256,
    )

    assert dispatcher.num_max_dispatch_tokens_per_rank == 256
    assert FakeBuffer.low_latency_size_hint_args[0] == 256


def test_deepep_token_limit_is_inferred_from_engine_max_batch_size():
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
    from lmdeploy.pytorch.model_inputs import BuildModelContext

    engine_config = PytorchEngineConfig(max_batch_size=32)
    cache_config = ConfigBuilder.build_cache_config(engine_config)
    build_ctx = BuildModelContext(max_batch_size=cache_config.max_batches, num_spec_tokens=3)

    assert cache_config.max_batches == 32
    assert build_ctx.deep_ep_max_tokens_per_rank == 128


def test_all_fused_moe_builders_accept_deepep_token_limit():
    def build_args(module_path, class_name):
        tree = ast.parse((Path(__file__).parents[3] / module_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'build':
                        return [arg.arg for arg in item.args.args]
        raise AssertionError(f'{class_name}.build not found')

    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/cuda/moe/default.py',
                                                            'TritonFusedMoEBuilder')
    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py',
                                                            'TritonFusedMoEV4FP4Builder')
    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/cuda/moe/v4_fp4.py',
                                                            'DeepGemmFusedMoEV4Builder')
    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/dlinfer/moe.py',
                                                            'DlinferFusedMoEBuilder')


def test_eplb_env_vars_are_lmdeploy_prefixed():
    envs_text = (Path(__file__).parents[3] / 'lmdeploy/pytorch/envs.py').read_text()

    assert "'LMDEPLOY_EPLB_NUM_GROUPS'" in envs_text
    assert "'LMDEPLOY_EPLB_EXPERTS_STATISTIC_FILE'" in envs_text
    assert "'LMDEPLOY_EPLB_RANKS_PER_NODE'" in envs_text
    assert "'LMDEPLOY_EPLB_NUM_REDUNDANT_EXPERTS'" in envs_text

    old_env_vars = [
        'EPLB' + '_NUM_GROUPS',
        'EPLB' + '_EXPERTS_STATISTIC_FILE',
        'RANKS' + '_PER_NODES',
        'EPLB' + '_NUM_REDUNDANT_EXPERTS',
    ]
    for env_var in old_env_vars:
        assert f"'{env_var}'" not in envs_text


def test_imports_do_not_require_removed_or_ep_only_packages(monkeypatch):
    real_import = builtins.__import__
    blocked_package = 'dl' + 'blas'

    def guarded_import(name, *args, **kwargs):
        if (name == blocked_package or name.startswith(blocked_package + '.') or name == 'deep_gemm'
                or name.startswith('deep_gemm.')):
            raise AssertionError(f'unexpected optional package import: {name}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    modules = [
        'lmdeploy.pytorch.backends.cuda.moe',
        'lmdeploy.pytorch.backends.cuda.moe.default',
        'lmdeploy.pytorch.backends.cuda.moe.blocked_fp8',
        'lmdeploy.pytorch.backends.cuda.graph_runner',
        'lmdeploy.pytorch.nn.eplb',
        'lmdeploy.pytorch.check_env.dist',
    ]
    for module in modules:
        importlib.import_module(module)


def test_eplb_global_metadata_uses_explicit_runtime_errors(monkeypatch):
    from lmdeploy.pytorch.nn import eplb

    monkeypatch.setattr(eplb, '_global_eplb_metadata', None)
    with pytest.raises(RuntimeError, match='not been initialized'):
        eplb.get_global_eplb_metadata()

    monkeypatch.setattr(eplb, '_global_eplb_metadata', object())
    with pytest.raises(RuntimeError, match='already been initialized'):
        eplb.init_global_eplb_metadata(ep_size=1, num_routed_experts=1, num_hidden_layers=1)


def test_fp8_ep_prefill_quant_uses_configured_dtype_and_scale_fmt(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8

    calls = []

    def fake_quant(x, block_size, dtype=None, scale_fmt=None):
        calls.append((x, block_size, dtype, scale_fmt))
        return 'quant', 'scale'

    monkeypatch.setattr(blocked_fp8, 'per_token_group_quant_fp8', fake_quant)

    fusedmoe = blocked_fp8.FusedMoENormal.__new__(blocked_fp8.FusedMoENormal)
    fusedmoe.block_size = 64
    fusedmoe.fp8_dtype = torch.float8_e5m2
    fusedmoe.scale_fmt = 'ue8m0'

    hidden_states = object()
    assert fusedmoe.per_token_group_quant_fp8(hidden_states) == ('quant', 'scale')
    assert calls[-1] == (hidden_states, 64, torch.float8_e5m2, 'ue8m0')

    assert fusedmoe.per_token_group_quant_fp8(hidden_states, dtype=torch.float8_e4m3fn, scale_fmt=None) == ('quant',
                                                                                                           'scale')
    assert calls[-1] == (hidden_states, 64, torch.float8_e4m3fn, 'ue8m0')


def test_fp8_ep_builder_passes_activation_dtype_and_scale_fmt(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8

    calls = []

    def fake_build_deepep_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return 'moe'

    monkeypatch.setattr(blocked_fp8, 'build_deepep_moe', fake_build_deepep_moe)
    impl = blocked_fp8.FusedDeepEpMoEBlockedF8Impl.__new__(blocked_fp8.FusedDeepEpMoEBlockedF8Impl)
    impl.ep_size = 2
    impl.ep_group = object()
    impl.num_experts = 8
    impl.hidden_dim = 16
    impl.block_size = 64
    impl.top_k = 2
    impl.out_dtype = torch.bfloat16
    impl.fp8_dtype = torch.float8_e5m2
    impl.scale_fmt = 'ue8m0'
    impl.num_max_dispatch_tokens_per_rank = 256
    impl.layer_idx = 3

    assert blocked_fp8.FusedDeepEpMoEBlockedF8Impl.fusedmoe_build(impl, low_latency_mode=False) == 'moe'

    assert calls[0][1]['fp8_dtype'] == torch.float8_e5m2
    assert calls[0][1]['scale_fmt'] == 'ue8m0'
    assert calls[0][1]['num_max_dispatch_tokens_per_rank'] == 256


def test_bf16_ep_builder_passes_low_latency_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import default

    calls = []

    def fake_build_deepep_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return 'moe'

    monkeypatch.setattr(default, 'build_deepep_moe', fake_build_deepep_moe)
    impl = default.FusedMoEEPImpl.__new__(default.FusedMoEEPImpl)
    impl.ep_size = 2
    impl.ep_group = object()
    impl.num_experts = 8
    impl.hidden_dim = 16
    impl.top_k = 2
    impl.layer_idx = 3
    impl.out_dtype = torch.bfloat16
    impl.num_max_dispatch_tokens_per_rank = 256

    assert default.FusedMoEEPImpl.fusedmoe_build(impl, low_latency_mode=True) == 'moe'

    assert calls[0][1]['num_max_dispatch_tokens_per_rank'] == 256


def test_v4_fp4_ep_builder_passes_low_latency_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import v4_fp4

    calls = []

    class FakeMoE:

        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(v4_fp4, 'V4FP4FusedMoENormal', FakeMoE)
    monkeypatch.setattr(v4_fp4, 'V4FP4FusedMoELowLatency', FakeMoE)
    impl = v4_fp4.TritonFusedMoEV4FP4EPImpl.__new__(v4_fp4.TritonFusedMoEV4FP4EPImpl)
    impl.ep_size = 2
    impl.ep_group = object()
    impl.num_experts = 8
    impl.num_local_experts = 4
    impl.hidden_dim = 16
    impl.ffn_dim = 32
    impl.top_k = 2
    impl.swiglu_limit = 0.0
    impl.scale_fmt = 'ue8m0'
    impl.layer_idx = 3
    impl.num_max_dispatch_tokens_per_rank = 256

    assert v4_fp4.TritonFusedMoEV4FP4EPImpl.fusedmoe_build(impl, low_latency_mode=True).__class__ is FakeMoE
    assert calls[0]['num_max_dispatch_tokens_per_rank'] == 256

    calls.clear()
    assert v4_fp4.TritonFusedMoEV4FP4EPImpl.fusedmoe_build(impl, low_latency_mode=False).__class__ is FakeMoE
    assert calls[0]['num_max_dispatch_tokens_per_rank'] == 256


def test_v4_fp4_ep_forward_uses_global_decode_mode(monkeypatch):
    from lmdeploy.pytorch import model_inputs
    from lmdeploy.pytorch.backends.cuda.moe import ep_utils, v4_fp4

    modes = []

    class FakeMoE:

        def forward(self,
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    gate_up_weight,
                    gate_up_scale,
                    down_weight,
                    down_scale,
                    expert_list=None):
            return hidden_states

    class FakeStepContext:
        is_decoding = False

        def global_is_decoding(self):
            return True

    class FakeStepContextManager:

        def current_context(self):
            return FakeStepContext()

    def fake_fusedmoe_build(low_latency_mode=False):
        modes.append(low_latency_mode)
        return FakeMoE()

    monkeypatch.setattr(model_inputs, 'get_step_ctx_manager', lambda: FakeStepContextManager())
    monkeypatch.setattr(ep_utils, 'split_inputs_by_attn_tp', lambda hidden, weights, ids: (hidden, weights, ids, None))
    monkeypatch.setattr(ep_utils, 'gather_outputs_by_attn_tp', lambda out_states, split_size: out_states)

    impl = v4_fp4.TritonFusedMoEV4FP4EPImpl.__new__(v4_fp4.TritonFusedMoEV4FP4EPImpl)
    impl.fusedmoe_build = fake_fusedmoe_build

    hidden_states = torch.empty(2, 4)
    topk_weights = torch.empty(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.long)
    out = v4_fp4.TritonFusedMoEV4FP4EPImpl.forward(
        impl,
        hidden_states,
        topk_weights,
        topk_ids,
        gate_up_weight=None,
        gate_up_scale=None,
        down_weight=None,
        down_scale=None,
    )

    assert out is hidden_states
    assert modes == [True]


def test_v4_fp4_layer_passes_build_context_deepep_token_limit(monkeypatch):
    from lmdeploy.pytorch.nn.moe import v4_fp4

    calls = []

    class FakeDistConfig:

        def get_tp_by_layer(self, layer_type):
            return 1, object()

    class FakeDistContext:
        dist_config = FakeDistConfig()
        moe_tp_group = type('FakeGroup', (), {'gpu_group': object()})()
        ep_gpu_group = object()

    class FakeDistManager:

        def current_context(self):
            return FakeDistContext()

    class FakeBuilder:

        @staticmethod
        def build(**kwargs):
            calls.append(kwargs)
            return object()

    class FakeBackend:

        def get_layer_impl_builder(self, op_type):
            return FakeBuilder

    class FakeBuildContext:
        deep_ep_max_tokens_per_rank = 384

    monkeypatch.setattr(v4_fp4, 'get_dist_manager', lambda: FakeDistManager())
    monkeypatch.setattr(v4_fp4, 'get_ep_world_rank', lambda: (2, 1))
    monkeypatch.setattr(v4_fp4, 'get_tp_world_rank', lambda *args, **kwargs: (1, 0))
    monkeypatch.setattr(v4_fp4, 'get_backend', lambda: FakeBackend())
    monkeypatch.setattr(v4_fp4, 'get_build_model_context', lambda: FakeBuildContext())

    layer = v4_fp4.FusedMoEV4FP4(hidden_dim=16,
                                 ffn_dim=32,
                                 num_experts=4,
                                 top_k=2,
                                 device=torch.device('cpu'))

    assert layer.impl is not None
    assert calls[0]['num_max_dispatch_tokens_per_rank'] == 384


def test_blocked_fp8_async_prefill_passes_weight_dtype_and_scale_fmt():
    from lmdeploy.pytorch.nn.moe.base import MoeType
    from lmdeploy.pytorch.nn.moe.blocked_fp8 import FusedMoEBlockedF8

    class FakeWeight:
        dtype = torch.float8_e5m2

    class FakeGateUp:
        weight = FakeWeight()

    class FakeFusedMoE:

        def __init__(self):
            self.quant_args = None

        def per_token_group_quant_fp8(self, hidden_states, dtype=None, scale_fmt=None):
            self.quant_args = (hidden_states, dtype, scale_fmt)
            return ('quant', 'scale')

        def capture(self):
            return 'event'

    layer = FusedMoEBlockedF8.__new__(FusedMoEBlockedF8)
    layer.scale_fmt = 'ue8m0'
    layer.gate_up = FakeGateUp()
    fusedmoe = FakeFusedMoE()
    layer.fusedmoe_build = lambda low_latency_mode=False: fusedmoe
    hidden_states = object()
    state = {'moe_type': MoeType.DSAsyncPrefill, 'hidden_states': hidden_states}

    out_state = FusedMoEBlockedF8.before_dispatch(layer, state)

    assert fusedmoe.quant_args == (hidden_states, torch.float8_e5m2, 'ue8m0')
    assert out_state['hidden_states'] == ('quant', 'scale')
    assert out_state['previous_event'] == 'event'
