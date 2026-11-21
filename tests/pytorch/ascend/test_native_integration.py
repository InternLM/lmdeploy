# Copyright (c) OpenMMLab. All rights reserved.
"""CPU contract tests; NPU kernels and collectives are tested separately on
A3."""

import ast
import asyncio
import copy
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.ascend import graph_runner
from lmdeploy.pytorch.backends.ascend.cache import AscendCacheBackend
from lmdeploy.pytorch.backends.ascend.model_config import load_modelslim_config
from lmdeploy.pytorch.backends.ascend.modelslim import map_modelslim_param_name
from lmdeploy.pytorch.backends.ascend.op_backend import _get_device_kv_start_indices
from lmdeploy.pytorch.backends.ascend.rejection_sampling import AscendRejectionSamplingImpl
from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin

ROOT = Path(__file__).resolve().parents[3]


def _context(device):
    return get_device_manager().context(DeviceContext(device))


def test_native_selection_and_cuda_import_are_isolated():
    # A fresh interpreter catches accidental transitive imports and import-time patches.
    code = """
import importlib.abc, sys, torch
import lmdeploy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'dlinfer', 'torch_npu', 'triton'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin
before = dict(CudaGraphMixin.__dict__)
compile_before = torch.compile
from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.ascend import graph_runner, model_config, modelslim
assert get_backend('ascend').__module__ == 'lmdeploy.pytorch.backends.ascend.op_backend'
assert get_backend('cuda').__module__ == 'lmdeploy.pytorch.backends.cuda.op_backend'
assert dict(CudaGraphMixin.__dict__) == before
assert torch.compile is compile_before
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_no_dlinfer_import_in_native_packages():
    for relative in ('lmdeploy/pytorch/backends/ascend', 'lmdeploy/pytorch/kernels/ascend'):
        for path in (ROOT / relative).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    assert all(not name.name.startswith('dlinfer') for name in node.names), path
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith('dlinfer'), path


def test_front_door_initializes_native_runtime(monkeypatch):
    from lmdeploy.pytorch.backends.ascend import runtime
    from lmdeploy.utils import try_import_deeplink

    calls = []
    monkeypatch.setattr(runtime, 'initialize', lambda: calls.append('native'))
    try_import_deeplink('ascend')
    try_import_deeplink('npu')
    try_import_deeplink('cuda')
    assert calls == ['native', 'native']


@pytest.mark.parametrize('api', ['auto_awq', 'smooth_quant'])
@pytest.mark.parametrize('device', ['npu', 'ascend'])
def test_lite_device_aliases_initialize_and_pass_torch_device(monkeypatch, tmp_path, api, device):
    from lmdeploy.pytorch.backends.ascend import runtime

    module = importlib.import_module(f'lmdeploy.lite.apis.{api}')
    calls = []

    class ReachedCalibration(Exception):
        pass

    def calibrate(model, dataset, samples, seqlen, workdir, torch_device, **kwargs):
        assert calls == ['initialized']
        assert torch_device == 'npu'
        raise ReachedCalibration

    monkeypatch.setattr(runtime, 'initialize', lambda: calls.append('initialized'))
    monkeypatch.setattr(module, 'calibrate', calibrate)
    with pytest.raises(ReachedCalibration):
        getattr(module, api)(str(tmp_path), device=device)


def test_device_name_keeps_original_error_and_can_retry(monkeypatch):
    from lmdeploy.pytorch.backends.ascend.op_backend import SocVersion

    class TestSocVersion(SocVersion):
        pass

    original_error = RuntimeError('driver unavailable')
    responses = iter([original_error, 'Ascend910'])

    def get_device_name():
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(torch, 'npu', SimpleNamespace(get_device_name=get_device_name), raising=False)
    with pytest.raises(RuntimeError, match='driver unavailable') as exc_info:
        TestSocVersion.is_Ascend910()
    assert exc_info.value is original_error
    assert TestSocVersion.is_Ascend910()
    assert not TestSocVersion.is_Ascend310P()


def test_ascend_sampling_excludes_banned_tokens_at_low_temperature():
    from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs

    words = torch.tensor([[1, -1], [0, -1], [-1, -1], [1, 1]])
    scores = torch.tensor([[0., 2.], [2., 0.], [2., 0.], [0., 2.]])
    sampling_inputs = SamplingInputs(temperature=torch.full((4, ), 1e-5),
                                     bad_words=words,
                                     bad_mask=words >= 0,
                                     max_num_logprobs=-1,
                                     logits_processors=[[]] * 4,
                                     max_top_k=1)
    with _context('ascend'):
        processor = FusedLogitsProcessor(sampling_inputs)
        logits, _ = asyncio.run(processor(scores))
        assert processor.sampling(logits).tolist() == [0, 1, 0, 0]


def test_validation_cli_rejects_ascend_graph_with_int8_kv(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / 'tools/ascend/validate_pipeline.py'),
        '--device', 'ascend', '--quant-policy', '8', '--model', str(tmp_path / 'missing-model'),
        '--output', str(tmp_path / 'output.json'),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Ascend INT8 KV cache validation requires --eager' in result.stderr
    assert 'Traceback' not in result.stderr


def test_engine_platform_name_is_ascend():
    from lmdeploy.pytorch.devices import is_ascend

    with _context('ascend'):
        assert is_ascend()
    with _context('npu'):
        assert not is_ascend()
        with pytest.raises(RuntimeError, match='Unsupported device type: npu'):
            get_backend()


def test_shared_vision_compile_policy_is_local_to_cuda(monkeypatch):
    from lmdeploy.pytorch.backends.default.compile import compile_for_cuda

    compiled_calls = []

    def compile_impl(func, **options):
        compiled_calls.append(options)
        return lambda x: func(x) + 10

    monkeypatch.setattr(torch, 'compile', compile_impl)

    @compile_for_cuda(dynamic=True)
    def helper(x):
        return x + 1

    with _context('ascend'):
        assert helper(2) == 3
        assert compiled_calls == []
    with _context('cuda'):
        assert helper(2) == helper(2) == 13
        assert compiled_calls == [{'dynamic': True}]
    with _context('ascend'):
        assert helper(2) == 3
    assert torch.compile is compile_impl


@pytest.mark.parametrize('num_spec_tokens', [0, 1, 3])
def test_qwen35_state_layout_and_draft(num_spec_tokens):
    text = SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=256,
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=1,
    )
    hf = SimpleNamespace(text_config=text, model_type='qwen3_5', architectures=['Qwen3_5ForConditionalGeneration'])
    cfg = Qwen3_5ModelConfigBuilder.build(
        copy.deepcopy(hf), tp=2, device_type='ascend', num_spec_tokens=num_spec_tokens
    )
    assert cfg.num_layers == 1
    conv = (4 + num_spec_tokens, 96)
    state = (4, 8, 16) if num_spec_tokens == 0 else (1 + num_spec_tokens, 4, 8, 16)
    assert cfg.states_shapes == [(conv, torch.bfloat16)] * 2 + [(state, torch.float32)] * 2
    assert not cfg.state_cache_is_indexed and not cfg.state_cache_specs
    draft = Qwen3_5ModelConfigBuilder.build(
        copy.deepcopy(hf),
        tp=2,
        device_type='ascend',
        is_draft_model=True,
        spec_method='qwen3_5_mtp',
        num_spec_tokens=num_spec_tokens,
    )
    assert draft.states_shapes == [] and draft.num_layers == 1


@pytest.mark.parametrize(
    ('kind', 'expected'), [('FLOAT', None), ('W8A8_DYNAMIC', 'smooth_quant'), ('W8A8', 'modelslim_w8a8_static')]
)
def test_modelslim_metadata_dispatch(tmp_path, kind, expected):
    import json

    description = {'model.layers.0.q_proj.weight': kind}
    (tmp_path / 'quant_model_description.json').write_text(json.dumps(description))
    hf = SimpleNamespace()
    load_modelslim_config(hf, tmp_path)
    cfg = QuantizationConfig.from_config(hf)
    assert cfg.quant_dtype == torch.int8
    assert cfg.get_quant_method('model.layers.0.q_proj') == expected
    assert cfg.get_quant_method('model.layers.0.norm', 'norm') is None
    assert cfg.get_quant_method('unlisted') is None


def test_modelslim_rejects_inconsistent_fused_projections():
    cfg = QuantizationConfig.from_config(
        SimpleNamespace(
            quantization_config=dict(
                quant_method='modelslim',
                quant_description={'mlp.gate_proj.weight': 'W8A8', 'mlp.up_proj.weight': 'FLOAT'},
            )
        )
    )
    with pytest.raises(ValueError, match='mixes quant types'):
        cfg.get_quant_method('mlp.gate_up_proj')


def test_modelslim_auxiliary_name_mapping():
    cfg = SimpleNamespace(quantization_config={'quant_method': 'modelslim'})
    params = {'a.scale': torch.empty(3), 'a.input_scale': torch.empty(1)}
    assert map_modelslim_param_name(cfg, 'a.weight_scale', params) == 'a.scale'
    assert map_modelslim_param_name(cfg, 'a.weight_offset', params) is None
    assert map_modelslim_param_name(cfg, 'a.input_scale', params) == 'a.input_scale'
    assert map_modelslim_param_name(SimpleNamespace(), 'a.weight_scale', params) == 'a.weight_scale'


def test_paged_slot_mapping_variable_query_lengths():
    q = torch.tensor([2, 1, 3], dtype=torch.int32)
    kv = torch.tensor([5, 9, 6], dtype=torch.int32)
    cu = torch.tensor([0, 2, 3, 6], dtype=torch.int32)
    table = torch.tensor([[2, 4, 8], [5, 6, 7], [9, 3, 1]], dtype=torch.int32)
    actual = _get_device_kv_start_indices(q, kv, cu, table, 4, 6)
    assert actual.tolist() == [11, 16, 28, 39, 12, 13]


def test_mtp_graph_subclass_buffers_survive_native_dispatch(monkeypatch):
    class Model(CudaGraphMixin):
        def make_buffers_cudagraph(self, graph_meta, **kwargs):
            buffers = super().make_buffers_cudagraph(graph_meta, **kwargs)
            buffers['target_hidden_states'] = torch.zeros(1, graph_meta.max_tokens, 8)
            return buffers

        def fill_buffers_cudagraph(self, graph_meta, input_ids, **kwargs):
            result = super().fill_buffers_cudagraph(graph_meta, input_ids, **kwargs)
            buffer = graph_meta.input_buffers['target_hidden_states']
            buffer[:, : input_ids.numel()] = kwargs['target_hidden_states']
            result['target_hidden_states'] = buffer
            return result

    model = Model()
    meta = SimpleNamespace(
        max_batchs=4,
        max_tokens=4,
        num_blocks=3,
        device='cpu',
        is_ssm=False,
        use_mrope=False,
        mla_index_topk=2048,
        vocab_size=32,
    )
    moe = SimpleNamespace(x_active_mask=None)
    ctx = SimpleNamespace(moe_metadata=moe)
    monkeypatch.setattr(graph_runner, 'get_step_ctx_manager', lambda: SimpleNamespace(current_context=lambda: ctx))
    attn = SimpleNamespace(
        block_offsets=torch.ones(2, 3, dtype=torch.int32),
        kv_seqlens=torch.tensor([4, 5]),
        kv_start_indices=torch.tensor([3, 4]),
        cache_seqlens=None,
        is_multi_token_decoding=False,
    )
    args = dict(
        input_ids=torch.tensor([[1, 2]]),
        position_ids=torch.tensor([[3, 4]]),
        past_key_values=[],
        attn_metadata=attn,
        inputs_embeds=None,
        target_hidden_states=torch.ones(1, 2, 8),
    )
    with _context('ascend'):
        meta.input_buffers = model.make_buffers_cudagraph(meta, **args)
        result = model.fill_buffers_cudagraph(meta, **args)
        model.update_context_cudagraph(meta, ctx)
    assert result['target_hidden_states'].shape == (1, 4, 8)
    assert torch.equal(result['target_hidden_states'][:, :2], args['target_hidden_states'])
    assert ctx.q_seqlens.tolist() == [1, 1, 1, 1]
    assert ctx.kv_seqlens.tolist() == [4, 5, 0, 0]


@pytest.mark.parametrize('device', ['cuda', 'ascend'])
def test_awq_zero_layout_and_shard_loading(device):
    from lmdeploy.pytorch.nn.linear.awq import AwqLinear, MergedAwqLinear, QKVAwqLinear

    with _context(device):
        _, _, zeros, _ = AwqLinear.create_weights(None, 16, 32, 4, 8, False, torch.float16, 'cpu')
    assert zeros.shape == ((2, 32) if device == 'ascend' else (2, 4))
    assert zeros.dtype == (torch.float16 if device == 'ascend' else torch.int32)
    for cls in (MergedAwqLinear, QKVAwqLinear):
        layer = SimpleNamespace(
            get_tp_world_rank=lambda: (2, 1),
            out_names_map={'q': 0},
            all_out_features=[16],
            elem_per_int=8,
            group_size=8,
            num_replicate_kv_heads=1,
        )
        param = torch.nn.Parameter(zeros[:, : 16 if device == 'ascend' else 2].clone(), requires_grad=False)
        param._weight_type = 'qzeros'
        loaded = torch.arange(zeros.numel(), dtype=zeros.dtype).reshape(zeros.shape)
        cls.weight_loader(layer, param, loaded, 'q')
        torch.testing.assert_close(param, loaded.chunk(2, dim=-1)[1])


def test_rejection_greedy_and_mixed_torch_path(monkeypatch):
    monkeypatch.setenv('DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON', '0')
    impl = AscendRejectionSamplingImpl()
    draft = torch.tensor([[1, 3], [2, 3]])
    targets = torch.tensor([[1, 2, 4], [2, 3, 4]])
    output, rejected, last = impl.forward_greedy(targets, draft)
    assert output.tolist() == [[1, 2, -1], [2, 3, 4]]
    assert rejected.tolist() == [1, 0] and last.tolist() == [2, 4]
    logits = torch.full((2, 2, 5), -1000.0)
    logits.scatter_(-1, targets[:, :2, None], 1000.0)
    output, rejected, last = impl.forward(logits, draft, targets[:, -1], is_greedy=torch.tensor([True, False]))
    assert output.tolist() == [[1, 2, -1], [2, 3, 4]]
    assert rejected.tolist() == [1, 0] and last.tolist() == [2, 4]


def test_ascend_cache_uses_contiguous_layout():
    assert get_backend('ascend').get_cache_backend() is AscendCacheBackend
    from lmdeploy.pytorch.engine.cache_engine.layout import ContiguousStateCacheLayout

    assert isinstance(AscendCacheBackend.build_state_layout(()), ContiguousStateCacheLayout)


def test_native_cache_allocator_handles_small_mixed_dtype_pools():
    # Replaces dlinfer's removed CacheEngine monkey-patch test using the
    # current allocation interface. The smallest cache is below alignment.
    from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

    specs = tuple(
        CacheTensorSpec(name, CacheDesc(shape=[width], dtype=dtype))
        for name, width, dtype in (
            ('key', 128, torch.bfloat16),
            ('value', 128, torch.bfloat16),
            ('indexer', 128, torch.bfloat16),
            ('scale', 1, torch.float32),
        )
    )
    layout = AscendCacheBackend.build_block_layout(specs, num_layers=2)
    allocation = layout.allocate(num_blocks=3, device='cpu')
    assert [tuple(t.shape) for t in allocation.tensor_views] == [(2, 3, 128)] * 3 + [(2, 3, 1)]
    assert [t.dtype for t in allocation.tensor_views] == [torch.bfloat16] * 3 + [torch.float32]
    for index, tensor in enumerate(allocation.tensor_views):
        tensor.fill_(index + 1)
    for index, tensor in enumerate(allocation.tensor_views):
        assert torch.all(tensor == index + 1)


def test_deepseek_v32_rejects_ascend_before_building_model_config():
    from lmdeploy.pytorch.configurations.deepseek_v32 import DeepseekV32ModelConfigBuilder

    with pytest.raises(ValueError, match='DeepSeek-V3.2 is not supported on Ascend.*DSA indexer'):
        DeepseekV32ModelConfigBuilder.build(SimpleNamespace(model_type='deepseek_v32'), device_type='ascend')


def test_glm_config_uses_native_sparse_cache_without_flashmla(monkeypatch, tmp_path):
    from lmdeploy.pytorch.configurations import deepseek_v2
    from lmdeploy.pytorch.configurations.glm_moe_dsa import GlmMoeDsaModelConfigBuilder

    monkeypatch.setattr(deepseek_v2, 'flash_mla_available', lambda: False)
    hf = SimpleNamespace(
        model_type='glm_moe_dsa',
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        qk_head_dim=32,
        vocab_size=128,
        index_topk=4,
        eos_token_id=2,
        architectures=['GlmMoeDsaForCausalLM'],
    )
    (tmp_path / 'quant_model_description.json').write_text('{"layer.weight": "W8A8"}')
    result = GlmMoeDsaModelConfigBuilder.build(copy.deepcopy(hf), model_path=str(tmp_path), device_type='ascend')
    assert (result.k_head_dim, result.v_head_dim, result.split_mla_kv_cache) == (16, 32, True)
    assert not result.use_flash_mla and result.mla_index_topk == 4
    assert deepseek_v2.flash_mla_available() is False
    with pytest.raises(AssertionError, match='flash_mla'):
        GlmMoeDsaModelConfigBuilder.build(copy.deepcopy(hf), device_type='cuda')


def test_modelslim_attention_scales_follow_rope_permutation():
    from lmdeploy.pytorch.models.deepseek_v2 import DeepseekV2ForCausalLM

    model = object.__new__(DeepseekV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(quantization_config={'quant_method': 'modelslim'})
    scale = torch.nn.Parameter(torch.zeros(8))
    input_scale = torch.nn.Parameter(torch.zeros(1))
    params = {'layer.self_attn.q_b_proj.scale': scale, 'layer.self_attn.q_b_proj.input_scale': input_scale}
    mapping = [('q_b_proj', 4, 0)]
    model._load_weight_attention('layer.self_attn.q_b_proj.weight_scale', torch.arange(8.0), params, mapping)
    assert scale.tolist() == [0, 2, 1, 3, 4, 6, 5, 7]
    model._load_weight_attention('layer.self_attn.q_b_proj.input_scale', torch.tensor([0.25]), params, mapping)
    assert input_scale.item() == 0.25
    # Offsets belong to the checkpoint, not the runtime parameter set.
    model._load_weight_attention('layer.self_attn.q_b_proj.weight_offset', torch.zeros(8), params, mapping)


@pytest.mark.parametrize('device', ['cuda', 'ascend'])
def test_glm_indexer_component_selection_does_not_mutate_env(monkeypatch, device):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.models import glm_moe_dsa

    monkeypatch.setattr(envs, 'disable_dsa_indexer_fusion', False)
    monkeypatch.setattr(
        glm_moe_dsa,
        'build_colwise_linear',
        lambda in_features, out_features, **kwargs: torch.nn.Linear(in_features, out_features, bias=False),
    )
    monkeypatch.setattr(glm_moe_dsa, 'ApplyRotaryEmb', lambda: lambda q, k, *args, **kwargs: (q, k))
    monkeypatch.setattr(glm_moe_dsa, 'IndexerTopKFP8', lambda *args, **kwargs: lambda q, k, w, **kw: (q, k, w))
    cfg = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        qk_rope_head_dim=2,
        index_topk=2,
        q_lora_rank=4,
        num_hidden_layers=1,
        indexer_rope_interleave=True,
    )
    with _context(device):
        indexer = glm_moe_dsa.GlmMoeDsaIndexer(cfg, 0, dtype=torch.float32, device='cpu')
    assert indexer.use_fusion == (device == 'cuda')
    assert envs.disable_dsa_indexer_fusion is False
    if device == 'ascend':
        # The native forward must not call fast_hadamard_transform or fused CUDA preprocessing.
        q, k, w = indexer(torch.randn(1, 3, 8), torch.randn(1, 3, 4), (torch.ones(3, 2), torch.zeros(3, 2)))
        assert q.shape == (3, 2, 4) and k.shape == (3, 4) and w.shape == (3, 2)


def test_native_operator_builds_without_external_registry(monkeypatch):
    from types import ModuleType

    # Only the vendor extension is a stand-in. LMDeploy's real backend/wrappers are imported.
    monkeypatch.setitem(sys.modules, 'torch_npu', ModuleType('torch_npu'))
    from lmdeploy.pytorch.backends.activation import SiluAndMulBuildSpec
    from lmdeploy.pytorch.backends.apply_rotary_emb import ApplyRotaryEmbBuildSpec
    from lmdeploy.pytorch.backends.linear import LinearBuildSpec
    from lmdeploy.pytorch.backends.moe import SoftmaxTopKBuildSpec
    from lmdeploy.pytorch.backends.norm import RMSNormBuildSpec
    from lmdeploy.pytorch.backends.rejection_sampling import RejectionSamplingBuildSpec

    backend = get_backend('ascend')
    specs = [
        SiluAndMulBuildSpec(False),
        ApplyRotaryEmbBuildSpec(),
        LinearBuildSpec(8, 8, False, torch.float32),
        RMSNormBuildSpec(8),
        RejectionSamplingBuildSpec(),
        SoftmaxTopKBuildSpec(2),
    ]
    for spec in specs:
        impl = backend.build_op(spec)
        assert type(impl).__module__.startswith('lmdeploy.pytorch.backends.ascend')
