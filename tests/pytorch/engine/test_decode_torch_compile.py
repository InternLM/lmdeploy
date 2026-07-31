# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock


def test_decode_torch_compile_returns_raw_model_when_disabled(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner

    model = object()
    monkeypatch.setattr(graph_runner, 'enable_decode_torch_compile', False)
    monkeypatch.setattr(graph_runner.torch, 'compile', lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_runner, '_configure_decode_torch_compile', lambda: None)

    assert graph_runner._build_decode_model_forward(model) is model


def test_prefill_does_not_build_decode_model_forward(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner

    raw_calls = []

    class Model:

        def __call__(self, **kwargs):
            raw_calls.append(kwargs)
            return 'raw_output'

        @staticmethod
        def make_output_buffers(output):
            return {'output': output}

    context = SimpleNamespace(global_is_decoding=lambda: False)
    runner = object.__new__(graph_runner.CUDAGraphRunner)
    runner.model = Model()
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: context)
    runner.backend_config = SimpleNamespace(device_type='cuda')
    runner.decode_model_forward = None
    runner._prepare_inputs = lambda **kwargs: kwargs
    runner.enable_graph = lambda **kwargs: True

    build = Mock(side_effect=AssertionError('prefill must not compile'))
    monkeypatch.setattr(graph_runner, 'get_deepep_state', lambda: SimpleNamespace(enabled=lambda: False))
    monkeypatch.setattr(graph_runner, '_build_decode_model_forward', build)

    assert runner(input_ids='prefill') == {'output': 'raw_output'}
    assert raw_calls == [{'input_ids': 'prefill'}]
    assert runner.decode_model_forward is None
    build.assert_not_called()


def test_decode_model_forward_is_built_lazily_once(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner

    model = object()
    compiled_model = object()
    build_calls = []
    captured_forwards = []

    class SingleGraphRunner:

        def __init__(self, model, *, model_forward, **kwargs):
            captured_forwards.append(model_forward)

        @staticmethod
        def capture(**kwargs):
            return 'captured_output'

    context = SimpleNamespace(global_is_decoding=lambda: True)
    runner = object.__new__(graph_runner.CUDAGraphRunner)
    runner.model = model
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: context)
    runner.backend_config = SimpleNamespace(device_type='cuda')
    runner.decode_model_forward = None
    runner._runner_map = {}
    runner._prepare_inputs = lambda **kwargs: kwargs
    runner.enable_graph = lambda **kwargs: True
    runner.get_graph_key = lambda **kwargs: (1, True, False, 1)
    runner._get_max_tokens = lambda *args: 1
    runner.num_blocks = 1
    runner.graph_pool_handle = object()
    runner.model_config = object()
    runner.device = 'cuda'

    def build(model_arg):
        build_calls.append(model_arg)
        return compiled_model

    monkeypatch.setattr(graph_runner, 'get_deepep_state', lambda: SimpleNamespace(enabled=lambda: False))
    monkeypatch.setattr(graph_runner, '_build_decode_model_forward', build)
    monkeypatch.setattr(graph_runner, 'CUDASingleGraphRunner', SingleGraphRunner)

    assert runner(input_ids='decode', attn_metadata=SimpleNamespace(q_seqlens=object())) == 'captured_output'
    assert runner._get_decode_model_forward() is compiled_model
    assert build_calls == [model]
    assert captured_forwards == [compiled_model]


def test_non_cuda_graph_runner_keeps_raw_model(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner

    model = object()
    runner = object.__new__(graph_runner.CUDAGraphRunner)
    runner.model = model
    runner.backend_config = SimpleNamespace(device_type='maca')
    runner.decode_model_forward = None

    build = Mock(side_effect=AssertionError('non-CUDA backend must not compile'))
    monkeypatch.setattr(graph_runner, '_build_decode_model_forward', build)

    assert runner._get_decode_model_forward() is model
    build.assert_not_called()


def test_decode_torch_compile_uses_non_fullgraph_mode(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner

    model = object()
    compiled_model = object()
    compile_call = {}
    compiler_configured = []

    def compile(model_arg, **kwargs):
        compile_call.update(model=model_arg, **kwargs)
        return compiled_model

    monkeypatch.setattr(graph_runner, 'enable_decode_torch_compile', True)
    monkeypatch.setattr(graph_runner.torch, 'compile', compile)
    monkeypatch.setattr(graph_runner, '_configure_decode_torch_compile',
                        lambda: compiler_configured.append(True))

    assert graph_runner._build_decode_model_forward(model) is compiled_model
    assert compiler_configured == [True]
    assert compile_call == {
        'model': model,
        'fullgraph': False,
        'dynamic': False,
        'options': {
            'emulate_divison_rounding': True,
            'emulate_precision_casts': True,
            'triton.cudagraphs': False,
        },
    }


def test_decode_torch_compile_configures_dynamo_once(monkeypatch):
    from torch._dynamo import config, trace_rules

    from lmdeploy.pytorch.backends.cuda import graph_runner

    registered_modules = []
    configure = graph_runner._configure_decode_torch_compile
    configure.cache_clear()
    monkeypatch.setattr(config, 'accumulated_cache_size_limit', 256)
    if hasattr(config, 'cache_size_limit'):
        monkeypatch.setattr(config, 'cache_size_limit', 8)
    monkeypatch.setattr(trace_rules, 'add', registered_modules.append)

    configure()
    configure()

    assert config.accumulated_cache_size_limit == 1024
    if hasattr(config, 'cache_size_limit'):
        assert config.cache_size_limit == 1024
    assert registered_modules == [
        'lmdeploy.pytorch.kernels',
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        'flash_attn_interface',
        'triton',
    ]
    configure.cache_clear()
