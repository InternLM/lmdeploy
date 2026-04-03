import sys
from types import ModuleType, SimpleNamespace

from lmdeploy.cli import serve as serve_module
from lmdeploy.messages import TurbomindEngineConfig


def _make_api_server_args(**overrides):
    args = dict(model_path='QuantTrio/Qwen3.5-27B-AWQ',
                model_name='qwen35-awq',
                backend='turbomind',
                dtype='auto',
                tp=2,
                dp=1,
                ep=1,
                cp=1,
                nnodes=1,
                node_rank=0,
                dist_init_addr=None,
                max_batch_size=None,
                session_len=4096,
                model_format='awq',
                quant_policy=8,
                rope_scaling_factor=0.0,
                cache_max_entry_count=0.8,
                cache_block_seq_len=64,
                enable_prefix_caching=True,
                linear_prefix_cache_interval_blocks=4,
                max_prefill_token_num=8192,
                num_tokens_per_iter=0,
                max_prefill_iters=1,
                async_=1,
                communicator='nccl',
                disable_metrics=False,
                adapters=None,
                device='cuda',
                eager_mode=False,
                disable_vision_encoder=False,
                logprobs_mode='raw_logits',
                dllm_block_length=64,
                dllm_unmasking_strategy='low_confidence_dynamic',
                dllm_denoising_steps=0,
                dllm_confidence_threshold=0.0,
                enable_return_routed_experts=False,
                distributed_executor_backend=None,
                chat_template=None,
                vision_max_batch_size=1,
                server_name='127.0.0.1',
                server_port=23333,
                allow_origins=['*'],
                allow_credentials=False,
                allow_methods=['*'],
                allow_headers=['*'],
                allow_terminate_by_client=False,
                enable_abort_handling=False,
                log_level='info',
                api_keys=None,
                ssl=None,
                proxy_url=None,
                max_log_len=None,
                disable_fastapi_docs=False,
                max_concurrent_requests=None,
                reasoning_parser='qwen-qwq',
                tool_call_parser='qwen3coder',
                hf_overrides=None)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_api_server_turbomind_forwards_hybrid_prefix_cache_options(monkeypatch):
    captured = {}
    fake_api_server = ModuleType('lmdeploy.serve.openai.api_server')

    def fake_serve(model_path, **kwargs):
        captured['model_path'] = model_path
        captured.update(kwargs)

    fake_api_server.serve = fake_serve

    monkeypatch.setitem(sys.modules, 'lmdeploy.serve.openai.api_server', fake_api_server)
    monkeypatch.setattr('lmdeploy.archs.autoget_backend', lambda _: 'turbomind')
    monkeypatch.setattr(serve_module, 'get_max_batch_size', lambda device: 13)
    monkeypatch.setattr(serve_module, 'get_chat_template', lambda *_: None)
    monkeypatch.setattr(serve_module, 'get_speculative_config', lambda _: None)

    serve_module.SubCliServe.api_server(_make_api_server_args())

    assert captured['backend'] == 'turbomind'
    assert captured['model_path'] == 'QuantTrio/Qwen3.5-27B-AWQ'
    assert isinstance(captured['backend_config'], TurbomindEngineConfig)
    assert captured['backend_config'].enable_prefix_caching is True
    assert captured['backend_config'].linear_prefix_cache_interval_blocks == 4


def test_api_server_turbomind_uses_default_cuda_batch_size(monkeypatch):
    captured = {}
    fake_api_server = ModuleType('lmdeploy.serve.openai.api_server')

    def fake_serve(model_path, **kwargs):
        captured['model_path'] = model_path
        captured.update(kwargs)

    fake_api_server.serve = fake_serve

    monkeypatch.setitem(sys.modules, 'lmdeploy.serve.openai.api_server', fake_api_server)
    monkeypatch.setattr('lmdeploy.archs.autoget_backend', lambda _: 'turbomind')
    monkeypatch.setattr(serve_module, 'get_max_batch_size', lambda device: 7)
    monkeypatch.setattr(serve_module, 'get_chat_template', lambda *_: None)
    monkeypatch.setattr(serve_module, 'get_speculative_config', lambda _: None)

    serve_module.SubCliServe.api_server(_make_api_server_args(max_batch_size=None))

    assert captured['backend_config'].max_batch_size == 7
