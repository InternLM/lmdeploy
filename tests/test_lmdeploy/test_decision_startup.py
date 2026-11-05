# Copyright (c) OpenMMLab. All rights reserved.
"""Server task selection and initialization without loading GPU weights."""
import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole


@pytest.mark.parametrize('task', ['intern-decision', 'Intern-Decision-4B', 'unknown'])
def test_cli_task_values(monkeypatch, task):
    from lmdeploy.cli.serve import SubCliServe

    parser = argparse.ArgumentParser()
    monkeypatch.setattr(SubCliServe, 'subparsers', parser.add_subparsers())
    SubCliServe.add_parser_api_server()
    args = ['api_server', '/models/checkpoint', '--task', task]
    if task == 'intern-decision':
        assert parser.parse_args(args).task == task
    else:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(args)
        assert exc.value.code == 2


@pytest.fixture
def server_startup(monkeypatch):
    from test_decision_service import FakeEngine

    from lmdeploy.model import BaseChatTemplate
    from lmdeploy.serve.openai import api_server

    tokenizer = SimpleNamespace(
        chat_template='checkpoint template',
        get_added_vocab=lambda: {'<decision>': 900},
        get_vocab=lambda: {},
        convert_tokens_to_ids=lambda token: {'<decision>': 900}[token],
        encode=lambda text, **kwargs: [900 if text == '<decision>' else 32 + ANSWER_SYMBOLS.index(text)],
    )
    engine = FakeEngine()
    engine.chat_template = BaseChatTemplate()
    engine.backend = 'pytorch'
    engine.backend_config = PytorchEngineConfig()
    engine.speculative_config = None
    engine.tokenizer.model = SimpleNamespace(model=tokenizer)
    engine.hf_cfg = SimpleNamespace(vocab_size=1000)
    engine.close = Mock()
    engine.is_sleeping = False
    build, run = Mock(return_value=engine), Mock()
    monkeypatch.setattr(api_server, 'get_task', Mock(return_value=('llm', build)))
    monkeypatch.setattr(api_server, 'set_parsers', lambda *args: None)
    monkeypatch.setattr(api_server, 'create_lifespan_handler', lambda *args: None)
    monkeypatch.setattr(api_server.uvicorn, 'run', run)
    return api_server, build, run


@pytest.mark.parametrize('task', ['Intern-Decision-4B', 'unknown'])
def test_python_rejects_unknown_task_before_building_engine(server_startup, task):
    api_server, build, run = server_startup
    with pytest.raises(ValueError, match='Unknown task'):
        api_server.serve('/unused', task=task)
    build.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('logprobs_mode', [None, 'raw_logits'])
def test_startup_publishes_context_only_when_enabled(
        server_startup, monkeypatch, tmp_path, enabled, logprobs_mode):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    (tmp_path / 'inference.py').write_text('DEFAULT_TEMPERATURE = 2.1')
    api_server.serve(str(tmp_path), backend='pytorch', task='intern-decision' if enabled else None,
                     backend_config=PytorchEngineConfig(logprobs_mode=logprobs_mode),
                     generation_config='lmdeploy')
    with TestClient(run.call_args.kwargs['app']) as client:
        response = client.post('/v1/systemone', json={
            'state': 'test', 'questions': {'q': {'type': 'noul', 'instructions': 'Check the state.'}},
        })
        openai_response = client.post('/v1/decisions', json={
            'model': build.return_value.model_name, 'input': 'test',
            'questions': [{'type': 'predicate', 'instructions': 'Check the state.'}],
        })
    assert response.status_code == (200 if enabled else 503)
    assert openai_response.status_code == (200 if enabled else 503)
    if enabled:
        assert context.decision_context.temperature == 2.1
    build.assert_called_once()


def test_invalid_checkpoint_closes_engine(server_startup, tmp_path):
    api_server, build, run = server_startup
    engine = build.return_value
    engine.hf_cfg.vocab_size = 10
    (tmp_path / 'inference.py').write_text('DEFAULT_TEMPERATURE = 2.1')
    with pytest.raises(ValueError, match='exceed the model vocabulary'):
        api_server.serve(str(tmp_path), task='intern-decision', generation_config='lmdeploy')
    engine.close.assert_called_once()
    run.assert_not_called()


@pytest.mark.parametrize('size, temperature', [
    ('0.8B', 2.747760550703), ('2B', 2.100509348278), ('4B', 1.99241824),
])
@pytest.mark.parametrize('model_name', [None, 'custom-name'])
@pytest.mark.parametrize('has_metadata', [False, True])
def test_remote_checkpoint_uses_resolved_calibration(
        server_startup, monkeypatch, tmp_path, size, temperature, model_name, has_metadata):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / 'cached-snapshot'
    checkpoint.mkdir()
    if has_metadata:
        (checkpoint / 'inference.py').write_text('DEFAULT_TEMPERATURE = 0.75')
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(api_server, 'get_model', download)
    model_id = f'internlm/Intern-Decision-{size}'
    backend_config = PytorchEngineConfig(download_dir=str(tmp_path / 'cache'), revision='pinned-revision')

    api_server.serve(model_id, model_name=model_name, backend_config=backend_config,
                     task='intern-decision', generation_config='lmdeploy')

    download.assert_called_once_with(model_id, backend_config.download_dir, 'pinned-revision')
    assert build.call_args.kwargs['model_path'] == str(checkpoint)
    assert build.call_args.kwargs['model_name'] == (model_name or model_id)
    assert context.decision_context.temperature == (0.75 if has_metadata else temperature)
    run.assert_called_once()


def test_missing_calibration_fails_before_building_engine(server_startup, monkeypatch, tmp_path):
    api_server, build, run = server_startup
    download = Mock(side_effect=AssertionError('Local checkpoints must not be downloaded'))
    monkeypatch.setattr(api_server, 'get_model', download)
    with pytest.raises(ValueError, match='Cannot read decision temperature'):
        api_server.serve(str(tmp_path), model_name='Intern-Decision-4B',
                         task='intern-decision', generation_config='lmdeploy')
    download.assert_not_called()
    build.assert_not_called()
    api_server.get_task.assert_not_called()
    build.return_value.close.assert_not_called()
    run.assert_not_called()


@pytest.fixture(params=['turbomind', 'speculative', 'prefill', 'decode', 'raw_logprobs'])
def unsupported_decision_config(request):
    if request.param == 'turbomind':
        return TurbomindEngineConfig(), None, 'requires the PyTorch backend'
    if request.param == 'speculative':
        return PytorchEngineConfig(), SimpleNamespace(), 'does not support speculative decoding'
    if request.param == 'raw_logprobs':
        return PytorchEngineConfig(logprobs_mode='raw_logprobs'), None, 'requires logprobs_mode'
    role = EngineRole.Prefill if request.param == 'prefill' else EngineRole.Decode
    return PytorchEngineConfig(role=role), None, 'requires the hybrid engine role'


def test_unsupported_config_fails_before_loading_resources(
        server_startup, monkeypatch, unsupported_decision_config):
    api_server, build, run = server_startup
    config, speculative, message = unsupported_decision_config
    download, generation_config = Mock(), Mock()
    monkeypatch.setattr(api_server, 'get_model', download)
    monkeypatch.setattr(api_server, 'resolve_default_gen_config', generation_config)
    with pytest.raises(ValueError, match=message):
        api_server.serve('remote/checkpoint', backend_config=config, task='intern-decision',
                         speculative_config=speculative)
    download.assert_not_called()
    generation_config.assert_not_called()
    api_server.get_task.assert_not_called()
    build.assert_not_called()
    run.assert_not_called()


def test_dp_unsupported_config_fails_before_starting_workers(monkeypatch, unsupported_decision_config):
    import importlib

    launcher = importlib.import_module('lmdeploy.serve.openai.launch_server')
    config, speculative, message = unsupported_decision_config
    process, start_method, host_ip = Mock(), Mock(), Mock()
    monkeypatch.setattr(launcher.mp, 'Process', process)
    monkeypatch.setattr(launcher.mp, 'set_start_method', start_method)
    monkeypatch.setattr(launcher, 'get_host_ip', host_ip)
    with pytest.raises(ValueError, match=message):
        launcher.launch_server(1, 0, 'remote/checkpoint', config,
                               task='intern-decision', speculative_config=speculative)
    process.assert_not_called()
    start_method.assert_not_called()
    host_ip.assert_not_called()


def test_raw_logprobs_is_rejected_by_direct_context_creation():
    from lmdeploy.serve.parsers.decision_parser import create_decision_context

    engine = SimpleNamespace(backend='pytorch', backend_config=PytorchEngineConfig(logprobs_mode='raw_logprobs'),
                             speculative_config=None)
    with pytest.raises(ValueError, match='requires logprobs_mode'):
        create_decision_context(engine, '/unused', temperature=1.0)


def test_ordinary_server_keeps_raw_logprobs(server_startup, tmp_path):
    api_server, build, run = server_startup
    config = PytorchEngineConfig(logprobs_mode='raw_logprobs')
    api_server.serve(str(tmp_path), backend='pytorch', backend_config=config, generation_config='lmdeploy')
    assert build.call_args.kwargs['backend_config'] is config
    assert build.call_args.kwargs['backend_config'].logprobs_mode == 'raw_logprobs'
    run.assert_called_once()


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf'), True])
def test_dp_invalid_temperature_fails_before_starting_workers(monkeypatch, temperature):
    import importlib

    launcher = importlib.import_module('lmdeploy.serve.openai.launch_server')
    process = Mock()
    monkeypatch.setattr(launcher.mp, 'Process', process)
    monkeypatch.setattr(launcher.mp, 'set_start_method', Mock())
    with pytest.raises(ValueError, match='temperature|Temperature'):
        launcher.launch_server(1, 0, 'remote/checkpoint', PytorchEngineConfig(),
                               task='intern-decision', decision_temperature=temperature)
    process.assert_not_called()


def test_calibration_is_resolved_before_engine_initialization(server_startup, monkeypatch, tmp_path):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    metadata = tmp_path / 'inference.py'
    metadata.write_text('DEFAULT_TEMPERATURE = 0.75')
    engine = build.return_value

    def build_engine(**kwargs):
        metadata.write_text('DEFAULT_TEMPERATURE = 99')
        return engine

    build.side_effect = build_engine
    api_server.serve(str(tmp_path), task='intern-decision', generation_config='lmdeploy')
    assert context.decision_context.temperature == 0.75
    run.assert_called_once()


def test_startup_explicit_calibration_without_metadata(server_startup, monkeypatch, tmp_path):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    api_server.serve(str(tmp_path), task='intern-decision', decision_temperature=0.75,
                     generation_config='lmdeploy')
    assert context.decision_context.temperature == 0.75
    assert 'decision_temperature' not in build.call_args.kwargs
    run.assert_called_once()


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf'), True])
def test_invalid_explicit_calibration_fails_before_engine(server_startup, temperature):
    api_server, build, run = server_startup
    with pytest.raises(ValueError, match='temperature|Temperature'):
        api_server.serve('/unused', task='intern-decision', decision_temperature=temperature)
    build.assert_not_called()
    run.assert_not_called()


def test_calibration_override_requires_decision_task(server_startup):
    api_server, build, run = server_startup
    with pytest.raises(ValueError, match='requires task'):
        api_server.serve('/unused', decision_temperature=1.0)
    build.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize('dp', [1, 2])
def test_cli_forwards_decision_temperature(monkeypatch, dp):
    import importlib

    from lmdeploy.cli.serve import SubCliServe

    parser = argparse.ArgumentParser()
    monkeypatch.setattr(SubCliServe, 'subparsers', parser.add_subparsers())
    SubCliServe.add_parser_api_server()
    args = parser.parse_args(['api_server', '/models/checkpoint', '--task', 'intern-decision',
                              '--decision-temperature', '0.75', '--dp', str(dp), '--max-batch-size', '2'])
    target = 'api_server' if dp == 1 else 'launch_server'
    module = importlib.import_module(f'lmdeploy.serve.openai.{target}')
    launch = Mock()
    monkeypatch.setattr(module, 'serve' if dp == 1 else 'launch_server', launch)
    SubCliServe.api_server(args)
    assert launch.call_args.kwargs['decision_temperature'] == 0.75
    assert launch.call_args.kwargs['task'] == 'intern-decision'
