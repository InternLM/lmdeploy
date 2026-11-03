# Copyright (c) OpenMMLab. All rights reserved.
"""Server task selection and initialization without loading GPU weights."""
import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS
from lmdeploy.messages import PytorchEngineConfig


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
def test_startup_publishes_context_only_when_enabled(server_startup, monkeypatch, tmp_path, enabled):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    (tmp_path / 'inference.py').write_text('DEFAULT_TEMPERATURE = 2.1')
    api_server.serve(str(tmp_path), backend='pytorch', task='intern-decision' if enabled else None,
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
    with pytest.raises(ValueError, match='exceed the model vocabulary'):
        api_server.serve(str(tmp_path), task='intern-decision', generation_config='lmdeploy')
    engine.close.assert_called_once()
    run.assert_not_called()


@pytest.mark.parametrize('size, temperature', [
    ('0.8B', 2.747760550703), ('2B', 2.100509348278), ('4B', 1.99241824),
])
@pytest.mark.parametrize('model_name', [None, 'custom-name'])
def test_remote_checkpoint_uses_resolved_calibration(
        server_startup, monkeypatch, tmp_path, size, temperature, model_name):
    api_server, build, run = server_startup
    context = api_server.ServerContext()
    monkeypatch.setattr(api_server, 'ServerContext', lambda: context)
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / 'cached-snapshot'
    checkpoint.mkdir()
    (checkpoint / 'inference.py').write_text(f'DEFAULT_TEMPERATURE = {temperature}')
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(api_server, 'get_model', download)
    model_id = f'internlm/Intern-Decision-{size}'
    backend_config = PytorchEngineConfig(download_dir=str(tmp_path / 'cache'), revision='pinned-revision')

    api_server.serve(model_id, model_name=model_name, backend_config=backend_config,
                     task='intern-decision', generation_config='lmdeploy')

    download.assert_called_once_with(model_id, backend_config.download_dir, 'pinned-revision')
    assert build.call_args.kwargs['model_path'] == str(checkpoint)
    assert build.call_args.kwargs['model_name'] == (model_name or model_id)
    assert context.decision_context.temperature == temperature
    run.assert_called_once()


def test_missing_calibration_closes_engine_without_starting_server(server_startup, monkeypatch, tmp_path):
    api_server, build, run = server_startup
    download = Mock(side_effect=AssertionError('Local checkpoints must not be downloaded'))
    monkeypatch.setattr(api_server, 'get_model', download)
    with pytest.raises(ValueError, match='Cannot read decision temperature'):
        api_server.serve(str(tmp_path), task='intern-decision', generation_config='lmdeploy')
    download.assert_not_called()
    build.return_value.close.assert_called_once()
    run.assert_not_called()
