import os
import tempfile

from lmdeploy.serve.async_engine import get_names_from_model


def test_get_names_from_hf_model():
    cases = [
        # model repo_id from huggingface hub, model_name, chat_template_name
        ('InternLM/internlm2_5-7b-chat', 'internlm2.5-7b-chat', 'internlm2'),
        ('InternLM/internlm2_5-7b-chat', None, 'internlm2'),
    ]
    for model_path, model_name, chat_template in cases:
        _model_name, _chat_template = get_names_from_model(
            model_path=model_path, model_name=model_name)
        assert _chat_template == chat_template
        assert _model_name == model_name if model_name else model_path


def test_get_names_from_turbomind_model():
    workspace = tempfile.TemporaryDirectory('internlm2_5-7b-chat').name
    os.makedirs(os.path.join(workspace, 'triton_models', 'weights'),
                exist_ok=True)

    import yaml

    expected_chat_template = 'internlm2'
    config = dict(model_config=dict(chat_template=expected_chat_template))
    with open(f'{workspace}/triton_models/weights/config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    _, chat_template = get_names_from_model(workspace)
    assert chat_template == expected_chat_template
