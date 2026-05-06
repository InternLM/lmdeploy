# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import _turbomind as _tm
import pytest
from transformers import PretrainedConfig

from lmdeploy.turbomind.models.qwen3 import Qwen3TextModel


def _make_qwen3_stub():
    model = Qwen3TextModel.__new__(Qwen3TextModel)
    model.cfg = PretrainedConfig(hidden_size=4, vocab_size=8, tie_word_embeddings=False)
    model._contexts = []
    model._root_handles = []
    model._model_tp_ranks = []
    model._layer_prefix = 'model.layers'
    model._tie_embeddings = False
    model._get = Mock(side_effect=lambda key: f'tensor:{key}')
    model._linear = Mock(side_effect=lambda key: f'linear:{key}')
    model.norm = Mock(side_effect=lambda weight: f'norm:{weight}')
    model.layers = Mock(side_effect=lambda pfx: f'layers:{pfx}')
    return model


class _FakeRoot:

    last = None

    def __init__(self, *args, **kwargs):
        self.add_token_embeds = Mock()
        self.add_lm_head = Mock()
        self.norm = None
        self.layers = None
        self.build = Mock()
        _FakeRoot.last = self


def test_qwen3_model_uses_optional_checkpoint_prefix(monkeypatch):
    import lmdeploy.turbomind.models.qwen3 as qwen3_mod

    monkeypatch.setattr(qwen3_mod, 'TextModelBuilder', _FakeRoot)
    model = _make_qwen3_stub()

    model.model(pfx='language_model.')

    model._get.assert_any_call('language_model.model.embed_tokens.weight')
    model._get.assert_any_call('language_model.model.norm.weight')
    model._linear.assert_called_once_with('language_model.lm_head')
    model.layers.assert_called_once_with('language_model.model.layers')

    root = _FakeRoot.last
    root.add_token_embeds.assert_called_once_with(
        'tensor:language_model.model.embed_tokens.weight')
    root.add_lm_head.assert_called_once_with('linear:language_model.lm_head')
    root.build.assert_called_once_with()


def test_qwen3_model_default_prefix_preserves_plain_keys(monkeypatch):
    import lmdeploy.turbomind.models.qwen3 as qwen3_mod

    monkeypatch.setattr(qwen3_mod, 'TextModelBuilder', _FakeRoot)
    model = _make_qwen3_stub()

    model.model()

    model._get.assert_any_call('model.embed_tokens.weight')
    model._get.assert_any_call('model.norm.weight')
    model._linear.assert_called_once_with('lm_head')
    model.layers.assert_called_once_with('model.layers')


def _internvl_cfg(inner_arch='Qwen3ForCausalLM'):
    return PretrainedConfig(
        architectures=['InternVLChatModel'],
        llm_config=PretrainedConfig(
            architectures=[inner_arch],
            num_hidden_layers=1,
            vocab_size=8,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            model_type='qwen3',
            num_attention_heads=2,
            hidden_size=4,
            head_dim=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            intermediate_size=8,
            attention_bias=False,
        ),
    )


def test_internvl35_model_creates_qwen3_text_model():
    from lmdeploy.turbomind.models.internvl3_5 import InternVL3_5Model

    model = InternVL3_5Model(
        _internvl_cfg(),
        resolver=Mock(data_type=_tm.DataType.TYPE_FP16))

    assert isinstance(model.text_model, Qwen3TextModel)
    assert model.vision_model is None


def test_internvl35_model_delegates_runtime_params_and_export(monkeypatch):
    from lmdeploy.turbomind.models.internvl3_5 import InternVL3_5Model

    fake_text_model = Mock()
    fake_text_cls = Mock(return_value=fake_text_model)
    monkeypatch.setattr(
        'lmdeploy.turbomind.models.internvl3_5.Qwen3TextModel',
        fake_text_cls)

    resolver = Mock()
    model = InternVL3_5Model(_internvl_cfg(), resolver=resolver)

    assert fake_text_cls.call_args.args[0].architectures == ['Qwen3ForCausalLM']
    assert fake_text_cls.call_args.kwargs == {'resolver': resolver}

    attn_tp = Mock()
    mlp_tp = Mock()
    model_tp = Mock()
    model.bind_runtime(
        ctx='ctx',
        root_handles=['root'],
        attn_tp=attn_tp,
        mlp_tp=mlp_tp,
        model_tp=model_tp,
    )
    fake_text_model.bind_runtime.assert_called_once_with(
        ctx='ctx',
        root_handles=['root'],
        attn_tp=attn_tp,
        mlp_tp=mlp_tp,
        model_tp=model_tp,
    )

    params = {'language_model.lm_head.weight': object()}
    model.set_params(params)
    fake_text_model.set_params.assert_called_once_with(params)

    model.model()
    fake_text_model.model.assert_called_once_with(pfx='language_model.')

    fake_text_model.cfg.vocab_size = 32000
    assert model._vocab_size == 32000


def test_internvl35_model_requires_llm_config():
    from lmdeploy.turbomind.models.internvl3_5 import InternVL3_5Model

    cfg = {'architectures': ['InternVLChatModel']}

    with pytest.raises(ValueError, match='llm_config'):
        InternVL3_5Model(cfg, resolver=Mock())


def test_internvl35_model_requires_inner_architecture():
    from lmdeploy.turbomind.models.internvl3_5 import InternVL3_5Model

    cfg = {'architectures': ['InternVLChatModel'], 'llm_config': {}}

    with pytest.raises(ValueError, match='llm_config.architectures'):
        InternVL3_5Model(cfg, resolver=Mock())


def test_internvl35_model_rejects_unsupported_inner_architecture():
    from lmdeploy.turbomind.models.internvl3_5 import InternVL3_5Model

    with pytest.raises(ValueError, match='GptOssForCausalLM'):
        InternVL3_5Model(_internvl_cfg('GptOssForCausalLM'), resolver=Mock())


def test_supported_archs_maps_internvl_chat_model():
    from lmdeploy.turbomind.supported_models import SUPPORTED_ARCHS

    assert SUPPORTED_ARCHS['InternVLChatModel'] == 'internvl3_5'


def test_internvl35_model_is_registered():
    from lmdeploy.turbomind.models import InternVL3_5Model  # noqa: F401
    from lmdeploy.turbomind.models.base import INPUT_MODELS

    assert INPUT_MODELS.get('internvl3_5') is InternVL3_5Model
