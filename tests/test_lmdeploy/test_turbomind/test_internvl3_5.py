# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import _turbomind as _tm
import pytest
from transformers import PretrainedConfig

from lmdeploy.turbomind.checkpoint import Prefix
from lmdeploy.turbomind.models.qwen3 import Qwen3TextModel


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


def test_internvl35_model_delegates_runtime_and_model(monkeypatch):
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

    fake_ckpt = Mock()
    pfx = Prefix(fake_ckpt)
    model.model(pfx)

    fake_text_model.model.assert_called_once()
    inner_pfx = fake_text_model.model.call_args[0][0]
    assert isinstance(inner_pfx, Prefix)
    assert inner_pfx.ckpt is fake_ckpt
    assert inner_pfx.prefix == 'language_model'

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
