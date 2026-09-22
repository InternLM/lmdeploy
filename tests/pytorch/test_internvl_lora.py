from unittest.mock import MagicMock, patch

import lmdeploy.pytorch.adapter.adapter as adapter_module
from lmdeploy.pytorch.models.internvl import InternVLChatModel
from lmdeploy.pytorch.models.internvl3_hf import InternVLForConditionalGeneration

_LORA_WEIGHTS = [('base_model.model.language_model.model.layers.0.mlp.down_proj.lora_A.weight', None)]


class _LanguageModelWithoutLora:
    """Inner language model that does not implement ``load_lora_weights`` so
    the VLM wrapper falls back to the standalone adapter loader."""


def _assert_fallback_passes_model(model_cls):
    model = model_cls.__new__(model_cls)
    model.language_model = _LanguageModelWithoutLora()

    with patch.object(adapter_module, 'load_lora_weights') as mock_loader:
        model.load_lora_weights(_LORA_WEIGHTS, adapter_id=7)

    # The fallback must call the standalone loader as
    # load_lora_weights(model, weights, adapter_id). The top-level model is
    # required because its named_parameters() carry the ``language_model.``
    # prefix the loader maps against. It was previously called as
    # load_lora_weights(weights, adapter_id), raising
    # ``TypeError: load_lora_weights() missing 1 required positional argument:
    # 'adapter_id'``.
    mock_loader.assert_called_once_with(model, _LORA_WEIGHTS, 7)


def test_internvl_chat_model_lora_fallback_passes_model():
    _assert_fallback_passes_model(InternVLChatModel)


def test_internvl_hf_lora_fallback_passes_model():
    # Regression: InternVLForConditionalGeneration referenced
    # ``self.model.language_model``, but the constructor stores
    # ``self.language_model``. ``self.model`` did not exist, so hasattr swallowed
    # the AttributeError and the fallback fired unconditionally with wrong arity.
    _assert_fallback_passes_model(InternVLForConditionalGeneration)


def test_internvl_hf_lora_delegates_to_language_model():
    # When the inner language model implements load_lora_weights, the wrapper
    # must delegate to it -- which only works once the hasattr check looks at
    # self.language_model instead of the non-existent self.model.
    model = InternVLForConditionalGeneration.__new__(InternVLForConditionalGeneration)
    inner = MagicMock()
    model.language_model = inner

    model.load_lora_weights(_LORA_WEIGHTS, adapter_id=9)

    inner.load_lora_weights.assert_called_once_with(_LORA_WEIGHTS, 9)
