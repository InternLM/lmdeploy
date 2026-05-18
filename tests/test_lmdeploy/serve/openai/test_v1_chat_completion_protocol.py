# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)


# ---- Task 1: Input field tests ----


def test_chat_completion_request_with_input_ids():
    """ChatCompletionRequest should accept input_ids."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[1, 2, 3],
    )
    assert req.input_ids == [1, 2, 3]


def test_chat_completion_request_with_image_data_str():
    """ChatCompletionRequest should accept image_data as a URL string."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[1, 2, 3],
        image_data="http://example.com/img.png",
    )
    assert req.image_data == "http://example.com/img.png"


def test_chat_completion_request_with_image_data_list():
    """ChatCompletionRequest should accept image_data as a list."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[1, 2, 3],
        image_data=["http://example.com/a.png", "http://example.com/b.png"],
    )
    assert len(req.image_data) == 2


def test_chat_completion_request_with_return_routed_experts():
    """ChatCompletionRequest should accept return_routed_experts."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="hello",
        return_routed_experts=True,
    )
    assert req.return_routed_experts is True


def test_chat_completion_request_new_field_defaults():
    """New fields should have sensible defaults."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="hello",
    )
    assert req.input_ids is None
    assert req.image_data is None
    assert req.return_routed_experts is False


# ---- Task 2: Output field tests ----


def test_chat_completion_response_with_routed_experts():
    """ChatCompletionResponse should carry routed_experts at the top level."""
    resp = ChatCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        routed_experts=[[[1, 2], [3, 4]]],
    )
    assert resp.routed_experts == [[[1, 2], [3, 4]]]


def test_chat_completion_stream_response_with_routed_experts():
    """ChatCompletionStreamResponse should carry routed_experts."""
    resp = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content="hi"),
                finish_reason=None,
            )
        ],
        routed_experts=[[[5, 6]]],
    )
    assert resp.routed_experts == [[[5, 6]]]


def test_chat_completion_response_defaults():
    """New output fields default to None."""
    resp = ChatCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )
    assert resp.routed_experts is None


def test_chat_completion_choice_output_token_logprobs():
    """ChatCompletionResponseChoice should carry output_token_logprobs."""
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hi"),
        finish_reason="stop",
        output_token_logprobs=[(0.95, 42), (0.88, 17)],
    )
    assert choice.output_token_logprobs == [(0.95, 42), (0.88, 17)]


# ---- Task 3: Validation tests ----


from lmdeploy.serve.openai.serving_chat_completion import check_request


class _FakeEngineConfig:
    logprobs_mode = None


class _FakeSessionManager:
    def has(self, session_id):
        return False


class _FakeServerContext:
    @staticmethod
    def get_engine_config():
        return _FakeEngineConfig()

    @staticmethod
    def get_session_manager():
        return _FakeSessionManager()


def test_input_ids_with_chat_messages_is_rejected():
    """input_ids cannot be used when messages is a non-empty chat history."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        input_ids=[1, 2, 3],
    )
    error = check_request(req, _FakeServerContext())
    assert error != ""
    assert "input_ids" in error.lower() or "messages" in error.lower()


def test_input_ids_with_nonempty_string_messages_is_rejected():
    """input_ids cannot be used when messages is a non-empty string."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="hello world",
        input_ids=[1, 2, 3],
    )
    error = check_request(req, _FakeServerContext())
    assert error != ""


def test_input_ids_with_empty_messages_is_ok():
    """input_ids is fine when messages is empty string."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[1, 2, 3],
    )
    error = check_request(req, _FakeServerContext())
    assert error == ""


def test_input_ids_empty_list_rejected():
    """input_ids must not be an empty list."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[],
    )
    error = check_request(req, _FakeServerContext())
    assert error != ""
    assert "input_ids" in error.lower()


def test_image_data_without_input_ids_rejected():
    """image_data without input_ids is not valid (no text prompt to attach to)."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        image_data="http://example.com/img.png",
    )
    error = check_request(req, _FakeServerContext())
    assert error != ""


def test_image_data_with_input_ids_is_ok():
    """image_data is valid alongside input_ids when messages is empty."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="",
        input_ids=[1, 2, 3],
        image_data="http://example.com/img.png",
    )
    error = check_request(req, _FakeServerContext())
    assert error == ""


def test_image_data_with_nonempty_messages_is_ignored():
    """image_data is silently ignored with chat-style messages (multimodal already in content)."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "describe this"}],
        image_data="http://example.com/img.png",
    )
    # No validation error — image_data is just ignored
    error = check_request(req, _FakeServerContext())
    assert error == ""


def test_image_data_with_string_messages_is_ok():
    """image_data is valid with a string messages (/generate-style prompt + image)."""
    req = ChatCompletionRequest(
        model="test-model",
        messages="describe this image",
        image_data="http://example.com/img.png",
    )
    error = check_request(req, _FakeServerContext())
    assert error == ""
