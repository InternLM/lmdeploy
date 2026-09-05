# Copyright (c) OpenMMLab. All rights reserved.
"""Protocol-neutral runner for chat-like serving endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators

from .exceptions import ErrorCode, RequestError


def should_validate_complete(
    request: ChatCompletionRequest,
    finish_reason: str | None,
) -> bool:
    """Whether parser validity may change this terminal finish reason."""
    return finish_reason in ('stop', 'length') and (
        bool(request.return_token_ids) or bool(request.return_routed_experts))


@dataclass
class ChatRunnerOptions:
    """Endpoint-specific runtime knobs for the shared chat runner."""

    input_ids: list[int] | None = None
    do_preprocess: bool = True
    adapter_name: str | None = None
    gen_config_kwargs: dict[str, Any] = field(default_factory=dict)
    preprocess_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatStreamChunk:
    """One parsed protocol-neutral delta from a chat generation stream."""

    delta_message: DeltaMessage
    tool_emitted: bool
    finish_reason: str | None
    token_ids: list[int]
    logprobs: list[dict[int, float]] | None
    input_token_len: int
    generate_token_len: int
    cached_tokens: int
    routed_experts: Any = None
    cache_block_ids: list[int] | None = None
    reasoning_tokens: int | None = None
    is_last_delta: bool = True


@dataclass
class ChatResult:
    """Complete parsed result from a chat generation request."""

    text: str | None
    tool_calls: list[Any] | None
    reasoning_content: str | None
    finish_reason: str | None
    input_token_len: int
    generate_token_len: int
    cached_tokens: int
    token_ids: list[int]
    logprobs: list[dict[int, float]]
    routed_experts: Any = None
    cache_block_ids: list[Any] = field(default_factory=list)
    remote_token_ids: list[Any] = field(default_factory=list)
    reasoning_tokens: int | None = None


@dataclass
class ChatRunner:
    """Prepared shared chat generation run."""

    request: ChatCompletionRequest
    response_parser: Any
    gen_config: GenerationConfig
    session: Any
    result_generator: Any
    session_manager: Any
    _closed: bool = False

    @classmethod
    async def prepare(
        cls,
        server_context,
        request: ChatCompletionRequest,
        options: ChatRunnerOptions | None = None,
    ) -> ChatRunner:
        """Prepare a chat-like request for streaming or non-streaming
        consumption."""
        options = options or ChatRunnerOptions()
        parser_cls = server_context.response_parser_cls

        try:
            if request.tool_choice == 'required' and not parser_cls.supports_required_tool_choice:
                raise ValueError(
                    f'Response parser {parser_cls.__name__!r} does not support `tool_choice="required"`.')
            response_parser = parser_cls(request)
            parsed_request = response_parser.request
        except ValueError as err:
            raise RequestError(ErrorCode.INVALID_REQUEST, str(err)) from err

        gen_config = _build_runner_generation_config(
            parsed_request,
            server_context,
            options.gen_config_kwargs,
        )
        adapter_name = options.adapter_name
        if adapter_name is None and parsed_request.model != server_context.async_engine.model_name:
            adapter_name = parsed_request.model

        engine_messages = None if options.input_ids is not None else parsed_request.messages
        session = server_context.create_session(parsed_request.session_id)
        preprocessed = await server_context.async_engine.preprocess(
            engine_messages,
            session,
            gen_config=gen_config,
            tools=parsed_request.tools,
            reasoning_effort=parsed_request.reasoning_effort,
            do_preprocess=options.do_preprocess,
            adapter_name=adapter_name,
            chat_template_kwargs=_chat_template_kwargs_from_request(parsed_request),
            input_ids=options.input_ids,
            media_io_kwargs=parsed_request.media_io_kwargs,
            mm_processor_kwargs=parsed_request.mm_processor_kwargs,
            **options.preprocess_kwargs,
        )
        result_generator = server_context.async_engine.generate(preprocessed, stream_response=True)
        return cls(
            request=parsed_request,
            response_parser=response_parser,
            gen_config=gen_config,
            session=session,
            result_generator=result_generator,
            session_manager=server_context.session_manager,
        )

    async def close(self) -> None:
        """Close the engine generator and remove the API session."""
        if self._closed:
            return
        self._closed = True
        cleanup_task = asyncio.create_task(
            cleanup_result_generators(
                [self.result_generator],
                [self.session],
                self.session_manager,
            ),
            name='chat_runner_cleanup',
        )
        await asyncio.shield(cleanup_task)

    async def stream(self) -> AsyncGenerator[ChatStreamChunk, None]:
        """Yield parser-normalized streaming chunks and clean up the
        session."""
        streaming_tools = False
        try:
            async for res in self.result_generator:
                delta_text = res.response or ''
                delta_token_ids = res.token_ids if res.token_ids is not None else []
                try:
                    stream_deltas = self.response_parser.stream_chunk(
                        delta_text,
                        delta_token_ids,
                        final=res.finish_reason is not None,
                    )
                    if not stream_deltas:
                        # Parser may buffer partial protocol tags and emit no visible delta
                        # while the engine still produced new tokens. Keep metadata attached.
                        if res.finish_reason is None and not delta_token_ids:
                            continue
                        stream_deltas = [(DeltaMessage(role='assistant', content=''), False)]

                    if (
                            should_validate_complete(self.request, res.finish_reason)
                            and not self.response_parser.validate_complete()):
                        res.finish_reason = 'parse_error'
                except Exception as err:
                    raise RequestError(ErrorCode.INVALID_REQUEST, f'Failed to parse output: {err}') from err

                for delta_index, (delta_message, tool_emitted) in enumerate(stream_deltas):
                    if tool_emitted:
                        streaming_tools = True

                    is_last_delta = delta_index == len(stream_deltas) - 1
                    finish_reason = res.finish_reason if is_last_delta else None
                    if (
                            self.request.tool_choice != 'none'
                            and self.response_parser.tool_parser is not None
                            and finish_reason == 'stop'
                            and streaming_tools):
                        finish_reason = 'tool_calls'

                    yield ChatStreamChunk(
                        delta_message=delta_message,
                        tool_emitted=tool_emitted,
                        finish_reason=finish_reason,
                        token_ids=delta_token_ids,
                        logprobs=res.logprobs,
                        input_token_len=res.input_token_len,
                        generate_token_len=res.generate_token_len,
                        cached_tokens=res.cached_tokens,
                        routed_experts=res.routed_experts if finish_reason is not None else None,
                        cache_block_ids=res.cache_block_ids,
                        reasoning_tokens=self.response_parser.reasoning_tokens,
                        is_last_delta=is_last_delta,
                    )
        finally:
            await self.close()

    async def collect(self, raw_request=None) -> ChatResult:
        """Collect, parse, validate, and clean up a non-streaming
        generation."""
        final_res = None
        text = ''
        final_token_ids: list[int] = []
        final_logprobs: list[dict[int, float]] = []
        cache_block_ids: list[Any] = []
        remote_token_ids: list[Any] = []
        try:
            async for res in self.result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    await self.session.async_abort()
                    raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
                final_res = res
                text += res.response or ''
                if res.token_ids:
                    final_token_ids.extend(res.token_ids)
                if res.logprobs:
                    final_logprobs.extend(res.logprobs)
                cache_block_ids.append(res.cache_block_ids)
                remote_token_ids.append(res.token_ids)
        finally:
            await self.close()

        if final_res is None:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'No generation output from engine.')

        try:
            raw_text = text
            text, tool_calls, reasoning_content = self.response_parser.parse_complete(text, final_token_ids)
            if (
                    should_validate_complete(self.request, final_res.finish_reason)
                    and not self.response_parser.validate_complete(raw_text)):
                final_res.finish_reason = 'parse_error'
            if isinstance(tool_calls, list) and len(tool_calls) and final_res.finish_reason == 'stop':
                final_res.finish_reason = 'tool_calls'
        except Exception as err:
            raise RequestError(ErrorCode.INVALID_REQUEST, f'Failed to parse output: {err}') from err

        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            finish_reason=final_res.finish_reason,
            input_token_len=final_res.input_token_len,
            generate_token_len=final_res.generate_token_len,
            cached_tokens=final_res.cached_tokens,
            token_ids=final_token_ids,
            logprobs=final_logprobs,
            routed_experts=final_res.routed_experts,
            cache_block_ids=cache_block_ids,
            remote_token_ids=remote_token_ids,
            reasoning_tokens=self.response_parser.reasoning_tokens,
        )


def _build_runner_generation_config(
    request: ChatCompletionRequest,
    server_context,
    gen_config_kwargs: dict[str, Any],
) -> GenerationConfig:
    gen_config_kwargs = _normalize_runner_gen_config_kwargs(request, gen_config_kwargs)
    max_new_tokens = request.max_completion_tokens
    if max_new_tokens is None:
        max_new_tokens = request.max_tokens
    response_format = request.response_format
    if isinstance(response_format, dict):
        request = request.model_copy(update={'response_format': None})
        gen_config_kwargs['response_format'] = response_format
    stop_words = request.stop
    if isinstance(stop_words, str):
        stop_words = [stop_words]
    return build_generation_config(
        request,
        server_context.default_gen_config,
        max_new_tokens=max_new_tokens,
        stop_words=stop_words,
        **gen_config_kwargs,
    )


def _normalize_runner_gen_config_kwargs(
    request: ChatCompletionRequest,
    gen_config_kwargs: dict[str, Any],
) -> dict[str, Any]:
    gen_config_kwargs = dict(gen_config_kwargs)
    if 'logprobs' in gen_config_kwargs:
        return gen_config_kwargs
    if request.logprobs:
        gen_config_kwargs['logprobs'] = request.top_logprobs or 1
    elif request.return_logprob:
        gen_config_kwargs['logprobs'] = 1
    elif {'logprobs', 'top_logprobs', 'return_logprob'} & request.model_fields_set:
        gen_config_kwargs['logprobs'] = None
    return gen_config_kwargs


def _chat_template_kwargs_from_request(request: ChatCompletionRequest) -> dict | None:
    chat_template_kwargs = dict(request.chat_template_kwargs or {})
    if request.enable_thinking is not None and chat_template_kwargs.get('enable_thinking') is None:
        chat_template_kwargs['enable_thinking'] = request.enable_thinking
    return chat_template_kwargs or None
