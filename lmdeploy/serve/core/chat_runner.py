# Copyright (c) OpenMMLab. All rights reserved.
"""Protocol-neutral runner for chat-like serving endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Any

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage, ToolChoice
from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators

from .content_stop import ChatContentStop
from .exceptions import ErrorCode, RequestError


@dataclass
class ChatRunnerOptions:
    """Endpoint-specific runtime knobs for the shared chat runner."""

    input_ids: list[int] | None = None
    do_preprocess: bool = True
    adapter_name: str | None = None
    gen_config_kwargs: dict[str, Any] = field(default_factory=dict)
    preprocess_kwargs: dict[str, Any] = field(default_factory=dict)
    match_glm_content_stop: bool = False


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
    stop_sequence: str | None = None


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
    stop_sequence: str | None = None


@dataclass
class ChatRunner:
    """Prepared shared chat generation run."""

    request: ChatCompletionRequest
    response_parser: Any
    gen_config: GenerationConfig
    session: Any
    result_generator: Any
    session_manager: Any
    prompt_text: str | None = None
    prompt_token_ids: list[int] | None = None
    content_stop: ChatContentStop | None = None
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
            if isinstance(request.tool_choice, ToolChoice):
                name = request.tool_choice.function.name
                matches = sum(tool.function.name == name for tool in request.tools or [])
                if matches != 1:
                    raise ValueError(f'Named tool_choice {name!r} must match exactly one declared tool.')
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
        content_stop = None
        if (options.match_glm_content_stop and gen_config.stop_words and not gen_config.ignore_eos
                and getattr(response_parser, 'supports_glm_content_stop', False)):
            engine_words = []
            multi_token_words = []
            tokenizer = server_context.async_engine.tokenizer
            for word in gen_config.stop_words:
                # indexes_containing_token() skips words that encode to more than one ID.
                if word and len(tokenizer.encode(word, add_bos=False)) > 1:
                    multi_token_words.append(word)
                else:
                    engine_words.append(word)
            if multi_token_words:
                gen_config.stop_words = engine_words or None
                content_stop = ChatContentStop(multi_token_words, gen_config.include_stop_str_in_output,
                                               response_parser.preview_stop_text,
                                               getattr(response_parser, 'resolve_content_separator', None))
        adapter_name = options.adapter_name
        if adapter_name is None and parsed_request.model != server_context.async_engine.model_name:
            adapter_name = parsed_request.model

        engine_messages = None if options.input_ids is not None else parsed_request.messages
        session = server_context.create_session(parsed_request.session_id)
        try:
            preprocessed = await server_context.async_engine.preprocess(
                engine_messages,
                session,
                gen_config=gen_config,
                tools=parsed_request.tools,
                reasoning_effort=parsed_request.reasoning_effort,
                do_preprocess=options.do_preprocess,
                adapter_name=adapter_name,
                trace_model=parsed_request.model,
                chat_template_kwargs=_chat_template_kwargs_from_request(parsed_request),
                input_ids=options.input_ids,
                media_io_kwargs=parsed_request.media_io_kwargs,
                mm_processor_kwargs=parsed_request.mm_processor_kwargs,
                **options.preprocess_kwargs,
            )
            if getattr(response_parser, 'requires_output_chunks', False):
                response_parser.bind_generation_config(preprocessed.gen_config)
            if content_stop is not None:
                gen_config = preprocessed.gen_config
                # PyTorch disables an unsatisfiable minimum after resolving max_new_tokens.
                # Use that same threshold for text stops without changing the engine's config.
                if (gen_config.min_new_tokens and gen_config.min_new_tokens > gen_config.max_new_tokens
                        and server_context.async_engine.backend == 'pytorch'):
                    gen_config = replace(gen_config, min_new_tokens=0)
            prompt_token_ids = list(preprocessed.inputs['input_ids']) if parsed_request.return_token_ids else None
            generate_kwargs = {'content_stop': content_stop} if content_stop is not None else {}
            result_generator = server_context.async_engine.generate(
                preprocessed, stream_response=True, **generate_kwargs)
        except BaseException:
            # This runner owns the newly created API session even if engine
            # preprocessing fails before it takes ownership (e.g. grammar
            # compilation). Also release it on cancellation during preparation.
            server_context.session_manager.remove(session)
            raise
        return cls(
            request=parsed_request,
            response_parser=response_parser,
            gen_config=gen_config,
            session=session,
            result_generator=result_generator,
            session_manager=server_context.session_manager,
            content_stop=content_stop,
            prompt_text=preprocessed.inputs.get('prompt') if parsed_request.return_prompt_text else None,
            prompt_token_ids=prompt_token_ids,
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

    async def stream(self, *, collect_token_metadata: bool = False) -> AsyncGenerator[ChatStreamChunk, None]:
        """Yield parser-normalized streaming chunks and clean up the
        session."""
        streaming_tools = False
        named_calls: dict[int, str] | None = {} if isinstance(self.request.tool_choice, ToolChoice) else None
        return_token_metadata = bool(
            collect_token_metadata
            or self.request.return_token_ids
            or self.request.logprobs
            or self.request.return_logprob
        )
        try:
            async for res in self.result_generator:
                delta_text = res.response or ''
                delta_token_ids = res.token_ids if res.token_ids is not None else []
                try:
                    parser_kwargs = {}
                    if getattr(self.response_parser, 'requires_output_chunks', False):
                        parser_kwargs['engine_final'] = res.finish_reason is not None
                    stream_deltas = self.response_parser.stream_chunk(
                        delta_text,
                        delta_token_ids,
                        final=res.finish_reason is not None,
                        **parser_kwargs,
                    )
                    if self.content_stop is not None:
                        if res.finish_reason == 'stop' and delta_token_ids:
                            # The engine included its terminal ID. Preserve that stop and flush any pending prefix.
                            stream_deltas = self.content_stop.filter_deltas([], final=True) + stream_deltas
                        else:
                            stream_deltas = self.content_stop.filter_deltas(
                                stream_deltas, final=res.finish_reason is not None,
                                can_stop=res.generate_token_len > (self.gen_config.min_new_tokens or 0))
                    if not stream_deltas:
                        # With no visible parser output, emit only when this
                        # engine result still carries transport-level data.
                        has_token_metadata = return_token_metadata and bool(delta_token_ids)
                        if (
                            res.finish_reason is None
                            and not has_token_metadata
                            and res.cache_block_ids is None
                        ):
                            continue
                        stream_deltas = [(DeltaMessage(role='assistant', content=''), False)]

                except Exception as err:
                    raise RequestError(ErrorCode.INVALID_REQUEST, f'Failed to parse output: {err}') from err

                for delta_index, (delta_message, tool_emitted) in enumerate(stream_deltas):
                    if tool_emitted:
                        streaming_tools = True

                    is_last_delta = delta_index == len(stream_deltas) - 1
                    stop_sequence = getattr(res, 'stop_sequence', None)
                    finish_reason = res.finish_reason if is_last_delta else None
                    if (finish_reason in ('length', 'stop') and self.content_stop is not None
                            and self.content_stop.matched):
                        finish_reason, stop_sequence = 'stop', self.content_stop.matched
                    if named_calls is not None:
                        for call in delta_message.tool_calls or []:
                            if call.function is not None and call.function.name:
                                named_calls[call.index] = call.function.name
                        self._validate_named_tool_output(
                            list(named_calls.values()), finish_reason, stop_sequence)
                    if (
                            self.request.tool_choice != 'none'
                            and self.response_parser.tool_parser is not None
                            and finish_reason == 'stop'
                            and stop_sequence is None
                            and streaming_tools):
                        finish_reason = 'tool_calls'

                    yield ChatStreamChunk(
                        delta_message=delta_message,
                        tool_emitted=tool_emitted,
                        finish_reason=finish_reason,
                        token_ids=delta_token_ids if is_last_delta else [],
                        logprobs=res.logprobs if is_last_delta else None,
                        input_token_len=res.input_token_len,
                        generate_token_len=res.generate_token_len,
                        cached_tokens=res.cached_tokens,
                        routed_experts=res.routed_experts if finish_reason is not None else None,
                        cache_block_ids=res.cache_block_ids,
                        reasoning_tokens=self.response_parser.reasoning_tokens,
                        is_last_delta=is_last_delta,
                        stop_sequence=stop_sequence if is_last_delta else None,
                    )
        finally:
            await self.close()

    async def _collect_content_stream(self, raw_request) -> ChatResult:
        """Collect the same parsed deltas on which a streaming request would
        stop."""
        content = []
        reasoning = []
        tools = []
        token_ids = []
        logprobs = []
        cache_blocks = []
        remote_ids = []
        last = None
        stream = self.stream(collect_token_metadata=True)
        try:
            async for chunk in stream:
                if raw_request is not None and await raw_request.is_disconnected():
                    await self.session.async_abort()
                    raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
                last = chunk
                message = chunk.delta_message
                if message.content:
                    content.append(message.content)
                if message.reasoning_content:
                    reasoning.append(message.reasoning_content)
                if message.tool_calls:
                    tools.extend(message.tool_calls)
                if chunk.is_last_delta:
                    token_ids.extend(chunk.token_ids)
                    logprobs.extend(chunk.logprobs or [])
                    cache_blocks.append(chunk.cache_block_ids)
                    remote_ids.append(chunk.token_ids)
        finally:
            await stream.aclose()
        if last is None:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'No generation output from engine.')
        return ChatResult(
            text=''.join(content) or None,
            tool_calls=ToolParser.build_tool_calls(tools) or None,
            reasoning_content=''.join(reasoning) or None,
            finish_reason=last.finish_reason,
            input_token_len=last.input_token_len,
            generate_token_len=last.generate_token_len,
            cached_tokens=last.cached_tokens,
            token_ids=token_ids,
            logprobs=logprobs,
            routed_experts=last.routed_experts,
            cache_block_ids=cache_blocks,
            remote_token_ids=remote_ids,
            reasoning_tokens=last.reasoning_tokens,
            stop_sequence=last.stop_sequence,
        )

    async def collect(self, raw_request=None) -> ChatResult:
        """Collect, parse, and clean up a non-streaming generation."""
        if self.content_stop is not None:
            return await self._collect_content_stream(raw_request)
        final_res = None
        text = ''
        final_token_ids: list[int] = []
        final_logprobs: list[dict[int, float]] = []
        cache_block_ids: list[Any] = []
        remote_token_ids: list[Any] = []
        output_chunks: list[tuple[str, list[int], bool]] | None = (
            [] if getattr(self.response_parser, 'requires_output_chunks', False) else None)
        try:
            async for res in self.result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    await self.session.async_abort()
                    raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
                final_res = res
                text += res.response or ''
                if output_chunks is not None:
                    # Keep even empty-text batches: their IDs may complete a
                    # UTF-8 character in the following native decode result.
                    output_chunks.append((res.response or '', res.token_ids or [], res.finish_reason is not None))
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
            parser_kwargs = {'output_chunks': output_chunks} if output_chunks is not None else {}
            text, tool_calls, reasoning_content = self.response_parser.parse_complete(
                text, final_token_ids, **parser_kwargs)
            if (isinstance(tool_calls, list) and len(tool_calls) and final_res.finish_reason == 'stop'
                    and getattr(final_res, 'stop_sequence', None) is None):
                final_res.finish_reason = 'tool_calls'
        except Exception as err:
            raise RequestError(ErrorCode.INVALID_REQUEST, f'Failed to parse output: {err}') from err

        if isinstance(self.request.tool_choice, ToolChoice):
            names = [call['function']['name'] if isinstance(call, dict) else call.function.name
                     for call in tool_calls or []]
            self._validate_named_tool_output(names, final_res.finish_reason, getattr(final_res, 'stop_sequence', None))

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
            stop_sequence=getattr(final_res, 'stop_sequence', None),
        )

    def _validate_named_tool_output(self, names: list[str], finish_reason: str | None,
                                    stop_sequence: str | None) -> None:
        """Do not report successful named generation without its parsed
        call."""
        tool_choice = self.request.tool_choice
        assert isinstance(tool_choice, ToolChoice)
        name = tool_choice.function.name
        invalid_calls = len(names) > 1 or any(generated != name for generated in names)
        missing_call = finish_reason in ('stop', 'tool_calls') and stop_sequence is None and not names
        if invalid_calls or missing_call:
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Generated output did not contain exactly one call to the named tool {name!r}.')


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
