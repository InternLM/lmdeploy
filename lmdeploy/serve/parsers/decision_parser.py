# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision request adaptation using the engine's existing model resources."""
from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from lmdeploy.intern_decision_encoding import (
    ANSWER_SYMBOLS,
    DECISION_TOKEN,
    DecisionOutputError,
    compile_row,
    decode_answers,
    resolve_temperature,
)
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.decision_protocol import (
    DecisionInputImage,
    DecisionInputMessage,
    DecisionRequest,
    DecisionResponse,
    OpenAIDecisionRequest,
    OpenAIDecisionResponse,
)

TASK_NAME = 'intern-decision'


@dataclass(frozen=True)
class DecisionInput:
    """Engine input compiled from one decision request."""

    messages: list[dict]
    gen_config: GenerationConfig
    preprocess_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionContext:
    """Small immutable model metadata; tokenizer/processor ownership stays with the engine."""

    marker_id: int
    symbol_ids: Mapping[str, int]
    image_token_id: int | None
    temperature: float


def create_decision_context(engine, model_path: str) -> DecisionContext:
    """Check task capabilities once without loading another tokenizer or processor."""
    if engine.backend != 'pytorch' or not isinstance(engine.backend_config, PytorchEngineConfig):
        raise ValueError(f'{TASK_NAME} requires the PyTorch backend')
    if engine.speculative_config is not None:
        raise ValueError(f'{TASK_NAME} does not support speculative decoding')
    if engine.backend_config.role != EngineRole.Hybrid:
        raise ValueError(f'{TASK_NAME} requires the hybrid engine role')

    tokenizer = engine.tokenizer.model.model
    if DECISION_TOKEN not in tokenizer.get_added_vocab():
        raise ValueError('Checkpoint must include its trained decision tokenizer')
    if not tokenizer.chat_template:
        raise ValueError('Decision checkpoint must supply its trained chat template')

    marker_id = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
    if tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [marker_id]:
        raise ValueError('Decision marker must encode to exactly one registered token')

    symbol_ids = {}
    for symbol in ANSWER_SYMBOLS:
        ids = tokenizer.encode(symbol, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Answer symbol {symbol!r} must encode to exactly one token')
        symbol_ids[symbol] = ids[0]
    if len(set(symbol_ids.values())) != len(ANSWER_SYMBOLS) or marker_id in symbol_ids.values():
        raise ValueError('Answer symbols must have distinct token IDs, separate from the decision marker')

    vocab_size = getattr(engine.hf_cfg, 'text_config', engine.hf_cfg).vocab_size
    if any(not 0 <= token < vocab_size for token in [marker_id, *symbol_ids.values()]):
        raise ValueError('Decision tokenizer ids exceed the model vocabulary')

    vision = getattr(engine, 'vl_encoder', None)
    image_token = None
    if vision is not None:
        processor = vision.model.processor
        if not getattr(processor, 'chat_template', None):
            raise ValueError('Decision checkpoint must supply its trained processor chat template')
        # The visual processor tokenizes the rendered image prompt. Do not replace
        # its tokenizer while other requests may be using it.
        visual_tokenizer = processor.tokenizer
        for text, token in [(DECISION_TOKEN, marker_id), *symbol_ids.items()]:
            if visual_tokenizer.encode(text, add_special_tokens=False) != [token]:
                raise ValueError('Decision text and visual tokenizer mappings disagree')
        image_token = getattr(processor, 'image_token', None)
    elif '<|image_pad|>' in tokenizer.get_vocab():
        image_token = '<|image_pad|>'
    return DecisionContext(
        marker_id=marker_id,
        symbol_ids=MappingProxyType(symbol_ids),
        image_token_id=tokenizer.convert_tokens_to_ids(image_token) if image_token else None,
        temperature=resolve_temperature(model_path),
    )


@dataclass(frozen=True)
class _QuestionMapping:
    """Request-owned wire values, separate from the checkpoint's string keys."""

    field: str
    kind: str
    name: str | None
    choice_keys: tuple[str, ...] = ()
    choice_values: tuple[str | bool, ...] = ()
    level_labels: tuple[str, ...] = ()


class DecisionParser:
    """Per-request task input/output conversion; no engine execution or image ownership."""

    def __init__(self, request, context: DecisionContext):
        self._question_mappings = None
        if isinstance(request, Mapping) and 'input' in request:
            request = OpenAIDecisionRequest.model_validate(request)
        if isinstance(request, OpenAIDecisionRequest):
            self.request = self._adapt_openai_request(request)
        else:
            self.request = DecisionRequest.model_validate(request)
        self.context = context
        self.compiled = None
        self.candidate_ids = ()
        self.score_row_indices = ()
        self.expected_score_rows = 0

    @staticmethod
    def _adapt_openai_input(value: str | list[DecisionInputMessage]) -> tuple[Any, list[str]]:
        """Extract images in encounter order and retain one-based references in state."""
        if isinstance(value, str):
            return value, []
        if len(value) == 1 and isinstance(value[0].content, str):
            return value[0].content, []
        state, images = [], []
        for message in value:
            if isinstance(message.content, str):
                content = message.content
            else:
                content = []
                for part in message.content:
                    if isinstance(part, DecisionInputImage):
                        images.append(part.image_url)
                        content.append(dict(type='image', image_index=len(images)))
                    else:
                        content.append(dict(type='text', text=part.text))
            state.append(dict(role=message.role, content=content))
        return state, images

    def _adapt_openai_request(self, request: OpenAIDecisionRequest) -> DecisionRequest:
        """Normalize wire questions once and retain their inverse mapping.

        Input normalization and typed choice encoding follow intern-decision-inference's decisions_api
        (Apache-2.0, a8fe089); model identity is checked by the HTTP endpoint.
        """
        reserved = {question.name for question in request.questions if question.name is not None}
        questions = {}
        mappings = []
        for index, question in enumerate(request.questions):
            key = question.name
            if not key or key in questions:
                key = f'question_{index}'
                while key in reserved or key in questions:
                    key = '_' + key
            item = dict(type=question.type, instructions=question.instructions)
            choice_keys, choice_values, labels = (), (), ()
            if question.type == 'predicate':
                item['type'] = 'noul'
            elif question.type == 'choice':
                choice_values = tuple(choice.value for choice in question.choices)
                typed = any(isinstance(value, bool) for value in choice_values)
                choice_keys = tuple(json.dumps(value, ensure_ascii=False) if typed else value
                                    for value in choice_values)
                item['criteria'] = dict(zip(choice_keys, (choice.description for choice in question.choices),
                                            strict=True))
            else:
                labels = tuple(level.label for level in question.levels)
                item['criteria'] = [f'{level.label}: {level.description}' if level.description else level.label
                                    for level in question.levels]
            questions[key] = item
            mappings.append(_QuestionMapping(key, question.type, question.name, choice_keys, choice_values, labels))
        state, images = self._adapt_openai_input(request.input)
        internal = DecisionRequest(model=request.model, state=state, questions=questions, images=images)
        self._question_mappings = tuple(mappings)
        return internal

    def prepare_input(self) -> DecisionInput:
        row = self.request.model_dump(exclude={'model'})
        self.compiled = compile_row(row)
        self.candidate_ids = tuple(
            tuple(self.context.symbol_ids[symbol] for symbol in self.compiled.symbols[field])
            for field in self.compiled.fields)
        return DecisionInput(
            messages=self.compiled.messages,
            # The candidate set comes from the checkpoint, so every decision
            # request on this engine names the same ids. Within decision-only
            # batches the engine's union of requested ids therefore equals this
            # set and no per-request column projection happens.
            gen_config=GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0,
                                        logprob_extra_token_ids=sorted(set(self.context.symbol_ids.values()))),
            preprocess_kwargs=dict(do_preprocess=True,
                                   chat_template_kwargs=dict(enable_thinking=False, add_vision_id=True)),
        )

    def finalize_input(self, prepared) -> None:
        ids = prepared.inputs['input_ids']
        markers = [i for i, token in enumerate(ids) if token == self.context.marker_id]
        if len(markers) != len(self.compiled.fields) or markers[0] < 1:
            raise ValueError('Decision marker count or position mismatch')
        if self.context.image_token_id is not None:
            slots = [i for i, token in enumerate(ids) if token == self.context.image_token_id]
            if (not self.request.images and slots) or any(i >= markers[0] for i in slots):
                raise ValueError('Image placeholder count or position mismatch')
        # Scoring row 0 predicts the token at start + 1 in the expanded input.
        start = markers[0] - 1
        self.score_row_indices = tuple(position - (start + 1) for position in markers)
        self.expected_score_rows = len(ids) - (start + 1)
        prepared.gen_config.logprob_start_len = start

    def parse_output(self, output, *, model_name: str, timing: dict[str, float]) -> dict:
        if output.finish_reason == 'abort':
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE, 'The decision request was aborted by the engine')
        if output.logprobs is None:
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Scoring returned no input logprobs (finish_reason={output.finish_reason})')
        if len(output.logprobs) != self.expected_score_rows:
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Expected {self.expected_score_rows} scoring rows, got {len(output.logprobs)}')
        rows = [output.logprobs[i] for i in self.score_row_indices]
        try:
            logits = [[row[token] for token in ids] for row, ids in zip(rows, self.candidate_ids)]
        except KeyError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Scoring rows lack a requested candidate token') from exc
        try:
            answers = decode_answers(self.request.questions, self.compiled.fields, logits,
                                     temperature=self.context.temperature)
        except DecisionOutputError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, str(exc)) from exc
        if self._question_mappings is not None:
            return self._format_openai_response(answers, output, model_name=model_name)
        return DecisionResponse(
            request_id=secrets.token_hex(16), model=model_name, answers=answers,
            usage=dict(input_tokens=output.input_token_len, output_tokens=len(answers),
                       decision_count=len(answers), generated_tokens=0),
            timing=timing,
            calibration=dict(method='temperature-scaling', temperature=self.context.temperature),
        ).model_dump()

    def _format_openai_response(self, answers, output, *, model_name: str) -> dict:
        results = []
        for mapping in self._question_mappings:
            source = answers[mapping.field]
            answer = dict(type=mapping.kind, name=mapping.name)
            if mapping.kind == 'predicate':
                answer['probability'] = source['noul']
            elif mapping.kind == 'choice':
                answer.update(
                    choice=mapping.choice_values[mapping.choice_keys.index(source['choice'])],
                    confidence=source['confidence'],
                    probabilities=[dict(value=value, probability=source['probabilities'][key])
                                   for key, value in zip(mapping.choice_keys, mapping.choice_values, strict=True)],
                )
            else:
                answer.update(
                    score=source['score'], confidence=source['confidence'],
                    probabilities=[dict(value=index, label=label, probability=source['probabilities'][str(index)])
                                   for index, label in enumerate(mapping.level_labels)],
                )
            results.append(answer)
        return OpenAIDecisionResponse(
            model=model_name, answers=results,
            usage=dict(input_tokens=output.input_token_len, output_tokens=output.generate_token_len,
                       total_tokens=output.input_token_len + output.generate_token_len,
                       # Cache writes are not exposed by GenOut. Zero is a documented
                       # SDK compatibility placeholder, not a measured write count.
                       input_tokens_details=dict(cached_tokens=output.cached_tokens, cache_write_tokens=0),
                       output_tokens_details=dict(reasoning_tokens=0)),
        ).model_dump()
