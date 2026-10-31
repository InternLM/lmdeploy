# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inputs/schema.py,
# src/inference/encoding.py and inference.py (DecisionEngine.predict, scale_result).
# Modified for LMDeploy: inference-only prompts, engine-expanded image placeholders,
# candidate-logit decoding with finite checks and temperature calibration.
"""The trained Intern-Decision prompt, token encoding and candidate-score decoding."""

from __future__ import annotations

import ast
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

# Reuse the serving logger without importing engine dependencies into standalone scorers.
logger = logging.getLogger('lmdeploy')

DECISION_TOKEN = '<decision>'
ANSWER_SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
# Fall back to the official 4B calibration when checkpoint metadata is unavailable.
DEFAULT_TEMPERATURE = 1.99241824
SYSTEM_PROMPT = ('You are a careful decision assistant. Use the state and decision schema in '
                 'the user message to make the requested decisions. For every field, choose '
                 'exactly one answer symbol (e.g. A, B, C, ...) from its listed options and '
                 'return one valid JSON object mapping each field name to its chosen symbol. '
                 'Use the field names and symbols exactly as given. Do not include '
                 'explanations, Markdown, or extra text.')


@dataclass(frozen=True)
class CompiledExample:
    messages: list[dict[str, Any]]
    fields: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]


class DecisionOutputError(RuntimeError):
    """Candidate scores cannot be decoded into a consistent decision result."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


class AnswerOption(NamedTuple):
    """One selectable answer: its canonical value and the prompt description."""

    value: str
    description: str


def answer_options(question: Mapping[str, Any]) -> list[AnswerOption]:
    """Return the selectable answers of one question in answer-symbol order.

    Position matters: the first option is rendered as ``A``, the second as ``B``
    and so on, and decoding maps those symbols back through the same order.
    """
    kind = question.get('type')
    criteria = question.get('criteria')
    if kind == 'choice':
        if not isinstance(criteria, Mapping):
            raise ValueError('choice criteria must be an object')
        return [AnswerOption(str(key), str(value)) for key, value in criteria.items()]
    if kind == 'score':
        if isinstance(criteria, list):
            return [AnswerOption(str(index), str(value)) for index, value in enumerate(criteria)]
        if isinstance(criteria, Mapping):
            return [AnswerOption(str(key), str(value)) for key, value in criteria.items()]
        raise ValueError('score criteria must be a list or object')
    if kind == 'noul':
        descriptions = criteria if isinstance(criteria, Mapping) else {}
        yes = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {'yes', 'true', '1'}),
            'The answer is yes (affirmative, or align with the claim).',
        )
        no = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {'no', 'false', '0'}),
            'The answer is no (negative, or disagree with the claim).',
        )
        return [AnswerOption('no', no), AnswerOption('yes', yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def compile_row(row: Mapping[str, Any]) -> CompiledExample:
    questions = row.get('questions')
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(f"{row.get('id', '<missing id>')}: questions must be a non-empty object")

    fields: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} is not an object")
        options = answer_options(question)
        if not options:
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} has no options")
        if len(options) > len(ANSWER_SYMBOLS):
            raise ValueError(f"At most {len(ANSWER_SYMBOLS)} single-token answer symbols are supported")
        field_name = str(field)
        fields.append(field_name)
        field_symbols = tuple(ANSWER_SYMBOLS[:len(options)])
        symbols[field_name] = field_symbols
        schema_lines.append(f"{field_name}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(field_symbols, options):
            schema_lines.append(f"    {symbol} = {value}: {description}")

    state = _json(row.get('state'))
    user_text = ("Return one answer for every field using the supplied answer symbols.\n\n"
                 "## State\n"
                 f"{state}\n"
                 "## Decision schema\n" + "\n".join(schema_lines))
    if DECISION_TOKEN in user_text or any(DECISION_TOKEN in field for field in fields):
        raise ValueError('Reserved decision marker appears in input evidence')
    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    images = row.get('images', [])
    images = images if isinstance(images, list) else [images]
    image_blocks = [
        {'type': 'image_url', 'image_url': {'url': image} if isinstance(image, str) else image}
        for image in images
    ]
    messages = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': [*image_blocks, {'type': 'text', 'text': user_text}] if image_blocks else user_text
        },
        {
            'role': 'assistant',
            'content': skeleton
        },
    ]
    return CompiledExample(
        messages=messages,
        fields=tuple(fields),
        symbols=symbols,
    )


def validate_temperature(temperature) -> float:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return temperature


def resolve_temperature(model_path: str | Path) -> float:
    """Read calibration for local Intern-Decision family checkpoints.

    This reader follows the Intern-Decision-specific convention of a top-level
    numeric DEFAULT_TEMPERATURE in the checkpoint's inference.py. It is not a
    general checkpoint metadata format. Parse the constant without importing or
    executing the module or downloading files. Missing or invalid metadata falls
    back to the official Intern-Decision-4B calibration.
    """
    source = Path(model_path) / 'inference.py'
    try:
        tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
        values = []
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id == 'DEFAULT_TEMPERATURE' for target in targets):
                values.append(statement.value)
        if len(values) != 1:
            raise ValueError('Expected one top-level DEFAULT_TEMPERATURE assignment')
        value = ast.literal_eval(values[0])
        if type(value) not in (int, float):
            raise ValueError('DEFAULT_TEMPERATURE must be a numeric constant')
        temperature = validate_temperature(value)
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError, OverflowError) as exc:
        logger.warning(f'Cannot read decision temperature from {source}: {exc}; '
                       f'using the 4B default {DEFAULT_TEMPERATURE}.')
        return DEFAULT_TEMPERATURE
    logger.info(f'Decision temperature from {source}: {temperature}.')
    return temperature


def argmax(probs: dict[str, float]) -> str:
    return min(probs, key=lambda label: (-probs[label], label))


def scale_probabilities(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a candidate distribution exactly as the checkpoint's reference does."""
    if temperature == 1.0:
        return dict(probs)
    logs = {label: math.log(p) if p else -math.inf for label, p in probs.items()}
    maximum = max(logs.values())
    weights = {label: math.exp((logp - maximum) / temperature) for label, logp in logs.items()}
    total = sum(weights.values())
    result = {label: weight / total for label, weight in weights.items()}
    if argmax(result) != argmax(probs):
        raise DecisionOutputError('Floating-point temperature scaling changed argmax')
    return result


def decode_answers(questions: dict,
                   fields: tuple[str, ...],
                   candidate_logits: list[list[float]],
                   *,
                   temperature: float = DEFAULT_TEMPERATURE) -> dict:
    """Decode candidate logits in field/option order into calibrated answers."""
    import torch

    temperature = validate_temperature(temperature)
    if len(candidate_logits) != len(fields):
        raise DecisionOutputError('Decision field mapping is inconsistent')
    answers = {}
    for field, logits in zip(fields, candidate_logits):
        question = questions[field]
        options = answer_options(question)
        values = [option.value for option in options]
        logits = torch.tensor(logits, dtype=torch.float32)
        if logits.numel() != len(values) or not bool(torch.isfinite(logits).all()):
            raise DecisionOutputError('Expected one finite logit per candidate')
        probs = dict(zip(values, torch.softmax(logits, dim=-1).tolist()))
        probs = scale_probabilities(probs, temperature)
        best = argmax(probs)
        kind = question['type']
        answer = dict(type=kind, probabilities=probs, confidence=probs[best])
        if kind == 'noul':
            answer['noul'] = probs['yes']
        elif kind == 'score':
            answer['score'] = sum(float(value) * probs[value] for value in values)
            answer['legend'] = {option.value: option.description for option in options}
        else:
            answer['choice'] = best
        answer['source'] = 'local'
        answer['decision'] = best
        answers[field] = answer
    return answers
