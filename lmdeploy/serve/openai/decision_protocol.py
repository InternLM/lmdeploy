# Copyright (c) OpenMMLab. All rights reserved.
"""Native Intern-Decision and OpenAI Decisions request/response schemas."""
import json
import math
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS
from lmdeploy.serve.openai.protocol import ImageDataFormat

# Question-count limit inherited from the upstream Intern-Decision request validation.
MAX_QUESTIONS = 16


class DecisionRequest(BaseModel):
    """One Jev-style request: ``state``, ``questions`` and optional images.

    Images use the same URL/data URL/path strings or parameter dictionaries as ``/generate``.
    A single reference or a list is accepted; shared multimodal preprocessing validates and loads them.
    """
    model_config = ConfigDict(extra='forbid', strict=True)
    state: Any
    questions: dict[str, dict[str, Any]]
    images: ImageDataFormat = Field(default_factory=list)
    model: str | None = None
    # Training-only metadata is accepted but never enters the inference prompt.
    targets: Any = Field(default=None, exclude=True)
    answer: Any = Field(default=None, exclude=True)
    id: Any = Field(default=None, exclude=True)

    @field_validator('state')
    @classmethod
    def json_state(cls, value):
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('state must contain finite JSON data') from exc
        return value

    @field_validator('questions')
    @classmethod
    def valid_questions(cls, questions):
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f'Supply 1–{MAX_QUESTIONS} questions')
        clean = {}
        for name, question in questions.items():
            if not name:
                raise ValueError('Question names must be nonempty strings')
            kind, criteria = question.get('type'), question.get('criteria')
            if kind not in ('choice', 'score', 'noul'):
                raise ValueError('Question type must be choice, score, or noul')
            if kind == 'choice' and not isinstance(criteria, dict):
                raise ValueError('Choice criteria must be an object')
            if kind == 'score':
                if not isinstance(criteria, (list, dict)):
                    raise ValueError('Score criteria must be a list or object')
                if isinstance(criteria, dict):
                    try:
                        if any(isinstance(key, bool) or not math.isfinite(float(key)) for key in criteria):
                            raise ValueError('Score option keys must be finite numbers')
                    except (TypeError, ValueError) as exc:
                        raise ValueError('Score option keys must be finite numbers') from exc
            if kind != 'noul' and not 1 <= len(criteria) <= len(ANSWER_SYMBOLS):
                raise ValueError(f'Supply 1–{len(ANSWER_SYMBOLS)} options')
            if isinstance(criteria, dict) and any(not isinstance(key, str) for key in criteria):
                raise ValueError('Option keys must be strings')
            clean[name] = {key: question[key] for key in ('type', 'instructions', 'criteria') if key in question}
        return clean


class DecisionResponse(BaseModel):
    request_id: str
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    timing: dict[str, float]
    calibration: dict[str, Any]


class _OpenAIDecisionSchema(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class DecisionQuestionBase(_OpenAIDecisionSchema):
    name: str | None = None
    instructions: str


class DecisionPredicateQuestion(DecisionQuestionBase):
    type: Literal['predicate']


class DecisionChoiceOption(_OpenAIDecisionSchema):
    value: str | bool
    description: str = ''


class DecisionChoiceQuestion(DecisionQuestionBase):
    type: Literal['choice']
    choices: list[DecisionChoiceOption] = Field(min_length=2, max_length=len(ANSWER_SYMBOLS))

    @field_validator('choices')
    @classmethod
    def unique_choices(cls, choices):
        # JSON strings and booleans are distinct choices, including true and "true".
        if len({(type(choice.value), choice.value) for choice in choices}) != len(choices):
            raise ValueError('Choice values must be unique')
        return choices


class DecisionScoreLevel(_OpenAIDecisionSchema):
    label: str
    description: str = ''


class DecisionScoreQuestion(DecisionQuestionBase):
    type: Literal['score']
    levels: list[DecisionScoreLevel] = Field(min_length=2, max_length=10)


class DecisionInputText(_OpenAIDecisionSchema):
    type: Literal['input_text']
    text: str


class DecisionInputImage(_OpenAIDecisionSchema):
    type: Literal['input_image']
    image_url: str
    # Use the local processor's default policy; other detail profiles are not mapped.
    detail: Literal['auto'] | None = None

    @field_validator('image_url')
    @classmethod
    def supported_reference(cls, value):
        url = urlsplit(value)
        if url.scheme in ('http', 'https') and url.netloc:
            return value
        if value.startswith('data:image/') and ';base64,' in value:
            return value
        raise ValueError('image_url must be an HTTP(S) URL or an inline base64 image data URL')


class DecisionInputMessage(_OpenAIDecisionSchema):
    type: Literal['message'] = 'message'
    role: Literal['user']
    content: str | list[Annotated[DecisionInputText | DecisionInputImage, Field(discriminator='type')]] = Field(
        min_length=1)


class OpenAIDecisionRequest(_OpenAIDecisionSchema):
    """OpenAI Decisions text/image input, subject to the local checkpoint's limits."""

    model: str
    input: str | Annotated[list[DecisionInputMessage], Field(min_length=1)]
    questions: list[Annotated[
        DecisionPredicateQuestion | DecisionChoiceQuestion | DecisionScoreQuestion,
        Field(discriminator='type'),
    ]] = Field(min_length=1, max_length=MAX_QUESTIONS)


class DecisionPredicateAnswer(_OpenAIDecisionSchema):
    type: Literal['predicate']
    name: str | None
    probability: float


class DecisionChoiceProbability(_OpenAIDecisionSchema):
    value: str | bool
    probability: float


class DecisionChoiceAnswer(_OpenAIDecisionSchema):
    type: Literal['choice']
    name: str | None
    choice: str | bool
    confidence: float
    probabilities: list[DecisionChoiceProbability]


class DecisionScoreProbability(_OpenAIDecisionSchema):
    value: int
    label: str
    probability: float


class DecisionScoreAnswer(_OpenAIDecisionSchema):
    type: Literal['score']
    name: str | None
    score: float
    confidence: float
    probabilities: list[DecisionScoreProbability]


class DecisionTokenUsage(_OpenAIDecisionSchema):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int]
    output_tokens_details: dict[str, int]


class OpenAIDecisionResponse(_OpenAIDecisionSchema):
    model: str
    answers: list[Annotated[
        DecisionPredicateAnswer | DecisionChoiceAnswer | DecisionScoreAnswer,
        Field(discriminator='type'),
    ]]
    usage: DecisionTokenUsage
