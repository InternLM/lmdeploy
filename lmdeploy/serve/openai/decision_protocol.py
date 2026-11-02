# Copyright (c) OpenMMLab. All rights reserved.
"""Systemone and OpenAI Decisions HTTP request/response schemas."""
import json
import math
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS
from lmdeploy.serve.parsers.decision_assets import ImageAsset

# Question-count limit inherited from the upstream Intern-Decision request validation.
MAX_QUESTIONS = 16


# Systemone schemas follow intern-decision-inference a8fe089/request_validation.py.
def _systemone_content(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    text.encode('utf-8')
    if '<decision>' in text:
        raise ValueError('Reserved decision marker is not allowed.')
    return value


def _systemone_name(value):
    if not value.strip():
        raise ValueError('Must not be blank.')
    return _systemone_content(value)


_SystemoneContent = Annotated[str | dict[str, JsonValue] | list[JsonValue], AfterValidator(_systemone_content)]
_SystemoneName = Annotated[str, AfterValidator(_systemone_name)]


class _SystemoneQuestion(BaseModel):
    model_config = ConfigDict(strict=True, extra='ignore', allow_inf_nan=False)
    instructions: _SystemoneContent | None = None


class SystemoneChoiceQuestion(_SystemoneQuestion):
    type: Literal['choice']
    criteria: dict[_SystemoneName, _SystemoneContent | None] = Field(min_length=1, max_length=62)


class SystemoneScoreQuestion(_SystemoneQuestion):
    type: Literal['score']
    criteria: list[_SystemoneContent] | dict[str, _SystemoneContent] = Field(min_length=1, max_length=62)

    @field_validator('criteria')
    @classmethod
    def numeric_levels(cls, value):
        if isinstance(value, dict):
            numbers = [float(key) for key in value]
            if not all(math.isfinite(n) for n in numbers) or len(set(numbers)) != len(numbers):
                raise ValueError('Level keys must be distinct finite numeric strings.')
        return value


class SystemoneNoulQuestion(_SystemoneQuestion):
    type: Literal['noul']
    criteria: dict[str, _SystemoneContent | None] | None = None

    @field_validator('criteria')
    @classmethod
    def outcome_names(cls, value):
        if value is None:
            return value
        aliases = {'yes': True, 'true': True, '1': True, 'no': False, 'false': False, '0': False}
        keys = [key.lower() for key in value]
        if any(key not in aliases for key in keys):
            raise ValueError('Use yes/no, true/false, or 1/0 keys.')
        if len({aliases[key] for key in keys}) != len(keys):
            raise ValueError('Supply at most one description for each yes/no outcome.')
        return value

    @model_validator(mode='after')
    def described_question(self):
        if self.instructions is None and not any(v is not None for v in (self.criteria or {}).values()):
            raise ValidationError.from_exception_data(type(self).__name__, [{
                'type': 'value_error', 'loc': ('instructions',), 'input': None,
                'ctx': {'error': ValueError('Provide instructions or at least one non-null outcome description.')}}])
        return self


class SystemoneRequest(BaseModel):
    """Systemone assets protocol; model identity is checked against the loaded engine."""

    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)
    state: _SystemoneContent
    questions: dict[_SystemoneName, Annotated[
        SystemoneChoiceQuestion | SystemoneScoreQuestion | SystemoneNoulQuestion,
        Field(discriminator='type'),
    ]] = Field(min_length=1, max_length=MAX_QUESTIONS)
    assets: dict[str, ImageAsset] | None = None
    model: str = ''


class SystemoneResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]


class _OpenAIDecisionSchema(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class DecisionQuestionBase(_OpenAIDecisionSchema):
    name: str | None = None
    instructions: str

    @field_validator('name')
    @classmethod
    def nonblank_name(cls, value):
        if value is not None and not value.strip():
            raise ValueError('Question names must not be blank')
        return value


class DecisionPredicateQuestion(DecisionQuestionBase):
    type: Literal['predicate']


class DecisionChoiceOption(_OpenAIDecisionSchema):
    value: str | bool
    description: str = ''

    @field_validator('value')
    @classmethod
    def nonblank_value(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError('Choice values must not be blank')
        return value


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
    levels: list[DecisionScoreLevel] = Field(min_length=2, max_length=len(ANSWER_SYMBOLS))


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
        if value.lower().startswith('data:image/') and 'base64' in value.partition(',')[0].lower().split(';'):
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
    input: Annotated[str, Field(min_length=1)] | Annotated[list[DecisionInputMessage], Field(min_length=1)]
    questions: list[Annotated[
        DecisionPredicateQuestion | DecisionChoiceQuestion | DecisionScoreQuestion,
        Field(discriminator='type'),
    ]] = Field(min_length=1, max_length=MAX_QUESTIONS)
    # Caller-supplied metadata; excluded from the scoring prompt by the parser.
    safety_identifier: str | None = Field(default=None, max_length=64)

    @field_validator('input')
    @classmethod
    def image_count(cls, value):
        if not isinstance(value, str):
            count = sum(isinstance(part, DecisionInputImage) for message in value
                        if isinstance(message.content, list) for part in message.content)
            if count > 8:
                raise ValueError('At most 8 images are supported per request.')
        return value

    @field_validator('questions')
    @classmethod
    def unique_names(cls, questions):
        seen, errors = set(), []
        for index, question in enumerate(questions):
            if question.name is None:
                continue
            if question.name in seen:
                errors.append(dict(type='value_error', loc=(index, 'name'), input=question.name,
                                   ctx={'error': ValueError('Supplied question names must be unique')}))
            seen.add(question.name)
        if errors:
            raise ValidationError.from_exception_data(cls.__name__, errors)
        return questions


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
