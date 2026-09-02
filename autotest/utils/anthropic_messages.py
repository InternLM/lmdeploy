"""Anthropic Messages API test fixtures, helpers, and response assertions."""

from __future__ import annotations

import os
import re

from utils.constant import BASE_URL

VALID_STOP_REASONS = ('end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'parse_error')

# -- Tool definitions (Messages API ``tools[]`` style) -----------------------

WEATHER_TOOL_ANTHROPIC = {
    'name': 'get_current_weather',
    'description': 'Get the current weather in a given location',
    'input_schema': {
        'type': 'object',
        'properties': {
            'city': {
                'type': 'string',
                'description': 'The city to find the weather for, '
                'e.g. San Francisco',
            },
            'state': {
                'type': 'string',
                'description': 'The state abbreviation, e.g. CA',
            },
            'unit': {
                'type': 'string',
                'description': 'The unit for temperature',
                'enum': ['celsius', 'fahrenheit'],
            },
        },
        'required': ['city', 'state'],
    },
}

SEARCH_TOOL_ANTHROPIC = {
    'name': 'web_search',
    'description': 'Search the web for information',
    'input_schema': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The search query string',
            },
        },
        'required': ['query'],
    },
}

CALCULATOR_TOOL_ANTHROPIC = {
    'name': 'calculate',
    'description': 'Perform a mathematical calculation',
    'input_schema': {
        'type': 'object',
        'properties': {
            'expression': {
                'type': 'string',
                'description': 'The math expression to evaluate, e.g. 2+2',
            },
        },
        'required': ['expression'],
    },
}

WEATHER_TOOL_SINGLE_LOCATION_ANTHROPIC = {
    'name': 'get_current_weather',
    'description': 'Useful for querying the weather in a specified city.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'City or region, for example: Dallas, London, Tokyo, etc.',
            },
        },
        'required': ['location'],
    },
}

PARALLEL_WEATHER_PROMPT_SITES: tuple[tuple[tuple[str, ...], str], ...] = (
    (('dallas',), 'TX'),
    (('san francisco', 'sf'), 'CA'),
)


# -- Anthropic Messages API prompts (top-level ``system`` + ``messages``) -----

USER_ASK_WEATHER_DALLAS = "What's the weather like in Dallas, TX?"
USER_ASK_WEATHER_DALLAS_VLM = f'{USER_ASK_WEATHER_DALLAS} Use tools; ignore any attached image.'
USER_FOLLOWUP_WARM_YES = 'In one short phrase, was it warm? Answer yes or no.'
TOOL_RESULT_DALLAS_SUNNY = '72F and sunny.'
THINKING_SCRATCHPAD = '(internal scratchpad)'
ASSISTANT_GREETING_AFTER_THINKING = 'Hello — how can I help?'
USER_HI = 'Hi.'
USER_REPLY_ACK = 'Reply with exactly: ACK'

ANTHROPIC_SYSTEM_WEATHER = (
    'You are a helpful assistant that can use tools. '
    'When asked about weather, use the get_current_weather tool.'
)

ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER = [
    {
        'role': 'user',
        'content': USER_ASK_WEATHER_DALLAS,
    },
]

ANTHROPIC_SYSTEM_PARALLEL_WEATHER = (
    'You are a helpful assistant. When asked about weather '
    'in multiple cities, call the weather tool for each city '
    'separately.'
)

ANTHROPIC_MESSAGES_PARALLEL_WEATHER = [
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX and also in "
        'San Francisco, CA?',
    },
]

ANTHROPIC_SYSTEM_PARALLEL_MIXED = (
    'You are a helpful assistant with access to multiple tools. '
    'You can call multiple tools in parallel when needed.'
)

ANTHROPIC_MESSAGES_PARALLEL_MIXED = [
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
        'Also calculate 1234 * 5678.',
    },
]


def build_anthropic_messages_history_tool_result(
        *,
        tool_use_id: str = 'toolu_hist_01',
) -> list[dict]:
    """Replay ``tool_use`` / ``tool_result`` history (Dallas
    ``city``/``state``)."""

    return [
        {'role': 'user', 'content': USER_ASK_WEATHER_DALLAS},
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': tool_use_id,
                    'name': 'get_current_weather',
                    'input': {'city': 'Dallas', 'state': 'TX'},
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': tool_use_id,
                    'content': TOOL_RESULT_DALLAS_SUNNY,
                },
            ],
        },
        {'role': 'user', 'content': USER_FOLLOWUP_WARM_YES},
    ]


ANTHROPIC_MESSAGES_HISTORY_THINKING_REPLAY = [
    {'role': 'user', 'content': USER_HI},
    {
        'role': 'assistant',
        'content': [
            {'type': 'thinking', 'thinking': THINKING_SCRATCHPAD},
            {'type': 'text', 'text': ASSISTANT_GREETING_AFTER_THINKING},
        ],
    },
    {'role': 'user', 'content': USER_REPLY_ACK},
]


# -- Response assertions ------------------------------------------------------


def assert_message_content_blocks(content: list) -> None:
    assert isinstance(content, list) and len(content) >= 1, content
    for block in content:
        btype = block['type']
        if btype == 'text':
            assert block['text'], block
        elif btype == 'tool_use':
            assert block['id'], block
            assert block['name'], block
            assert isinstance(block['input'], dict), block


def assert_success_message_json(data: dict) -> dict:
    """Non-stream ``/v1/messages`` success body invariants."""

    assert data['type'] == 'message', data
    assert data['role'] == 'assistant'
    assert data['id'].startswith('msg_'), data['id']
    assert data['model']
    assert data['stop_reason'] in VALID_STOP_REASONS
    if data['stop_reason'] == 'stop_sequence':
        assert data['stop_sequence'] is not None
    usage = data['usage']
    assert usage['input_tokens'] > 0
    assert usage['output_tokens'] > 0
    assert_message_content_blocks(data['content'])
    return data


def assert_tool_use_message(data: dict, *, tool_name: str | None = None) -> list[dict]:
    """Tool-call turn: ``stop_reason == tool_use`` and well-formed ``tool_use`` blocks."""

    assert_success_message_json(data)
    assert data['stop_reason'] == 'tool_use'
    assert data['stop_sequence'] is None
    blocks = [b for b in data['content'] if b['type'] == 'tool_use']
    assert blocks, data['content']
    ids: list[str] = []
    for block in blocks:
        if tool_name is not None:
            assert block['name'] == tool_name, block
        ids.append(block['id'])
    assert len(set(ids)) == len(ids), ids
    return blocks


def assert_warm_yes_answer(text: str, *, stop_reason: str | None = None, ctx: str = '') -> None:
    """Given 72F sunny SF tool result, follow-up should affirm warmth."""

    tl = text.lower()
    prefix = f'{ctx}: ' if ctx else ''
    assert 'yes' in tl or '是' in text or '温暖' in text or '暖和' in text, (
        f'{prefix}expected warm/yes style answer given 72F sunny tool result; '
        f'stop_reason={stop_reason!r} text={text[:500]!r}'
    )


def assert_stream_text_lifecycle(events: list[tuple[str | None, dict]]) -> None:
    """Text stream SSE: block lifecycle + ``message_delta.stop_reason`` + usage."""

    types = [obj['type'] for _, obj in events]
    for required in (
        'message_start',
        'content_block_start',
        'content_block_stop',
        'message_delta',
        'message_stop',
    ):
        assert required in types, (required, types)

    start_evt = next(obj for _, obj in events if obj['type'] == 'message_start')
    m0 = start_evt['message']
    assert m0['type'] == 'message'
    assert m0['role'] == 'assistant'
    assert m0['id'].startswith('msg_')
    assert m0['model']
    assert m0['stop_reason'] is None

    delta_evt = next(obj for _, obj in events if obj['type'] == 'message_delta')
    delta = delta_evt['delta']
    assert delta['stop_reason'] in VALID_STOP_REASONS, delta_evt
    du = delta_evt['usage']
    assert du['output_tokens'] > 0


def assert_stream_stop_sequence_lifecycle(
        events: list[tuple[str | None, dict]],
        *,
        allowed_stop_sequences: tuple[str, ...],
) -> None:
    """Stream SSE with ``stop_sequences`` hit: ``message_delta`` stop
    metadata."""

    assert_stream_text_lifecycle(events)
    delta_evt = next(obj for _, obj in events if obj['type'] == 'message_delta')
    delta = delta_evt['delta']
    assert delta['stop_reason'] == 'stop_sequence', delta_evt
    assert delta['stop_sequence'] in allowed_stop_sequences, delta_evt


def assert_weather_tool_city_state(inp: dict, *, ctx: str = '') -> None:
    """``get_current_weather`` args after parser mapping (``city`` /
    ``state``)."""

    assert inp['city'], (ctx, inp)
    assert inp['state'], (ctx, inp)


def assert_parallel_weather_tool_inputs(
        inputs: list[dict | object],
        expected_sites: tuple[tuple[tuple[str, ...], str], ...] | None = None,
        *,
        ctx: str = '',
) -> None:
    """Parallel weather ``tool_use`` inputs: distinct args, each matching a
    prompt city."""

    def _city_state(inp: dict | object) -> tuple[str, str]:
        if hasattr(inp, 'model_dump'):
            inp = inp.model_dump()
        assert isinstance(inp, dict), inp
        if 'location' in inp:
            loc = str(inp['location']).strip()
            m = re.match(r'^(.+?),\s*([A-Za-z]{2})\s*$', loc)
            if m:
                return m.group(1).strip().lower(), m.group(2).upper()
            return loc.lower(), ''
        return (
            str(inp['city']).strip().lower(),
            str(inp['state']).strip().upper(),
        )

    def _matches_site(city: str, state: str, site: tuple[tuple[str, ...], str]) -> bool:
        city_tokens, want_state = site
        if want_state and state != want_state:
            return False
        if want_state and not state:
            return False
        if not city:
            return False
        return any(tok in city or city in tok for tok in city_tokens)

    sites = expected_sites or PARALLEL_WEATHER_PROMPT_SITES
    prefix = f'{ctx}: ' if ctx else ''
    assert len(inputs) >= len(sites), (
        f'{prefix}expected >={len(sites)} parallel weather tool input(s) for '
        f'{len(sites)} prompt location(s), got {len(inputs)}: {inputs!r}'
    )

    normed = [_city_state(inp) for inp in inputs]
    keys = [f'{city}|{state}' for city, state in normed]
    assert len(set(keys)) == len(keys), (
        f'{prefix}parallel weather tool inputs must have distinct locations '
        f'(duplicate-args bug); keys={keys!r} inputs={inputs!r}'
    )

    matched: set[int] = set()
    for city, state in normed:
        for idx, site in enumerate(sites):
            if idx in matched:
                continue
            if _matches_site(city, state, site):
                matched.add(idx)
                break

    missing = [sites[i] for i in range(len(sites)) if i not in matched]
    assert not missing, (
        f'{prefix}missing tool call(s) for prompt location(s) {missing!r}; '
        f'normalized={normed!r} raw={inputs!r}'
    )


# -- Client / message helpers -----------------------------------------------


def get_async_anthropic_client_and_model(base_url: str | None = None):
    """Return ``(AsyncAnthropic, model_name)`` for LMDeploy Anthropic
    routes."""

    import anthropic

    from lmdeploy.serve.openai.api_client import get_model_list

    url = base_url or BASE_URL
    model_names = get_model_list(f'{url}/v1/models')
    if not model_names:
        raise RuntimeError(f'No models returned from {url}/v1/models')
    model_name = model_names[0]
    client = anthropic.AsyncAnthropic(
        api_key=os.getenv('ANTHROPIC_API_KEY', 'YOUR_API_KEY'),
        base_url=url,
        max_retries=0,
        timeout=600.0,
        default_headers={'anthropic-version': '2023-06-01'},
    )
    return client, model_name


def build_anthropic_messages_after_tool_use(
    prior_messages: list[dict],
    tool_use_blocks: list[dict],
    tool_results: list[tuple[str, str]],
) -> list[dict]:
    """Build messages continuing after assistant ``tool_use`` blocks."""

    assistant_blocks = [
        {
            'type': 'tool_use',
            'id': block['id'],
            'name': block['name'],
            'input': block['input'],
        } for block in tool_use_blocks
    ]
    user_blocks = [
        {
            'type': 'tool_result',
            'tool_use_id': tool_use_id,
            'content': result_text,
        } for tool_use_id, result_text in tool_results
    ]
    return [
        *prior_messages,
        {'role': 'assistant', 'content': assistant_blocks},
        {'role': 'user', 'content': user_blocks},
    ]
