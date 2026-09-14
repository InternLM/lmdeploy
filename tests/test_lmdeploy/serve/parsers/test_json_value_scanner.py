# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json

import pytest

from lmdeploy.serve.parsers.tool_parser.json_value_scanner import JsonValueScanner


def _scan(value: str, chunk_size: int | None) -> str:
    scanner = JsonValueScanner()
    consumed_parts = []
    if chunk_size is None:
        consumed = scanner.feed(value)
        consumed_parts.append(value[:consumed])
    else:
        for start in range(0, len(value), chunk_size):
            chunk = value[start:start + chunk_size]
            consumed = scanner.feed(chunk)
            consumed_parts.append(chunk[:consumed])
    assert scanner.complete
    return ''.join(consumed_parts)


@pytest.mark.parametrize('chunk_size', [None, 1, 4, 8, 16, 256])
def test_scans_many_short_json_strings_in_linear_source_order(chunk_size):
    strings = ['x', '"', '\\', '{', '}', '[', ']', 'plain'] * 2048
    value = json.dumps(strings, separators=(',', ':'))

    assert _scan(value, chunk_size) == value


@pytest.mark.parametrize('chunk_size', [None, 1])
@pytest.mark.parametrize(
    'value',
    [
        '"escaped \\\" quote and \\\\ slash"',
        '{"nested":[{"text":"{}[]"},true,null,1.5]}',
    ],
)
def test_scans_escaped_and_nested_json_values(chunk_size, value):
    assert _scan(value, chunk_size) == value


def test_stops_at_end_of_root_value():
    scanner = JsonValueScanner()
    value = '["x",{"nested":[]}]'

    assert scanner.feed(value + ' trailing') == len(value)
    assert scanner.complete
