"""Unit tests for _merge_message_content function.

This module tests the content field normalization logic that ensures compatibility with vLLM's multimodal content
handling.
"""
from typing import Dict

import pytest


def _merge_message_content(msg: Dict) -> Dict:
    """Merge multimodal content blocks and ensure content field exists.

    This is a copy of the implementation from lmdeploy.serve.async_engine
    to avoid import dependency issues in tests.

    This function normalizes message content to match vLLM's behavior:
    1. Missing content field -> add content='' (empty string)
    2. None content -> convert to content='' (empty string)
    3. String content -> return as-is
    4. List content (multimodal) -> merge all text blocks with newline separator

    Args:
        msg: A message dict with 'role' and optionally 'content' field

    Returns:
        A message dict with 'content' field guaranteed to exist
    """
    # If content is missing or None, convert to empty string (matches vLLM behavior)
    # This prevents Jinja2 template errors when rendering chat templates
    if 'content' not in msg or msg['content'] is None:
        result = dict(msg)
        result['content'] = ''
        return result

    # If content is already a string, return as-is
    if isinstance(msg['content'], str):
        return msg

    # If content is a list, merge all text blocks into a single string
    # This matches vLLM's behavior: text_prompt = "\n".join(texts)
    content_parts = []
    for block in msg['content']:
        if isinstance(block, dict) and block.get('type') == 'text':
            content_parts.append(block.get('text', ''))
    merged_content = '\n'.join(content_parts)

    # Preserve all other fields in the message (e.g., tool_calls)
    result = dict(msg)
    result['content'] = merged_content
    return result


class TestMergeMessageContent:
    """Test suite for _merge_message_content function."""

    def test_missing_content_field(self):
        """Test that missing content field is added with empty string.

        This case occurs with assistant messages that only have tool_calls.
        """
        msg = {
            'role':
            'assistant',
            'tool_calls': [{
                'id': 'chatcmpl-tool-123',
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': '{"city": "Paris"}'
                }
            }]
        }
        result = _merge_message_content(msg)

        assert 'content' in result
        assert result['content'] == ''
        assert 'tool_calls' in result
        assert result['tool_calls'] == msg['tool_calls']

    def test_explicit_none_content(self):
        """Test that explicit None content is converted to empty string.

        This matches vLLM's behavior: None → [] → ''.join([]) → ''.
        """
        msg = {
            'role':
            'assistant',
            'content':
            None,
            'tool_calls': [{
                'id': 'chatcmpl-tool-456',
                'type': 'function',
                'function': {
                    'name': 'Bash',
                    'arguments': '{"command": "ls"}'
                }
            }]
        }
        result = _merge_message_content(msg)

        assert result['content'] == ''
        assert 'tool_calls' in result

    def test_string_content_unchanged(self):
        """Test that string content remains unchanged."""
        msg = {'role': 'user', 'content': 'Hello, world!'}
        result = _merge_message_content(msg)

        assert result['content'] == 'Hello, world!'
        assert result is msg  # Should return the same object

    def test_single_text_block(self):
        """Test extraction of single text block from list content."""
        msg = {'role': 'user', 'content': [{'type': 'text', 'text': 'Single block'}]}
        result = _merge_message_content(msg)

        assert result['content'] == 'Single block'

    def test_multiple_text_blocks_newline_join(self):
        """Test that multiple text blocks are merged with newline separator.

        This matches vLLM's behavior: text_prompt = "\\n".join(texts)
        """
        msg = {
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'First block'
            }, {
                'type': 'text',
                'text': 'Second block'
            }, {
                'type': 'text',
                'text': 'Third block'
            }]
        }
        result = _merge_message_content(msg)

        assert result['content'] == 'First block\nSecond block\nThird block'

    def test_mixed_content_types(self):
        """Test that only text blocks are extracted from mixed content.

        Non-text blocks (like image_url) should be filtered out.
        """
        msg = {
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'Analyze this image:'
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': 'http://example.com/img.jpg'
                }
            }, {
                'type': 'text',
                'text': 'What do you see?'
            }]
        }
        result = _merge_message_content(msg)

        assert result['content'] == 'Analyze this image:\nWhat do you see?'

    def test_empty_list_content(self):
        """Test that empty list content produces empty string."""
        msg = {'role': 'user', 'content': []}
        result = _merge_message_content(msg)

        assert result['content'] == ''

    def test_list_with_non_text_blocks_only(self):
        """Test content with only non-text blocks (e.g., only images)."""
        msg = {
            'role':
            'user',
            'content': [{
                'type': 'image_url',
                'image_url': {
                    'url': 'http://example.com/img1.jpg'
                }
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': 'http://example.com/img2.jpg'
                }
            }]
        }
        result = _merge_message_content(msg)

        assert result['content'] == ''

    def test_preserve_all_message_fields(self):
        """Test that all message fields are preserved during content merge."""
        msg = {
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Response'
            }],
            'tool_calls': [{
                'id': '123',
                'type': 'function'
            }],
            'name': 'assistant',
            'custom_field': 'custom_value'
        }
        result = _merge_message_content(msg)

        assert result['content'] == 'Response'
        assert result['tool_calls'] == msg['tool_calls']
        assert result['name'] == 'assistant'
        assert result['custom_field'] == 'custom_value'
        assert set(result.keys()) == set(msg.keys())

    def test_text_block_with_missing_text_field(self):
        """Test handling of text block without 'text' field."""
        msg = {
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'First'
                },
                {
                    'type': 'text'
                },  # Missing 'text' field
                {
                    'type': 'text',
                    'text': 'Third'
                }
            ]
        }
        result = _merge_message_content(msg)

        # Missing text field should be treated as empty string
        assert result['content'] == 'First\n\nThird'

    def test_gpt_oss_tool_call_scenario(self):
        """Test the specific GPT-OSS tool call scenario from the bug report.

        When GPT-OSS assistant returns tool calls, content is empty/missing.
        """
        msg = {
            'role':
            'assistant',
            'tool_calls': [{
                'id': 'chatcmpl-tool-UK9rkwzMAyxt9DxBezk7E2',
                'type': 'function',
                'function': {
                    'name': 'Bash',
                    'arguments': '{"command": "ls", "description": "List files in current directory"}'
                }
            }]
        }
        result = _merge_message_content(msg)

        # Should add content field with empty string
        assert 'content' in result
        assert result['content'] == ''
        # Should preserve tool_calls
        assert len(result['tool_calls']) == 1
        assert result['tool_calls'][0]['function']['name'] == 'Bash'


@pytest.mark.parametrize(
    'msg,expected_content',
    [
        # Basic cases
        ({
            'role': 'user',
            'content': 'test'
        }, 'test'),
        ({
            'role': 'user',
            'content': None
        }, ''),
        ({
            'role': 'assistant'
        }, ''),

        # List content cases
        ({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': 'a'
            }]
        }, 'a'),
        ({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': 'a'
            }, {
                'type': 'text',
                'text': 'b'
            }]
        }, 'a\nb'),

        # Empty cases
        ({
            'role': 'user',
            'content': []
        }, ''),
        ({
            'role': 'user',
            'content': [{
                'type': 'image_url'
            }]
        }, ''),
    ])
def test_merge_message_content_parametrized(msg, expected_content):
    """Parametrized test for various message content scenarios."""
    result = _merge_message_content(msg)
    assert result['content'] == expected_content


def test_batch_message_processing():
    """Test processing multiple messages in a batch (typical usage pattern)."""
    messages = [{
        'role': 'user',
        'content': 'Hello'
    }, {
        'role': 'assistant',
        'tool_calls': [{
            'id': '123',
            'type': 'function'
        }]
    }, {
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': 'Block 1'
        }, {
            'type': 'text',
            'text': 'Block 2'
        }]
    }]

    processed = [_merge_message_content(msg) for msg in messages]

    # Verify all messages have content field
    assert all('content' in msg for msg in processed)

    # Verify content values
    assert processed[0]['content'] == 'Hello'
    assert processed[1]['content'] == ''
    assert processed[2]['content'] == 'Block 1\nBlock 2'

    # Should pass model.py assertion
    assert all(isinstance(m, dict) and 'role' in m and 'content' in m for m in processed)
