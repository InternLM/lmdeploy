import asyncio
import sys

import pytest
from PIL import Image

from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.constants import Modality

multimodal_module = sys.modules[MultimodalProcessor.__module__]


class TestMergeMessageContent:
    """Test suite for merge_message_content function."""

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
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

        assert result['content'] == ''
        assert 'tool_calls' in result

    def test_string_content_unchanged(self):
        """Test that string content remains unchanged."""
        msg = {'role': 'user', 'content': 'Hello, world!'}
        result = MultimodalProcessor.merge_message_content(msg)

        assert result['content'] == 'Hello, world!'
        assert result is msg  # Should return the same object

    def test_single_text_block(self):
        """Test extraction of single text block from list content."""
        msg = {'role': 'user', 'content': [{'type': 'text', 'text': 'Single block'}]}
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

        assert result['content'] == 'Analyze this image:\nWhat do you see?'

    def test_empty_list_content(self):
        """Test that empty list content produces empty string."""
        msg = {'role': 'user', 'content': []}
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

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
        result = MultimodalProcessor.merge_message_content(msg)

        # Should add content field with empty string
        assert 'content' in result
        assert result['content'] == ''
        # Should preserve tool_calls
        assert len(result['tool_calls']) == 1
        assert result['tool_calls'][0]['function']['name'] == 'Bash'


def test_async_parse_multimodal_item_supports_new_value_encodings(monkeypatch):
    """Test parsing direct-valued and dict-valued multimodal content."""
    image = Image.new('RGB', (1, 1))
    load_calls = []

    class FakeVideoMediaIO:

        def __init__(self, image_io=None, **kwargs):
            self.image_io = image_io
            self.kwargs = kwargs

    def fake_load_from_url(data_src, media_io):
        load_calls.append((data_src, type(media_io).__name__))
        if isinstance(media_io, FakeVideoMediaIO):
            return f'loaded:{data_src}', {'duration': 2}
        return f'loaded:{data_src}'

    monkeypatch.setattr(multimodal_module, 'VideoMediaIO', FakeVideoMediaIO)
    monkeypatch.setattr(multimodal_module, 'load_from_url', fake_load_from_url)

    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': 'describe'
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'http://example.com/a.png',
                    'detail': 'low'
                }
            },
            {
                'type': 'image',
                'image': '/tmp/b.png',
                'detail': 'high'
            },
            {
                'type': 'image',
                'image': image,
                'source': 'camera'
            },
            {
                'type': 'image_data',
                'image_data': {
                    'data': image,
                    'detail': 'auto'
                }
            },
            {
                'type': 'video',
                'video': 'file:///tmp/a.mp4',
                'fps': 1
            },
            {
                'type': 'time_series',
                'time_series': {
                    'url': 'file:///tmp/a.npy',
                    'sr': 16000
                }
            },
        ]
    }]

    parsed = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))
    content = parsed[0]['content']

    assert content[0] == {'type': 'text', 'text': 'describe'}
    assert content[1] == {'type': Modality.IMAGE, 'data': 'loaded:http://example.com/a.png', 'detail': 'low'}
    assert content[2] == {'type': Modality.IMAGE, 'data': 'loaded:/tmp/b.png', 'detail': 'high'}
    assert content[3] == {'type': Modality.IMAGE, 'data': image, 'source': 'camera'}
    assert content[3]['data'] is image
    assert content[4] == {'type': Modality.IMAGE, 'data': image, 'detail': 'auto'}
    assert content[4]['data'] is image
    assert content[5] == {
        'type': Modality.VIDEO,
        'data': 'loaded:file:///tmp/a.mp4',
        'fps': 1,
        'video_metadata': {
            'duration': 2
        }
    }
    assert content[6] == {'type': Modality.TIME_SERIES, 'data': 'loaded:file:///tmp/a.npy', 'sr': 16000}
    assert load_calls == [
        ('http://example.com/a.png', 'ImageMediaIO'),
        ('/tmp/b.png', 'ImageMediaIO'),
        ('file:///tmp/a.mp4', 'FakeVideoMediaIO'),
        ('file:///tmp/a.npy', 'TimeSeriesMediaIO'),
    ]


@pytest.mark.parametrize('item', [{'type': 'image_url'}, {'type': 'image', 'image': {}},
                                  {'type': 'time_series', 'time_series': {'sr': 16000}}])
def test_async_parse_multimodal_item_rejects_missing_payload(item):
    """Test missing multimodal payloads fail with a clear error."""
    messages = [{'role': 'user', 'content': [item]}]

    with pytest.raises(ValueError, match='Expected .* direct value or a dict containing "url" or "data"'):
        asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))


def test_async_parse_multimodal_item_rejects_unknown_type():
    """Test unknown multimodal item types still fail explicitly."""
    messages = [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'file:///tmp/a.wav'}]}]

    with pytest.raises(NotImplementedError, match='unknown type: audio'):
        asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))


def test_has_multimodal_input_detects_all_supported_types():
    """Test multimodal detection includes every supported item type."""
    processor = MultimodalProcessor(tokenizer=None, chat_template=None)

    for item_type in ['image_url', 'image_data', 'image', 'video_url', 'video', 'time_series_url', 'time_series']:
        assert processor._has_multimodal_input([{'role': 'user', 'content': [{'type': item_type}]}])
    assert not processor._has_multimodal_input([{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}])


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
    result = MultimodalProcessor.merge_message_content(msg)
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

    processed = [MultimodalProcessor.merge_message_content(msg) for msg in messages]

    # Verify all messages have content field
    assert all('content' in msg for msg in processed)

    # Verify content values
    assert processed[0]['content'] == 'Hello'
    assert processed[1]['content'] == ''
    assert processed[2]['content'] == 'Block 1\nBlock 2'

    # Should pass model.py assertion
    assert all(isinstance(m, dict) and 'role' in m and 'content' in m for m in processed)
