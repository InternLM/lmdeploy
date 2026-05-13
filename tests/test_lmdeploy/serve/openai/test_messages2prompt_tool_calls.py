from lmdeploy.model import BaseChatTemplate


def test_messages2prompt_renders_assistant_tool_calls():
    """When an assistant message has tool_calls, they should be rendered in the
    prompt, not silently discarded."""
    template = BaseChatTemplate(
        meta_instruction='You are helpful.',
        system='<|sys|>',
        eosys='</|sys|>',
        user='<|usr|>',
        eoh='</|usr|>',
        assistant='<|asst|>',
        eoa='</|asst|>',
        tool='<|tool|>',
        eotool='</|tool|>',
    )
    messages = [
        {'role': 'user', 'content': 'What is the weather?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "NYC"}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Sunny, 72F'},
    ]
    prompt = template.messages2prompt(messages, sequence_start=True)
    assert 'get_weather' in prompt
    assert 'Sunny, 72F' in prompt


def test_messages2prompt_renders_tool_call_id():
    """When a tool message has tool_call_id, it should appear in the prompt."""
    template = BaseChatTemplate(
        meta_instruction='You are helpful.',
        system='<|sys|>',
        eosys='</|sys|>',
        user='<|usr|>',
        eoh='</|usr|>',
        assistant='<|asst|>',
        eoa='</|asst|>',
        tool='<|tool|>',
        eotool='</|tool|>',
    )
    messages = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_abc', 'type': 'function', 'function': {'name': 'greet', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'call_abc', 'content': 'Hi there!'},
    ]
    prompt = template.messages2prompt(messages, sequence_start=True)
    assert 'call_abc' in prompt


def test_merge_message_content_preserves_tool_calls_when_content_is_none():
    """merge_message_content should preserve tool_calls when content is
    None."""
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    msg = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call_1', 'type': 'function', 'function': {'name': 'test', 'arguments': '{}'}},
    ]}
    result = MultimodalProcessor.merge_message_content(msg)
    assert 'tool_calls' in result
    assert len(result['tool_calls']) == 1
    assert result['content'] == ''


def test_messages2prompt_without_tool_calls_unchanged():
    """Messages without tool_calls should produce the same prompt as before."""
    template = BaseChatTemplate(
        meta_instruction='You are helpful.',
        system='<|sys|>',
        eosys='</|sys|>',
        user='<|usr|>',
        eoh='</|usr|>',
        assistant='<|asst|>',
        eoa='</|asst|>',
    )
    messages = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there!'},
    ]
    prompt = template.messages2prompt(messages, sequence_start=True)
    # Should contain user content and assistant content normally
    assert 'Hello' in prompt
    assert 'Hi there!' in prompt
