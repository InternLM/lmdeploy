

def test_decode_tool_incremental_json_id_only_on_first_chunk():
    """When streaming a tool call, id should appear only on the name-delta
    chunk, not on subsequent argument-delta chunks."""
    from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser

    class TestToolParser(ToolParser):
        def get_tool_open_tag(self): return None
        def get_tool_close_tag(self): return None
        def get_tool_payload_format(self): return 'json'
        def decode_tool_incremental(self, added_text, *, final): return []
        def parse_tool_call_complete(self, payload): return None

    parser = TestToolParser(tokenizer=None)
    parser.start_tool_call()

    # Step 1: feed partial JSON with name
    deltas = parser._decode_tool_incremental_json('{"name": "get_weather", ', final=False)
    assert len(deltas) == 1
    name_delta = deltas[0]
    assert name_delta.function.name == 'get_weather'
    assert name_delta.id is not None
    assert name_delta.id.startswith('chatcmpl-tool-')
    assert name_delta.type == 'function'

    # Step 2: feed final chunk with arguments
    deltas = parser._decode_tool_incremental_json('"arguments": {"city": "NYC"}}', final=True)
    assert len(deltas) == 1
    args_delta = deltas[0]
    assert args_delta.function.arguments is not None
    assert args_delta.id is None  # id MUST be None on argument-delta
    assert args_delta.type is None
