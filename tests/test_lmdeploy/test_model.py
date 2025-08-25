import pytest

from lmdeploy.model import MODELS, best_match_model


@pytest.mark.parametrize(
    'model_path_and_name',
    [
        ('internlm/internlm-chat-7b', ['internlm']),
        ('internlm/internlm2-1_8b', ['base']),
        ('Qwen/Qwen-7B-Chat', ['qwen']),
        ('Qwen/Qwen2.5-7B-Instruct', ['hf']),
        # ('Qwen/Qwen2.5-VL-7B-Instruct', ['qwen2d5-vl']),
        ('Qwen/Qwen3-32B', ['hf']),
        ('Qwen/Qwen3-235B-A22B', ['hf']),
        ('codellama/CodeLlama-7b-hf', ['codellama']),
        # ('meta-llama/Llama-2-7b-chat-hf', ['llama2']),
        ('THUDM/chatglm2-6b', ['chatglm']),
        ('01-ai/Yi-34B-Chat', ['hf']),
        ('01-ai/Yi-6B-Chat', ['hf']),
        ('codellama/CodeLlama-34b-Instruct-hf', ['codellama']),
        ('deepseek-ai/deepseek-coder-6.7b-instruct', ['hf']),
        ('deepseek-ai/deepseek-vl-7b-chat', ['deepseek-vl']),
        ('deepseek-ai/deepseek-moe-16b-chat', ['hf']),
        ('internlm/internlm-xcomposer2-4khd-7b', ['hf']),
        ('internlm/internlm-xcomposer2d5-7b', ['hf']),
        # ('OpenGVLab/InternVL2_5-1B', ['internvl2_5']),
        # ('OpenGVLab/InternVL3-1B', ['internvl2_5']),
    ])
def test_best_match_model(model_path_and_name):
    deduced_name = best_match_model(model_path_and_name[0])
    if deduced_name is not None:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'
    else:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'


# @pytest.mark.parametrize('model_name', ['llama2', 'base', 'yi', 'qwen-7b', 'vicuna'])
# @pytest.mark.parametrize('meta_instruction', ['[fake meta_instruction]'])
# def test_model_config(model_name, meta_instruction):
#     from lmdeploy.model import ChatTemplateConfig
#     chat_template = ChatTemplateConfig(model_name, meta_instruction=meta_instruction).chat_template
#     prompt = chat_template.get_prompt('')
#     if model_name == 'base':
#         assert prompt == ''
#     else:
#         assert meta_instruction in prompt


def test_base_model():
    model = MODELS.get('internlm')(capability='completion')
    assert model.capability == 'completion'
    assert model.get_prompt('hi') == 'hi'
    assert model.messages2prompt('test') == 'test'


def test_vicuna():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('vicuna')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt

    model = MODELS.get('vicuna')(capability='chat', system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('vicuna')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


def test_prefix_response():
    model = MODELS.get('hf')(model_path='Qwen/Qwen3-8B')
    messages = [dict(role='assistant', content='prefix test')]
    prompt = model.messages2prompt(messages)
    assert prompt[-len('prefix test'):] == 'prefix test'


def test_internlm_chat():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('internlm')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is not None
    assert model.system == '<|System|>:'

    model = MODELS.get('internlm')(capability='chat', system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('internlm')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


def test_internlm_tool_call():
    messages = []
    messages.append({
        'role':
        'system',
        'name':
        'plugin',
        'content':
        '[{"description": "Compute the sum of two numbers", "name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}, {"description": "Calculate the product of two numbers", "name": "mul", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}]'  # noqa
    })
    messages.append({'role': 'user', 'content': 'Compute (3+5)*2'})
    messages.append({
        'content':
        '(3+5)*2 = 8*2 =',
        'role':
        'assistant',
        'tool_calls': [{
            'id': '1',
            'function': {
                'arguments': '{"a": 8, "b": 2}',
                'name': 'mul'
            },
            'type': 'function'
        }]
    })
    messages.append({'role': 'tool', 'content': '3+5=16', 'tool_call_id': '1'})
    model = MODELS.get('internlm2')(capability='chat')
    assert model.messages2prompt(
        messages
    ) == """<|im_start|>system name=<|plugin|>\n[{"description": "Compute the sum of two numbers", "name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}, {"description": "Calculate the product of two numbers", "name": "mul", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}]<|im_end|>\n<|im_start|>user\nCompute (3+5)*2<|im_end|>\n<|im_start|>assistant\n(3+5)*2 = 8*2 =<|action_start|><|plugin|>\n{"name": "mul", "parameters": {"a": 8, "b": 2}}<|action_end|><|im_end|>\n<|im_start|>environment\n3+5=16<|im_end|>\n<|im_start|>assistant\n"""  # noqa


def test_messages2prompt4internlm2_chat():
    model = MODELS.get('internlm2')()
    # Test with a single message
    messages = [
        {
            'role': 'system',
            'name': 'interpreter',
            'content': 'You have access to python environment.'
        },
        {
            'role': 'user',
            'content': 'use python drwa a line'
        },
        {
            'role': 'assistant',
            'content': '<|action_start|><|interpreter|>\ncode<|action_end|>\n'
        },
        {
            'role': 'environment',
            'name': 'interpreter',
            'content': "[{'type': 'image', 'content': 'image url'}]"
        },
    ]
    tools = [{
        'type': 'function',
        'function': {
            'name': 'add',
            'description': 'Compute the sum of two numbers',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'int',
                        'description': 'A number',
                    },
                    'b': {
                        'type': 'int',
                        'description': 'A number',
                    },
                },
                'required': ['a', 'b'],
            },
        }
    }]
    import json
    expected_prompt = (model.system.strip() + ' name=<|interpreter|>\nYou have access to python environment.' +
                       model.eosys + model.system.strip() +
                       f' name={model.plugin}\n{json.dumps(tools, ensure_ascii=False)}' + model.eosys + model.user +
                       'use python drwa a line' + model.eoh + model.assistant +
                       '<|action_start|><|interpreter|>\ncode<|action_end|>\n' + model.eoa + model.separator +
                       model.environment.strip() +
                       " name=<|interpreter|>\n[{'type': 'image', 'content': 'image url'}]" + model.eoenv +
                       model.assistant)
    actual_prompt = model.messages2prompt(messages, tools=tools)
    assert actual_prompt == expected_prompt

    # Test with a message where 'name' is not in name_map
    messages_invalid_name = [
        {
            'role': 'system',
            'name': 'invalid_name',
            'content': 'You have access to python environment.'
        },
        {
            'role': 'user',
            'content': 'use python draw a line'
        },
        {
            'role': 'assistant',
            'content': '\ncode\n'
        },
        {
            'role': 'environment',
            'name': 'invalid_name',
            'content': "[{'type': 'image', 'content': 'image url'}]"
        },
    ]
    expected_prompt_invalid_name = (model.system.strip() + '\nYou have access to python environment.' + model.eosys +
                                    model.user + 'use python draw a line' + model.eoh + model.assistant + '\ncode\n' +
                                    model.eoa + model.separator + model.environment.strip() +
                                    "\n[{'type': 'image', 'content': 'image url'}]" + model.eoenv + model.assistant)
    actual_prompt_invalid_name = model.messages2prompt(messages_invalid_name)
    assert actual_prompt_invalid_name == expected_prompt_invalid_name


# def test_llama3_1():
#     model = MODELS.get('hf')(model_path='meta-llama/Llama-3-7B-Chat')
#     messages = [dict(role='user', content='Can you check the top 5 trending songs on spotify?')]
#     tools = [{
#         'name': 'spotify_trending_songs',
#         'description': 'Get top trending songs on Spotify',
#         'parameters': {
#             'n': {
#                 'param_type': 'int',
#                 'description': 'Number of trending songs to get',
#                 'required': True
#             }
#         },
#     }]
#     actual_prompt = model.messages2prompt(messages, tools=tools)
#     expected_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n# Tool Instructions\n- Always execute python code in messages that you share.\n- When looking for real time information use relevant functions if available else fallback to brave_search\n\n\n\nYou have access to the following functions:\n\nUse the function \'spotify_trending_songs\' to: Get top trending songs on Spotify\n{"name": "spotify_trending_songs", "description": "Get top trending songs on Spotify", "parameters": {"n": {"param_type": "int", "description": "Number of trending songs to get", "required": true}}}\n\n\nIf a you choose to call a function ONLY reply in the following format:\n<{start_tag}={function_name}>{parameters}{end_tag}\nwhere\n\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function argument value as value.\nend_tag => `</function>`\n\nHere is an example,\n<function=example_function_name>{"example_name": "example_value"}</function>\n\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line"\n- Always add your sources when using search results to answer the user query\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCan you check the top 5 trending songs on spotify?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa
#     assert actual_prompt == expected_prompt


def test_baichuan():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('baichuan2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None

    model = MODELS.get('baichuan2')(capability='chat')
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt == '<reserved_106>' + prompt + '<reserved_107>'


def test_llama2():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('llama2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.meta_instruction is not None

    model = MODELS.get('llama2')(capability='chat', meta_instruction='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.meta_instruction == 'Provide answers in Python'

    model = MODELS.get('llama2')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


# def test_llama3():
#     conversation = [{'role': 'user', 'content': 'Are you ok?'}]

#     from lmdeploy.model import Llama3
#     t = Llama3(model_name='llama', capability='chat')
#     prompt = t.messages2prompt(conversation)
#     assert prompt == '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAre you ok?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa


def test_qwen():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('qwen')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is not None

    model = MODELS.get('qwen')(capability='chat')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt

    model = MODELS.get('qwen')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


def test_qwen2d5():

    model = MODELS.get('hf')(model_path='Qwen/Qwen2.5-7B-Instruct')

    # No tool call
    messages = [dict(role='user', content='What\'s the temperature in San Francisco now?')]
    no_tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                      'Cloud. You are a helpful '
                      "assistant.<|im_end|>\n<|im_start|>user\nWhat's the "
                      'temperature in San Francisco '
                      'now?<|im_end|>\n<|im_start|>assistant\n')
    assert model.messages2prompt(messages) == no_tool_prompt
    assert model.messages2prompt(messages, tools=[]) == no_tool_prompt

    messages.append({'role': 'assistant', 'content': 'I don\'t know.'})
    no_tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                      'Cloud. You are a helpful '
                      "assistant.<|im_end|>\n<|im_start|>user\nWhat's the "
                      'temperature in San Francisco '
                      "now?<|im_end|>\n<|im_start|>assistant\nI don't "
                      'know.')
    assert model.messages2prompt(messages) == no_tool_prompt
    # # Single tool call
    # tools = [{
    #     'name': 'get_current_temperature',
    #     'description': 'Get current temperature at a location.',
    #     'parameters': {
    #         'type': 'object',
    #         'properties': {
    #             'location': {
    #                 'type': 'string',
    #                 'description': 'The location to get the temperature for,'
    #                 ' in the format \'City, State, Country\'.'
    #             },
    #             'unit': {
    #                 'type': 'string',
    #                 'enum': ['celsius', 'fahrenheit'],
    #                 'description': 'The unit to return the temperature in. Defaults to '
    #                 '\'celsius\'.'
    #             }
    #         },
    #         'required': ['location']
    #     }
    # }]

    # messages = [dict(role='user', content='What\'s the temperature in San Francisco now?')]
    # tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
    #                'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
    #                'may call one or more functions to assist with the user '
    #                'query.\n\nYou are provided with function signatures '
    #                "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
    #                "\"function\", \"function\": {\"name\": "
    #                "\"get_current_temperature\", \"description\": \"Get "
    #                "current temperature at a location.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
    #                "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
    #                'unit to return the temperature in. Defaults to '
    #                "'celsius'.\"}}, \"required\": ["
    #                "\"location\"]}}}\n</tools>\n\nFor each function call, "
    #                'return a json object with function name and arguments '
    #                'within <tool_call></tool_call> XML tags:\n<tool_call>\n{'
    #                "\"name\": <function-name>, \"arguments\": "
    #                '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
    #                "|>user\nWhat's the temperature in San Francisco "
    #                'now?<|im_end|>\n<|im_start|>assistant\n')
    # assert model.messages2prompt(messages, tools=tools) == tool_prompt
    # # tool call send back
    # messages.append(
    #     dict(role='assistant',
    #          content='',
    #          tool_calls=[{
    #              'id': '0',
    #              'function': {
    #                  'arguments': '{"location": "San Francisco, CA, USA", "unit": "celsius"}',
    #                  'name': 'get_current_temperature'
    #              },
    #              'type': 'function'
    #          }]))
    # messages.append(
    #     dict(role='tool',
    #          name='get_current_temperature',
    #          content={
    #              'temperature': 26.1,
    #              'location': 'San Francisco, California, USA',
    #              'unit': 'celsius'
    #          },
    #          tool_call_id='0'))
    # tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
    #                'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
    #                'may call one or more functions to assist with the user '
    #                'query.\n\nYou are provided with function signatures '
    #                "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
    #                "\"function\", \"function\": {\"name\": "
    #                "\"get_current_temperature\", \"description\": \"Get "
    #                "current temperature at a location.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
    #                "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
    #                'unit to return the temperature in. Defaults to '
    #                "'celsius'.\"}}, \"required\": ["
    #                "\"location\"]}}}\n</tools>\n\nFor each function call, "
    #                'return a json object with function name and arguments '
    #                'within <tool_call></tool_call> XML tags:\n<tool_call>\n{'
    #                "\"name\": <function-name>, \"arguments\": "
    #                '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
    #                "|>user\nWhat's the temperature in San Francisco "
    #                'now?<|im_end|>\n<|im_start|>assistant\n\n<tool_call>\n'
    #                '{"name": "get_current_temperature", "arguments": '
    #                '{"location": "San Francisco, CA, USA", "unit": '
    #                '"celsius"}}\n</tool_call><|im_end|>\n<|im_start|>'
    #                'user\n<tool_response>\n{'
    #                "'temperature': 26.1, 'location': 'San Francisco, "
    #                "California, USA', 'unit': "
    #                "'celsius'}\n</tool_response><|im_end|>\n<|im_start"
    #                '|>assistant\n')
    # assert model.messages2prompt(messages, tools=tools) == tool_prompt
    # # Multi tool calling
    # tools = [{
    #     'name': 'get_current_temperature',
    #     'description': 'Get current temperature at a location.',
    #     'parameters': {
    #         'type': 'object',
    #         'properties': {
    #             'location': {
    #                 'type': 'string',
    #                 'description': 'The location to get the temperature for, in the format '
    #                 '\'City, State, Country\'.'
    #             },
    #             'unit': {
    #                 'type': 'string',
    #                 'enum': ['celsius', 'fahrenheit'],
    #                 'description': 'The unit to return the temperature in.'
    #                 ' Defaults to \'celsius\'.'
    #             }
    #         },
    #         'required': ['location']
    #     }
    # }, {
    #     'name': 'get_temperature_date',
    #     'description': 'Get temperature at a location and date.',
    #     'parameters': {
    #         'type': 'object',
    #         'properties': {
    #             'location': {
    #                 'type': 'string',
    #                 'description': 'The location to get the temperature for,'
    #                 ' in the format \'City, State, Country\'.'
    #             },
    #             'date': {
    #                 'type': 'string',
    #                 'description': 'The date to get the temperature for,'
    #                 ' in the format \'Year-Month-Day\'.'
    #             },
    #             'unit': {
    #                 'type': 'string',
    #                 'enum': ['celsius', 'fahrenheit'],
    #                 'description': 'The unit to return the temperature in.'
    #                 ' Defaults to \'celsius\'.'
    #             }
    #         },
    #         'required': ['location', 'date']
    #     }
    # }]
    # messages = [
    #     dict(role='user',
    #          content='Today is 2024-11-14, What\'s the temperature in'
    #          ' San Francisco now? How about tomorrow?')
    # ]
    # tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
    #                'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
    #                'may call one or more functions to assist with the user '
    #                'query.\n\nYou are provided with function signatures '
    #                "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
    #                "\"function\", \"function\": {\"name\": "
    #                "\"get_current_temperature\", \"description\": \"Get "
    #                "current temperature at a location.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
    #                "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
    #                'unit to return the temperature in. Defaults to '
    #                "'celsius'.\"}}, \"required\": [\"location\"]}}}\n{"
    #                "\"type\": \"function\", \"function\": {\"name\": "
    #                "\"get_temperature_date\", \"description\": \"Get "
    #                "temperature at a location and date.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"date\": {\"type\": \"string\", "
    #                "\"description\": \"The date to get the temperature for, "
    #                "in the format 'Year-Month-Day'.\"}, \"unit\": {\"type\": "
    #                "\"string\", \"enum\": [\"celsius\", \"fahrenheit\"], "
    #                "\"description\": \"The unit to return the temperature "
    #                "in. Defaults to 'celsius'.\"}}, \"required\": ["
    #                "\"location\", \"date\"]}}}\n</tools>\n\nFor each "
    #                'function call, return a json object with function name '
    #                'and arguments within <tool_call></tool_call> XML '
    #                "tags:\n<tool_call>\n{\"name\": <function-name>, "
    #                "\"arguments\": "
    #                '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
    #                "|>user\nToday is 2024-11-14, What's the temperature in "
    #                'San Francisco now? How about '
    #                'tomorrow?<|im_end|>\n<|im_start|>assistant\n')
    # assert model.messages2prompt(messages, tools=tools) == tool_prompt

    # messages.append(
    #     dict(role='tool',
    #          name='get_current_temperature',
    #          content={
    #              'temperature': 26.1,
    #              'location': 'San Francisco, California, USA',
    #              'unit': 'celsius'
    #          },
    #          tool_call_id='0'))
    # messages.append(
    #     dict(role='tool',
    #          name='get_temperature_date',
    #          content={
    #              'temperature': 25.9,
    #              'location': 'San Francisco, California, USA',
    #              'date': '2024-11-15',
    #              'unit': 'celsius'
    #          },
    #          tool_call_id='1'))
    # tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
    #                'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
    #                'may call one or more functions to assist with the user '
    #                'query.\n\nYou are provided with function signatures '
    #                "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
    #                "\"function\", \"function\": {\"name\": "
    #                "\"get_current_temperature\", \"description\": \"Get "
    #                "current temperature at a location.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
    #                "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
    #                'unit to return the temperature in. Defaults to '
    #                "'celsius'.\"}}, \"required\": [\"location\"]}}}\n{"
    #                "\"type\": \"function\", \"function\": {\"name\": "
    #                "\"get_temperature_date\", \"description\": \"Get "
    #                "temperature at a location and date.\", \"parameters\": {"
    #                "\"type\": \"object\", \"properties\": {\"location\": {"
    #                "\"type\": \"string\", \"description\": \"The location to "
    #                "get the temperature for, in the format 'City, State, "
    #                "Country'.\"}, \"date\": {\"type\": \"string\", "
    #                "\"description\": \"The date to get the temperature for, "
    #                "in the format 'Year-Month-Day'.\"}, \"unit\": {\"type\": "
    #                "\"string\", \"enum\": [\"celsius\", \"fahrenheit\"], "
    #                "\"description\": \"The unit to return the temperature "
    #                "in. Defaults to 'celsius'.\"}}, \"required\": ["
    #                "\"location\", \"date\"]}}}\n</tools>\n\nFor each "
    #                'function call, return a json object with function name '
    #                'and arguments within <tool_call></tool_call> XML '
    #                "tags:\n<tool_call>\n{\"name\": <function-name>, "
    #                "\"arguments\": "
    #                '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
    #                "|>user\nToday is 2024-11-14, What's the temperature in "
    #                'San Francisco now? How about '
    #                'tomorrow?<|im_end|>\n<|im_start|>user\n<tool_response'
    #                ">\n{'temperature': 26.1, 'location': 'San Francisco, "
    #                "California, USA', 'unit': "
    #                "'celsius'}\n</tool_response>\n<tool_response>\n{"
    #                "'temperature': 25.9, 'location': 'San Francisco, "
    #                "California, USA', 'date': '2024-11-15', 'unit': "
    #                "'celsius'}\n</tool_response><|im_end|>\n<|im_start"
    #                '|>assistant\n')
    # assert model.messages2prompt(messages, tools=tools) == tool_prompt


# def test_qwen2d5_vl():
#     prompt = 'hello, can u introduce yourself'
#     model = MODELS.get('hf')(model_path='Qwen/Qwen2.5-VL-Chat')
#     assert model.get_prompt(prompt, sequence_start=True) == prompt
#     assert model.get_prompt(prompt, sequence_start=False) == prompt

#     model = MODELS.get('qwen2d5-vl')(capability='chat')

#     messages = [dict(role='user', content='What\'s the temperature in San Francisco now?')]
#     res = ('<|im_start|>system\nYou are a helpful '
#            "assistant.<|im_end|>\n<|im_start|>user\nWhat's the "
#            'temperature in San Francisco '
#            'now?<|im_end|>\n<|im_start|>assistant\n')
#     assert model.messages2prompt(messages) == res

#     messages.append({'role': 'assistant', 'content': 'I don\'t know.'})
#     res = ('<|im_start|>system\nYou are a helpful '
#            "assistant.<|im_end|>\n<|im_start|>user\nWhat's the "
#            'temperature in San Francisco '
#            "now?<|im_end|>\n<|im_start|>assistant\nI don't "
#            'know.<|im_end|>\n<|im_start|>assistant\n')
#     assert model.messages2prompt(messages) == res


def test_codellama_completion():
    model = MODELS.get('codellama')(capability='completion')
    prompt = """\
import socket

def ping_exponential_backoff(host: str):"""
    assert model.get_prompt(prompt) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None


def test_codellama_infilling():
    model = MODELS.get('codellama')(capability='infilling')
    prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL>
    return result
'''
    _prompt = model.get_prompt(prompt)
    assert _prompt.find('<FILL>') == -1
    assert model.stop_words == ['<EOT>']

    model = MODELS.get('codellama')(capability='infilling', suffix_first=True)
    _prompt = model.get_prompt(prompt)
    assert _prompt.find('<FILL>') == -1


def test_codellama_chat():
    model = MODELS.get('codellama')(capability='chat', system='Provide answers in Python')
    prompt = 'Write a function that computes the set of sums of all contiguous sublists of a given list.'  # noqa: E501
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt.find('Provide answers in Python') != -1

    _prompt = model.get_prompt(prompt, sequence_start=False)
    assert _prompt.find('Provide answers in Python') == -1
    assert model.stop_words is None


def test_codellama_python_specialist():
    model = MODELS.get('codellama')(capability='python')
    prompt = """
    def remove_non_ascii(s: str) -> str:
"""
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None


def test_codellama_others():
    model = None
    with pytest.raises(AssertionError):
        model = MODELS.get('codellama')(capability='java')
    assert model is None


def test_deepseek():
    model = MODELS.get('hf')(model_path='deepseek-ai/DeepSeek-V2-Lite')
    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'hi'
    }]
    res = model.messages2prompt(messages)
    ref = """<｜begin▁of▁sentence｜>you are a helpful assistant

User: who are you

Assistant: I am an AI<｜end▁of▁sentence｜>User: hi

Assistant:"""
    assert res == ref


def test_deepseek_coder():
    model = MODELS.get('hf')(model_path='deepseek-ai/deepseek-coder-1.3b-instruct')
    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'hi'
    }]
    ref = """<｜begin▁of▁sentence｜>you are a helpful assistant### Instruction:
who are you
### Response:
I am an AI
<|EOT|>
### Instruction:
hi
### Response:
"""
    res = model.messages2prompt(messages)
    assert res == ref


def test_chatglm3():
    model_path_and_name = 'THUDM/chatglm3-6b'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'hf'
    model = MODELS.get(deduced_name)(model_path_and_name)
    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    ref = """[gMASK]sop<|system|>
 you are a helpful assistant<|user|>
 who are you<|assistant|>
 I am an AI<|user|>
 AGI is?<|assistant|>"""
    res = model.messages2prompt(messages)
    assert res == ref


def test_glm4():
    model_path_and_name = 'THUDM/glm-4-9b-chat'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'hf'
    model = MODELS.get(deduced_name)(model_path_and_name)
    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    ref = """[gMASK]<sop><|system|>
you are a helpful assistant<|user|>
who are you<|assistant|>
I am an AI<|user|>
AGI is?<|assistant|>"""
    res = model.messages2prompt(messages)
    assert res == ref


# def test_internvl_phi3():
#     assert best_match_model('OpenGVLab/InternVL-Chat-V1-5') == 'internvl-internlm2'
#     assert best_match_model('OpenGVLab/Mini-InternVL-Chat-2B-V1-5') == 'internvl-internlm2'

#     model_path_and_name = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
#     deduced_name = best_match_model(model_path_and_name)
#     assert deduced_name == 'internvl-phi3'

#     model = MODELS.get(deduced_name)()
#     messages = [{
#         'role': 'user',
#         'content': 'who are you'
#     }, {
#         'role': 'assistant',
#         'content': 'I am an AI'
#     }, {
#         'role': 'user',
#         'content': 'hi'
#     }]
#     res = model.messages2prompt(messages)
#     from huggingface_hub import hf_hub_download
#     hf_hub_download(repo_id=model_path_and_name, filename='conversation.py', local_dir='.')

#     try:
#         import os

#         from conversation import get_conv_template
#         template = get_conv_template('phi3-chat')
#         template.append_message(template.roles[0], messages[0]['content'])
#         template.append_message(template.roles[1], messages[1]['content'])
#         ref = template.get_prompt()
#         assert res.startswith(ref)
#         if os.path.exists('conversation.py'):
#             os.remove('conversation.py')
#     except ImportError:
#         pass

# def test_internvl2():
#     model = MODELS.get('internvl2-internlm2')()
#     messages = [{'role': 'user', 'content': 'who are you'}, {'role': 'assistant', 'content': 'I am an AI'}]
#     expected = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的'\
#         '书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'\
#         '<|im_end|><|im_start|>user\nwho are you<|im_end|><|im_start|>'\
#         'assistant\nI am an AI'
#     res = model.messages2prompt(messages)
#     assert res == expected

# def test_chemvlm():
#     deduced_name = best_match_model('AI4Chem/ChemVLM-8B')
#     assert deduced_name == 'hf'
#     model = MODELS.get(deduced_name)('AI4Chem/ChemVLM-8B')
#     messages = [{'role': 'user', 'content': 'who are you'}, {'role': 'assistant', 'content': 'I am an AI'}]
#     expected = '<s><|im_start|>system\nYou are an AI assistant whose name is '\
#         'InternLM (书生·浦语).<|im_end|>\n<|im_start|>user\nwho are you'\
#         '<|im_end|>\n<|im_start|>assistant\nI am an AI'
#     res = model.messages2prompt(messages)
#     assert res == expected

# def test_codegeex4():
#     model_path_and_name = 'THUDM/codegeex4-all-9b'
#     deduced_name = best_match_model(model_path_and_name)
#     assert deduced_name == 'hf'
#     # TODO: make an expected prompt


@pytest.mark.parametrize('model_path_and_name', [
    'microsoft/Phi-3-mini-128k-instruct',
    'microsoft/Phi-3-vision-128k-instruct',
    'microsoft/Phi-3.5-mini-instruct',
    'microsoft/Phi-3.5-vision-instruct',
    'microsoft/Phi-3.5-MoE-instruct',
])
def test_phi3(model_path_and_name):
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'hf'
    model = MODELS.get(deduced_name)(model_path_and_name)
    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path_and_name, trust_remote_code=True)
    ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # TODO: make an expected prompt
    res = model.messages2prompt(messages)
    assert res.startswith(ref)


@pytest.mark.parametrize('model_path_or_name', [
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'deepseek-ai/DeepSeek-R1',
    'deepseek-ai/DeepSeek-R1-Zero',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-V3',
])
def test_deepseek_r1(model_path_or_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
    deduced_name = best_match_model(model_path_or_name)
    assert deduced_name == 'hf'
    chat_template = MODELS.get(deduced_name)(model_path_or_name)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lm_res = chat_template.messages2prompt(messages)
    assert ref == lm_res


@pytest.mark.parametrize(
    'model_path_or_name',
    ['deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2'])
def test_deepseek_vl2(model_path_or_name):
    deduced_name = best_match_model(model_path_or_name)
    assert deduced_name == 'deepseek-vl2'

    chat_template = MODELS.get(deduced_name)()
    messages = [{
        'role': 'user',
        'content': 'This is image_1: <image>\n'
        'This is image_2: <image>\n'
        'This is image_3: <image>\n Can you tell me what are in the images?',
        'images': [
            'images/multi_image_1.jpeg',
            'images/multi_image_2.jpeg',
            'images/multi_image_3.jpeg',
        ],
    }, {
        'role': 'assistant',
        'content': ''
    }]

    ref = '<|User|>: This is image_1: <image>\nThis is image_2: <image>\nThis is image_3: <image>' + \
          '\n Can you tell me what are in the images?\n\n<|Assistant|>:'
    lm_res = chat_template.messages2prompt(messages)
    assert ref == lm_res


@pytest.mark.parametrize('model_path_or_name', [
    'Qwen/QwQ-32B',
    'Qwen/QwQ-32B-Preview',
    'Qwen/QwQ-32B-AWQ',
])
def test_qwq(model_path_or_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
    deduced_name = best_match_model(model_path_or_name)
    assert deduced_name == 'hf'
    chat_template = MODELS.get(deduced_name)(model_path_or_name)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lm_res = chat_template.messages2prompt(messages)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['Qwen/Qwen3-30B-A3B', 'Qwen/Qwen2.5-7B-Instruct'])
@pytest.mark.parametrize('enable_thinking', [True, False, None])
def test_qwen3(model_path, enable_thinking):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_template_name = best_match_model(model_path)
    assert chat_template_name == 'hf'
    chat_template = MODELS.get(chat_template_name)(model_path)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    if enable_thinking is None:
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        ref = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    lm_res = chat_template.messages2prompt(messages, enable_thinking=enable_thinking)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [None, True, False])
@pytest.mark.parametrize('has_user_sys', [True, False])
def test_interns1(model_path, enable_thinking, has_user_sys):
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        pytest.skip(reason=f'{model_path} not exists')

    chat_template_name = best_match_model(model_path)
    chat_template = MODELS.get(chat_template_name)(model_path)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    if not has_user_sys:
        messages = messages[1:]

    if enable_thinking is None:
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        ref = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    lm_res = chat_template.messages2prompt(messages, enable_thinking=enable_thinking)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [None, True, False])
@pytest.mark.parametrize('has_user_sys', [True, False])
def test_interns1_tools(model_path, enable_thinking, has_user_sys):
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        pytest.skip(reason=f'{model_path} not exists')

    chat_template_name = best_match_model(model_path)
    chat_template = MODELS.get(chat_template_name)(model_path)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'find_user_id_by_name_zip',
                'description':
                'Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email.',  # noqa: E501
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'first_name': {
                            'type': 'string',
                            'description': "The first name of the customer, such as 'John'."
                        },
                        'last_name': {
                            'type': 'string',
                            'description': "The last name of the customer, such as 'Doe'."
                        },
                        'zip': {
                            'type': 'string',
                            'description': "The zip code of the customer, such as '12345'."
                        }
                    },
                    'required': ['first_name', 'last_name', 'zip']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_order_details',
                'description': 'Get the status and details of an order.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'order_id': {
                            'type':
                            'string',
                            'description':
                            "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."  # noqa: E501
                        }
                    },
                    'required': ['order_id']
                }
            }
        }
    ]
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful assistant'
        },
        {
            'role': 'user',
            'content': "Hi there! I'm looking to return a couple of items from a recent order."
        },
        {
            'role':
            'assistant',
            'content':
            'Could you please provide your email address associated with the account, or share your first name, last name, and zip code?',  # noqa: E501
            'reasoning_content':
            'Okay, the user wants to return some items from a recent order. Let me start by authenticating their identity...'  # noqa: E501
        },
        {
            'role': 'user',
            'content': 'Sure, my name is Omar Anderson and my zip code is 19031.'
        },
        {
            'role':
            'assistant',
            'content':
            '<content>',
            'reasoning_content':
            "Since he didn't provide an email, I should use the find_user_id_by_name_zip function. Let me...",  # noqa: E501
            'tool_calls': [{
                'function': {
                    'arguments': '{"first_name": "Omar", "last_name": "Anderson", "zip": "19031"}',
                    'name': 'find_user_id_by_name_zip'
                },
                'id': 'chatcmpl-tool-a9f439084bfc4af29fee2e5105050a38',
                'type': 'function'
            }]
        },
        {
            'content': 'omar_anderson_3203',
            'name': 'find_user_id_by_name_zip',
            'role': 'tool'
        }
    ]
    if not has_user_sys:
        messages = messages[1:]
    if enable_thinking is None:
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    else:
        ref = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            tools=tools,
                                            enable_thinking=enable_thinking)
    lm_res = chat_template.messages2prompt(messages, enable_thinking=enable_thinking, tools=tools)
    assert ref == lm_res
