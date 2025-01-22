import pytest

from lmdeploy.model import MODELS, best_match_model


@pytest.mark.parametrize('model_path_and_name', [('internlm/internlm-chat-7b', ['internlm']),
                                                 ('internlm/internlm2-1_8b', ['base']),
                                                 ('models--internlm--internlm-chat-7b/snapshots/1234567', ['internlm']),
                                                 ('Qwen/Qwen-7B-Chat', ['qwen']),
                                                 ('Qwen/Qwen2.5-7B-Instruct', ['qwen2d5']),
                                                 ('codellama/CodeLlama-7b-hf', ['codellama']),
                                                 ('upstage/SOLAR-0-70b', ['solar', 'solar-70b']),
                                                 ('meta-llama/Llama-2-7b-chat-hf', ['llama2']),
                                                 ('THUDM/chatglm2-6b', ['chatglm']),
                                                 ('01-ai/Yi-6B-200k', ['yi', 'yi-200k']), ('01-ai/Yi-34B-Chat', ['yi']),
                                                 ('01-ai/Yi-6B-Chat', ['yi', 'yi-chat']),
                                                 ('WizardLM/WizardLM-70B-V1.0', ['wizardlm']),
                                                 ('codellama/CodeLlama-34b-Instruct-hf', ['codellama']),
                                                 ('deepseek-ai/deepseek-coder-6.7b-instruct', ['deepseek-coder']),
                                                 ('deepseek-ai/deepseek-vl-7b-chat', ['deepseek-vl']),
                                                 ('deepseek-ai/deepseek-moe-16b-chat', ['deepseek']),
                                                 ('internlm/internlm-xcomposer2-4khd-7b', ['internlm-xcomposer2']),
                                                 ('internlm/internlm-xcomposer2d5-7b', ['internlm-xcomposer2d5']),
                                                 ('tiiuae/falcon-7b', ['falcon']), ('workspace', ['base'])])
@pytest.mark.parametrize('suffix', ['', '-w4', '-4bit', '-16bit'])
def test_best_match_model(model_path_and_name, suffix):
    if model_path_and_name[0] == 'internlm/internlm2-1_8b' and suffix:
        return  # internlm/internlm2-1_8b-suffix will got None
    deduced_name = best_match_model(model_path_and_name[0] + suffix)
    if deduced_name is not None:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'
    else:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'


@pytest.mark.parametrize('model_name', ['llama2', 'base', 'yi', 'qwen-7b', 'vicuna'])
@pytest.mark.parametrize('meta_instruction', ['[fake meta_instruction]'])
def test_model_config(model_name, meta_instruction):
    from lmdeploy.model import ChatTemplateConfig
    chat_template = ChatTemplateConfig(model_name, meta_instruction=meta_instruction).chat_template
    prompt = chat_template.get_prompt('')
    if model_name == 'base':
        assert prompt == ''
    else:
        assert meta_instruction in prompt


def test_base_model():
    model = MODELS.get('llama')()
    assert model is not None
    assert model.capability == 'chat'
    assert model.get_prompt('test') == 'test'
    assert model.stop_words is None

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
    model = MODELS.get('internlm2')()
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


def test_llama3_1():
    model = MODELS.get('llama3_1')()
    messages = [dict(role='user', content='Can you check the top 5 trending songs on spotify?')]
    tools = [{
        'name': 'spotify_trending_songs',
        'description': 'Get top trending songs on Spotify',
        'parameters': {
            'n': {
                'param_type': 'int',
                'description': 'Number of trending songs to get',
                'required': True
            }
        },
    }]
    actual_prompt = model.messages2prompt(messages, tools=tools)
    expected_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n# Tool Instructions\n- Always execute python code in messages that you share.\n- When looking for real time information use relevant functions if available else fallback to brave_search\n\n\n\nYou have access to the following functions:\n\nUse the function \'spotify_trending_songs\' to: Get top trending songs on Spotify\n{"name": "spotify_trending_songs", "description": "Get top trending songs on Spotify", "parameters": {"n": {"param_type": "int", "description": "Number of trending songs to get", "required": true}}}\n\n\nIf a you choose to call a function ONLY reply in the following format:\n<{start_tag}={function_name}>{parameters}{end_tag}\nwhere\n\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function argument value as value.\nend_tag => `</function>`\n\nHere is an example,\n<function=example_function_name>{"example_name": "example_value"}</function>\n\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line"\n- Always add your sources when using search results to answer the user query\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCan you check the top 5 trending songs on spotify?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa
    assert actual_prompt == expected_prompt


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


def test_llama3():
    conversation = [{'role': 'user', 'content': 'Are you ok?'}]

    from lmdeploy.model import Llama3
    t = Llama3(model_name='llama', capability='chat')
    prompt = t.messages2prompt(conversation)
    assert prompt == '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAre you ok?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa


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
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('qwen2d5')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt

    model = MODELS.get('qwen2d5')(capability='chat')

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
                      'know.<|im_end|>\n<|im_start|>assistant\n')
    assert model.messages2prompt(messages) == no_tool_prompt
    # Single tool call
    tools = [{
        'name': 'get_current_temperature',
        'description': 'Get current temperature at a location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for,'
                    ' in the format \'City, State, Country\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'The unit to return the temperature in. Defaults to '
                    '\'celsius\'.'
                }
            },
            'required': ['location']
        }
    }]

    messages = [dict(role='user', content='What\'s the temperature in San Francisco now?')]
    tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                   'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
                   'may call one or more functions to assist with the user '
                   'query.\n\nYou are provided with function signatures '
                   "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
                   "\"function\", \"function\": {\"name\": "
                   "\"get_current_temperature\", \"description\": \"Get "
                   "current temperature at a location.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
                   "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
                   'unit to return the temperature in. Defaults to '
                   "'celsius'.\"}}, \"required\": ["
                   "\"location\"]}}}\n</tools>\n\nFor each function call, "
                   'return a json object with function name and arguments '
                   'within <tool_call></tool_call> XML tags:\n<tool_call>\n{'
                   "\"name\": <function-name>, \"arguments\": "
                   '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
                   "|>user\nWhat's the temperature in San Francisco "
                   'now?<|im_end|>\n<|im_start|>assistant\n')
    assert model.messages2prompt(messages, tools=tools) == tool_prompt

    messages.append(
        dict(role='tool',
             name='get_current_temperature',
             content={
                 'temperature': 26.1,
                 'location': 'San Francisco, California, USA',
                 'unit': 'celsius'
             },
             tool_call_id='0'))
    tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                   'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
                   'may call one or more functions to assist with the user '
                   'query.\n\nYou are provided with function signatures '
                   "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
                   "\"function\", \"function\": {\"name\": "
                   "\"get_current_temperature\", \"description\": \"Get "
                   "current temperature at a location.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
                   "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
                   'unit to return the temperature in. Defaults to '
                   "'celsius'.\"}}, \"required\": ["
                   "\"location\"]}}}\n</tools>\n\nFor each function call, "
                   'return a json object with function name and arguments '
                   'within <tool_call></tool_call> XML tags:\n<tool_call>\n{'
                   "\"name\": <function-name>, \"arguments\": "
                   '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
                   "|>user\nWhat's the temperature in San Francisco "
                   'now?<|im_end|>\n<|im_start|>user\n<tool_response>\n{'
                   "'temperature': 26.1, 'location': 'San Francisco, "
                   "California, USA', 'unit': "
                   "'celsius'}\n</tool_response><|im_end|>\n<|im_start"
                   '|>assistant\n')
    assert model.messages2prompt(messages, tools=tools) == tool_prompt
    # Multi tool calling
    tools = [{
        'name': 'get_current_temperature',
        'description': 'Get current temperature at a location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format '
                    '\'City, State, Country\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'The unit to return the temperature in.'
                    ' Defaults to \'celsius\'.'
                }
            },
            'required': ['location']
        }
    }, {
        'name': 'get_temperature_date',
        'description': 'Get temperature at a location and date.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for,'
                    ' in the format \'City, State, Country\'.'
                },
                'date': {
                    'type': 'string',
                    'description': 'The date to get the temperature for,'
                    ' in the format \'Year-Month-Day\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'The unit to return the temperature in.'
                    ' Defaults to \'celsius\'.'
                }
            },
            'required': ['location', 'date']
        }
    }]
    messages = [
        dict(role='user',
             content='Today is 2024-11-14, What\'s the temperature in'
             ' San Francisco now? How about tomorrow?')
    ]
    tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                   'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
                   'may call one or more functions to assist with the user '
                   'query.\n\nYou are provided with function signatures '
                   "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
                   "\"function\", \"function\": {\"name\": "
                   "\"get_current_temperature\", \"description\": \"Get "
                   "current temperature at a location.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
                   "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
                   'unit to return the temperature in. Defaults to '
                   "'celsius'.\"}}, \"required\": [\"location\"]}}}\n{"
                   "\"type\": \"function\", \"function\": {\"name\": "
                   "\"get_temperature_date\", \"description\": \"Get "
                   "temperature at a location and date.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"date\": {\"type\": \"string\", "
                   "\"description\": \"The date to get the temperature for, "
                   "in the format 'Year-Month-Day'.\"}, \"unit\": {\"type\": "
                   "\"string\", \"enum\": [\"celsius\", \"fahrenheit\"], "
                   "\"description\": \"The unit to return the temperature "
                   "in. Defaults to 'celsius'.\"}}, \"required\": ["
                   "\"location\", \"date\"]}}}\n</tools>\n\nFor each "
                   'function call, return a json object with function name '
                   'and arguments within <tool_call></tool_call> XML '
                   "tags:\n<tool_call>\n{\"name\": <function-name>, "
                   "\"arguments\": "
                   '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
                   "|>user\nToday is 2024-11-14, What's the temperature in "
                   'San Francisco now? How about '
                   'tomorrow?<|im_end|>\n<|im_start|>assistant\n')
    assert model.messages2prompt(messages, tools=tools) == tool_prompt

    messages.append(
        dict(role='tool',
             name='get_current_temperature',
             content={
                 'temperature': 26.1,
                 'location': 'San Francisco, California, USA',
                 'unit': 'celsius'
             },
             tool_call_id='0'))
    messages.append(
        dict(role='tool',
             name='get_temperature_date',
             content={
                 'temperature': 25.9,
                 'location': 'San Francisco, California, USA',
                 'date': '2024-11-15',
                 'unit': 'celsius'
             },
             tool_call_id='1'))
    tool_prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba '
                   'Cloud. You are a helpful assistant.\n\n# Tools\n\nYou '
                   'may call one or more functions to assist with the user '
                   'query.\n\nYou are provided with function signatures '
                   "within <tools></tools> XML tags:\n<tools>\n{\"type\": "
                   "\"function\", \"function\": {\"name\": "
                   "\"get_current_temperature\", \"description\": \"Get "
                   "current temperature at a location.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"unit\": {\"type\": \"string\", \"enum\": "
                   "[\"celsius\", \"fahrenheit\"], \"description\": \"The "
                   'unit to return the temperature in. Defaults to '
                   "'celsius'.\"}}, \"required\": [\"location\"]}}}\n{"
                   "\"type\": \"function\", \"function\": {\"name\": "
                   "\"get_temperature_date\", \"description\": \"Get "
                   "temperature at a location and date.\", \"parameters\": {"
                   "\"type\": \"object\", \"properties\": {\"location\": {"
                   "\"type\": \"string\", \"description\": \"The location to "
                   "get the temperature for, in the format 'City, State, "
                   "Country'.\"}, \"date\": {\"type\": \"string\", "
                   "\"description\": \"The date to get the temperature for, "
                   "in the format 'Year-Month-Day'.\"}, \"unit\": {\"type\": "
                   "\"string\", \"enum\": [\"celsius\", \"fahrenheit\"], "
                   "\"description\": \"The unit to return the temperature "
                   "in. Defaults to 'celsius'.\"}}, \"required\": ["
                   "\"location\", \"date\"]}}}\n</tools>\n\nFor each "
                   'function call, return a json object with function name '
                   'and arguments within <tool_call></tool_call> XML '
                   "tags:\n<tool_call>\n{\"name\": <function-name>, "
                   "\"arguments\": "
                   '<args-json-object>}\n</tool_call><|im_end|>\n<|im_start'
                   "|>user\nToday is 2024-11-14, What's the temperature in "
                   'San Francisco now? How about '
                   'tomorrow?<|im_end|>\n<|im_start|>user\n<tool_response'
                   ">\n{'temperature': 26.1, 'location': 'San Francisco, "
                   "California, USA', 'unit': "
                   "'celsius'}\n</tool_response>\n<tool_response>\n{"
                   "'temperature': 25.9, 'location': 'San Francisco, "
                   "California, USA', 'date': '2024-11-15', 'unit': "
                   "'celsius'}\n</tool_response><|im_end|>\n<|im_start"
                   '|>assistant\n')
    assert model.messages2prompt(messages, tools=tools) == tool_prompt


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


def test_deepseek_coder():
    model = MODELS.get('deepseek-coder')()
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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-instruct', trust_remote_code=True)
    ref = tokenizer.apply_chat_template(messages, tokenize=False)
    res = '<｜begin▁of▁sentence｜>' + model.messages2prompt(messages)
    assert res.startswith(ref)


def test_chatglm3():
    model_path_and_name = 'THUDM/chatglm3-6b'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'chatglm3'
    model = MODELS.get(deduced_name)()
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
    ref = tokenizer.apply_chat_template(messages, tokenize=False)
    res = model.messages2prompt(messages)
    assert res.startswith(ref)


def test_glm4():
    model_path_and_name = 'THUDM/glm-4-9b-chat'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'glm4'

    model = MODELS.get(deduced_name)()
    # check stop words
    assert model.stop_words == ['<|user|>', '<|endoftext|>', '<|observation|>']
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
    ref = tokenizer.apply_chat_template(messages, tokenize=False)
    res = model.messages2prompt(messages)
    assert res.startswith(ref)


def test_internvl_phi3():
    assert best_match_model('OpenGVLab/InternVL-Chat-V1-5') == 'internvl-internlm2'
    assert best_match_model('OpenGVLab/Mini-InternVL-Chat-2B-V1-5') == 'internvl-internlm2'

    model_path_and_name = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'internvl-phi3'

    model = MODELS.get(deduced_name)()
    messages = [{
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
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=model_path_and_name, filename='conversation.py', local_dir='.')

    try:
        import os

        from conversation import get_conv_template
        template = get_conv_template('phi3-chat')
        template.append_message(template.roles[0], messages[0]['content'])
        template.append_message(template.roles[1], messages[1]['content'])
        ref = template.get_prompt()
        assert res.startswith(ref)
        if os.path.exists('conversation.py'):
            os.remove('conversation.py')
    except ImportError:
        pass


def test_internvl2():
    model = MODELS.get('internvl2-internlm2')()
    messages = [{'role': 'user', 'content': 'who are you'}, {'role': 'assistant', 'content': 'I am an AI'}]
    expected = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的'\
        '书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'\
        '<|im_end|><|im_start|>user\nwho are you<|im_end|><|im_start|>'\
        'assistant\nI am an AI'
    res = model.messages2prompt(messages)
    assert res == expected


def test_chemvlm():
    deduced_name = best_match_model('AI4Chem/ChemVLM-8B')

    assert deduced_name == 'internvl-internlm2'
    model = MODELS.get(deduced_name)()
    messages = [{'role': 'user', 'content': 'who are you'}, {'role': 'assistant', 'content': 'I am an AI'}]
    expected = '<|im_start|>system\nYou are an AI assistant whose name is '\
        'InternLM (书生·浦语).<|im_end|>\n<|im_start|>user\nwho are you'\
        '<|im_end|>\n<|im_start|>assistant\nI am an AI'
    res = model.messages2prompt(messages)
    assert res == expected


def test_codegeex4():
    model_path_and_name = 'THUDM/codegeex4-all-9b'
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'codegeex4'
    model = MODELS.get(deduced_name)()
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
    ref = tokenizer.apply_chat_template(messages, tokenize=False)
    res = model.messages2prompt(messages)
    assert res.startswith(ref)


@pytest.mark.parametrize('model_path_and_name', [
    'microsoft/Phi-3-mini-128k-instruct',
    'microsoft/Phi-3-vision-128k-instruct',
    'microsoft/Phi-3.5-mini-instruct',
    'microsoft/Phi-3.5-vision-instruct',
    'microsoft/Phi-3.5-MoE-instruct',
])
def test_phi3(model_path_and_name):
    deduced_name = best_match_model(model_path_and_name)
    assert deduced_name == 'phi-3'
    model = MODELS.get(deduced_name)()
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
])
def test_deepseek_r1(model_path_or_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
    deduced_name = best_match_model(model_path_or_name)
    chat_template = MODELS.get(deduced_name)()

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
