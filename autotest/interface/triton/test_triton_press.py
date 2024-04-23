import random
import string

import pytest
import yaml
from utils.triton_engine import Engine

SERVER_ADDR = '0.0.0.0:33337'


def get_random_session_id():
    characters = string.digits
    random_chars = ''.join(random.choice(characters) for i in range(6))
    return int(random_chars)


class Test_TritonInterface_press:

    @pytest.mark.tmp
    def test_diff_dict(self):
        with open('autotest/interface/triton/gen_case_dict.yaml', 'r') as f:
            case_config = yaml.load(f, Loader=yaml.SafeLoader)

        for case in case_config.keys():
            try:
                dict_info = case_config[case]
                if dict_info is None:
                    dict_info = {}

                print('dict info:' + str(dict_info))
                engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

                session_id = get_random_session_id()
                engine.triton_infer(session_id, '你好，你叫什么名字', 'req_id:' + case,
                                    500, True, True)
                engine.triton_resume(session_id)
                engine.triton_end(session_id)
                engine.triton_set_session(None)
            except Exception as e:
                print(f'error: {e}')

    def test_start_stop_param(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction':
            'You are an AI assistant whose name is InternLM (书生·浦语).',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': ['ി'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

        # case1 - triton_infer有各类入参
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(1),
                            'req_id:' + str(1), 50, True, False)
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(2),
                            'req_id:' + str(2), 50, False, False)
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(3),
                            'req_id:' + str(3), 50, False, True)
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(4),
                            'req_id:' + str(4), 50, True, True)
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(5),
                            'req_id:' + str(5), 50, True, False)

    def test_start_stop_param2(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction':
            'You are an AI assistant whose name is InternLM (书生·浦语).',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': ['ി'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(1),
                            'req_id:' + str(1), 50, True, True)
        engine.triton_resume(session_id)
        engine.triton_end(session_id)

        engine.triton_set_session(None)
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(2),
                            'req_id:' + str(2), 50, False, True)
        engine.triton_resume(session_id)
        engine.triton_end(session_id)

        engine.triton_set_session(None)
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(3),
                            'req_id:' + str(3), 50, False, False)
        engine.triton_resume(session_id)
        engine.triton_end(session_id)

        engine.triton_set_session(None)
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(4),
                            'req_id:' + str(4), 50, True, True)
        engine.triton_resume(session_id)
        engine.triton_end(session_id)

    def test_long_input(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction':
            'You are an AI assistant whose name is InternLM (书生·浦语).',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': ['ി'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' * 10000,
                            'req_id:' + str(1), 50, True, True)
        engine.triton_resume(session_id)
        engine.triton_end(session_id)

    def test_history_crash_case_set(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction':
            'You are an AI assistant whose name is InternLM (书生·浦语).\n',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': ['ി'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

        # case1 - double end the ended session
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字',
                            'req_id:' + str(session_id), 1250)
        engine.triton_end(session_id)
        engine.triton_set_session(session_id)
        engine.triton_end(session_id)
        engine.triton_set_session(None)

        # case2 until the session_len is exhausted
        session_id = get_random_session_id()
        engine.triton_infer(session_id, '你好，你叫什么名字' + str(session_id),
                            'req_id:' + str(session_id), 200)
        for i in range(250):
            engine.triton_infer(session_id, '给我讲个200字的童话故事' + str(i),
                                'req_id:' + str(i), 200)
