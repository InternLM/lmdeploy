import random
import string

import pytest
from utils.triton_engine import Engine

from lmdeploy.serve.turbomind import chatbot

SERVER_ADDR = 'localhost:33337'


def get_random_session_id():
    characters = string.digits
    random_chars = ''.join(random.choice(characters) for i in range(6))
    return int(random_chars)


@pytest.mark.order(8)
@pytest.mark.triton
@pytest.mark.flaky(reruns=0)
class TestTritonInterface:

    def test_return_info(self):
        dict_info = {}

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)

        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id, '你好，你叫什么名字',
                                                  'req_id:', 50, True, False)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert len(res) > 0, res
        assert tokens <= 51, res

    def test_stopword(self):
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
            'stop_words': [' city'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id, 'Shanghai is',
                                                  'req_id:', 200, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert ' city' not in res
        assert tokens <= 201, res

    def test_stopwords(self):
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
            'stop_words': [' Shanghai', ' city', ' China'],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id, 'Shanghai is',
                                                  'req_id:', 200, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert ' Shanghai' not in res, res
        assert ' city' not in res, res
        assert ' China' not in res, res
        assert tokens <= 201, res

    def test_session_out_of_limit(self):
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
            'session_len': 120,
            'stop_words': [],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id, 'Shanghai is',
                                                  'req_id:', 80, True, True)
        assert status == chatbot.StatusCode.TRITON_SESSION_OUT_OF_LIMIT, status
        assert tokens == 0, res

    def test_meta_instruction(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction': 'You are an AI assistant whose name is puyu.',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': [],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id,
                                                  'What is your name?',
                                                  'req_id:', 50, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert ' puyu' or ' Puyu' in res, res
        assert tokens <= 51, res

    def test_resume_session(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction': 'You are an AI assistant whose name is puyu.',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': [],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id,
                                                  'What is your name?',
                                                  'req_id:', 50, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert ' puyu' or ' Puyu' in res, res
        assert tokens <= 51, res
        engine.triton_resume(session_id)
        status, res, tokens = engine.triton_infer(session_id,
                                                  'What is your name?',
                                                  'req_id:', 50, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        assert ' puyu' or ' Puyu' in res, res
        assert tokens <= 51, res

    def test_cancel_session(self):
        dict_info = {
            'assistant': '<|Assistant|>െ',
            'eoa': 'ി\n ',
            'eoh': '\n ',
            'eosys': '\n ',
            'meta_instruction': 'You are an AI assistant whose name is puyu.',
            'meta_tag': '<BOS>',
            'model_name': 'puyu',
            'repetition_penalty': 1.02,
            'replace_token': 'ി',
            'session_len': 8192,
            'stop_words': [],
            'system': '<|System|>െ',
            'temperature': 1,
            'top_k': 40,
            'top_p': 0.8,
            'user': '<|Human|>െ'
        }

        engine = Engine(SERVER_ADDR, log_level='ERROR', **dict_info)
        session_id = get_random_session_id()
        status, res, tokens = engine.triton_infer(session_id,
                                                  'What is your name?',
                                                  'req_id:', 50, True, True)
        assert status == chatbot.StatusCode.TRITON_STREAM_END, status
        status = engine.triton_cancel(session_id)
        assert status == chatbot.StatusCode.TRITON_SERVER_ERR, status
