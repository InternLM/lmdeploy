import pytest
from utils.restful_return_check import assert_completions_batch_return, assert_completions_stream_return

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333
MODEL = 'internlm/internlm2_5-20b'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


class TestRestfulInterfaceBase:

    @pytest.mark.internlm2_5
    def test_get_model(self, config):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        assert model_name == '/'.join([config.get('model_path'), MODEL]), api_client.available_models

    def test_encode(self):
        api_client = APIClient(BASE_URL)
        input_ids1, length1 = api_client.encode('Hi, pls intro yourself')
        input_ids2, length2 = api_client.encode('Hi, pls intro yourself', add_bos=False)
        input_ids3, length3 = api_client.encode('Hi, pls intro yourself', do_preprocess=True)
        input_ids4, length4 = api_client.encode('Hi, pls intro yourself', do_preprocess=True, add_bos=False)
        input_ids5, length5 = api_client.encode('Hi, pls intro yourself' * 100, add_bos=False)
        assert len(input_ids1) == length1 and length1 > 0
        assert len(input_ids2) == length2 and length2 > 0
        assert len(input_ids3) == length3 and length3 > 0
        assert len(input_ids4) == length4 and length4 > 0
        assert len(input_ids5) == length5 and length5 > 0
        assert length1 == length2 + 1
        assert input_ids2 == input_ids1[1:]
        assert input_ids1[0] == 1 and input_ids3[0] == 1
        assert length5 == length2 * 100
        assert input_ids5 == input_ids2 * 100

    def test_return(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for item in api_client.completions_v1(
                model=model_name,
                prompt='Hi, pls intro yourself',
                max_tokens=16,
                temperature=0.01,
        ):
            completion_tokens = item['usage']['completion_tokens']
            assert completion_tokens > 0
            assert completion_tokens <= 17
            assert completion_tokens >= 16
            assert item.get('choices')[0].get('finish_reason') in ['length']
        assert_completions_batch_return(item, model_name)

    def test_return_streaming(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for item in api_client.completions_v1(model=model_name,
                                              prompt='Hi, pls intro yourself',
                                              max_tokens=16,
                                              stream=True,
                                              temperature=0.01):
            outputList.append(item)
        assert_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_completions_stream_return(outputList[index], model_name)

    def test_max_tokens(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for item in api_client.completions_v1(model=model_name,
                                              prompt='Hi, pls intro yourself',
                                              max_tokens=16,
                                              temperature=0.01):
            completion_tokens = item['usage']['completion_tokens']
            assert completion_tokens > 0
            assert completion_tokens <= 17
            assert completion_tokens >= 16
            assert item.get('choices')[0].get('finish_reason') in ['length']

    def test_single_stopword(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for item in api_client.completions_v1(model=model_name,
                                              prompt='Shanghai is',
                                              max_tokens=200,
                                              stop=' Shanghai',
                                              temperature=0.01):
            assert ' Shanghai' not in item.get('choices')[0].get('text')
            assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']

    def test_array_stopwords(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for item in api_client.completions_v1(model=model_name,
                                              prompt='Shanghai is',
                                              max_tokens=200,
                                              stop=[' Shanghai', ' city', ' China'],
                                              temperature=0.01):
            assert ' Shanghai' not in item.get('choices')[0].get('text')
            assert ' city' not in item.get('choices')[0].get('text')
            assert ' China' not in item.get('choices')[0].get('text')
            assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']

    def test_completions_stream(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.completions_v1(model=model_name, prompt='Shanghai is', stream='true',
                                                temperature=0.01):
            outputList.append(output)

        for index in range(1, len(outputList) - 1):
            output = outputList[index]
            assert (output.get('model') == model_name)
            for message in output.get('choices'):
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']

    def test_completions_stream_stopword(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.completions_v1(model=model_name,
                                                prompt='Beijing is',
                                                stream='true',
                                                stop=' is',
                                                temperature=0.01):
            outputList.append(output)

        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert (output.get('model') == model_name)
            assert (output.get('object') == 'text_completion')
            for message in output.get('choices'):
                assert ' is' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']

    def test_completions_stream_stopwords(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        outputList = []
        for output in api_client.completions_v1(model=model_name,
                                                prompt='Beijing is',
                                                stream='true',
                                                stop=[' Beijing', ' city', ' China'],
                                                temperature=0.01):
            outputList.append(output)

        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert (output.get('model') == model_name)
            assert (output.get('object') == 'text_completion')
            for message in output.get('choices'):
                assert ' Beijing' not in message.get('text')
                assert ' city' not in message.get('text')
                assert ' China' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']

    def test_batch_prompt_order(self):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        for item in api_client.completions_v1(model=model_name,
                                              prompt=['你好', '今天天气怎么样', '你是谁', '帮我写一首以梅花为主题的五言律诗', '5+2等于多少'],
                                              max_tokens=200):
            assert '天气' in item.get('choices')[1].get('text')
            assert '梅' in item.get('choices')[3].get('text')
            assert '7' in item.get('choices')[4].get('text')
