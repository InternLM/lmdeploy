import pytest
import requests
from utils.constant import BACKEND_LIST, BASE_URL, RESTFUL_BASE_MODEL_LIST
from utils.restful_return_check import (
    assert_completions_batch_return,
    assert_completions_stream_return,
    assert_openai_invalid_request_error,
    get_client_and_model,
)

_COMPLETIONS_URL = f'{BASE_URL}/v1/completions'


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_BASE_MODEL_LIST)
class TestRestfulOpenAICompletions:

    def test_return(self, backend, model_case, openai_client_and_model):
        print(f'[test_return] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            temperature=0.01,
        )
        item = response.model_dump()
        completion_tokens = item['usage']['completion_tokens']
        assert completion_tokens > 0
        assert completion_tokens <= 17
        assert completion_tokens >= 16
        assert item.get('choices')[0].get('finish_reason') in ['length']
        print(f'[test_return] model_name={model_name!r} last_usage={item.get("usage")!r} '
              f'finish_reason={item.get("choices")[0].get("finish_reason")!r}')
        assert_completions_batch_return(item, model_name)

    def test_return_streaming(self, backend, model_case, openai_client_and_model):
        print(f'[test_return_streaming] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            stream=True,
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_return_streaming] model_name={model_name!r} stream_chunks={len(outputList)}')
        assert_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_completions_stream_return(outputList[index], model_name)

    def test_max_tokens(self, backend, model_case, openai_client_and_model):
        print(f'[test_max_tokens] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            temperature=0.01,
        )
        item = response.model_dump()
        completion_tokens = item['usage']['completion_tokens']
        assert completion_tokens > 0
        assert completion_tokens <= 17
        assert completion_tokens >= 16
        assert item.get('choices')[0].get('finish_reason') in ['length']
        print(f'[test_max_tokens] completion_tokens={completion_tokens} '
              f'finish_reason={item.get("choices")[0].get("finish_reason")!r}')

    def test_single_stopword(self, backend, model_case, openai_client_and_model):
        print(f'[test_single_stopword] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            max_tokens=200,
            stop=' Shanghai',
            temperature=0.01,
        )
        item = response.model_dump()
        assert ' Shanghai' not in item.get('choices')[0].get('text')
        assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_single_stopword] finish_reason={item.get("choices")[0].get("finish_reason")!r} '
              f'text_preview={((item.get("choices")[0].get("text")) or "")[:120]!r}')

    def test_array_stopwords(self, backend, model_case, openai_client_and_model):
        print(f'[test_array_stopwords] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            max_tokens=200,
            stop=[' Shanghai', ' city', ' China'],
            temperature=0.01,
        )
        item = response.model_dump()
        assert ' Shanghai' not in item.get('choices')[0].get('text')
        assert ' city' not in item.get('choices')[0].get('text')
        assert ' China' not in item.get('choices')[0].get('text')
        assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_array_stopwords] finish_reason={item.get("choices")[0].get("finish_reason")!r} '
              f'text_preview={((item.get("choices")[0].get("text")) or "")[:120]!r}')

    def test_completions_stream(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            stream=True,
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 1):
            output = outputList[index]
            assert output.get('model') == model_name
            for message in output.get('choices'):
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream] last_finish_reason={output_last.get("choices")[0].get("finish_reason")!r}')

    def test_completions_stream_stopword(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream_stopword] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Beijing is',
            stream=True,
            stop=' is',
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream_stopword] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert output.get('model') == model_name
            assert output.get('object') == 'text_completion'
            for message in output.get('choices'):
                assert ' is' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream_stopword] last_finish_reason='
              f'{output_last.get("choices")[0].get("finish_reason")!r}')

    def test_completions_stream_stopwords(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream_stopwords] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Beijing is',
            stream=True,
            stop=[' Beijing', ' city', ' China'],
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream_stopwords] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert output.get('model') == model_name
            assert output.get('object') == 'text_completion'
            for message in output.get('choices'):
                assert ' Beijing' not in message.get('text')
                assert ' city' not in message.get('text')
                assert ' China' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream_stopwords] last_finish_reason='
              f'{output_last.get("choices")[0].get("finish_reason")!r}')

    def test_batch_prompt_order(self, backend, model_case, openai_client_and_model):
        print(f'[test_batch_prompt_order] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt=['你好', '今天天气怎么样', '你是谁', '帮我写一首以梅花为主题的五言律诗', '5+2等于多少'],
            max_tokens=400,
            extra_body={'min_new_tokens': 50},
        )
        item = response.model_dump()
        print(f'[test_batch_prompt_order] batch_response={item!r}')
        assert '天' in item.get('choices')[1].get('text') or '雨' in item.get('choices')[1].get(
            'text') or '伞' in item.get('choices')[1].get('text'), item.get('choices')[1].get('text')
        assert '梅' in item.get('choices')[3].get('text') or '对仗' in item.get('choices')[3].get(
            'text') or '仄' in item.get('choices')[3].get('text') or '诗' in item.get('choices')[3].get(
                'text'), item.get('choices')[3].get('text')
        assert '7' in item.get('choices')[4].get('text') or '5+2' in item.get('choices')[4].get('text'), item.get(
            'choices')[4].get('text')

    @pytest.mark.parametrize(
        'invalid_payload',
        [
            pytest.param({'max_tokens': 0}, id='max_tokens_zero'),
            pytest.param({'max_tokens': -1}, id='max_tokens_negative'),
        ],
    )
    def test_rejects_invalid_request_parameters(
            self, backend, model_case, openai_client_and_model, invalid_payload):
        """Invalid types/ranges must return HTTP 400 (raw JSON, not SDK)."""
        _, model_name = openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            json={
                'model': model_name,
                'prompt': 'Hi, pls intro yourself',
                **invalid_payload,
            },
            timeout=30,
        )
        assert_openai_invalid_request_error(resp)
