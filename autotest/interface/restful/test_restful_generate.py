import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytest
import requests
from utils.config_utils import get_turbomind_model_list, get_workerid
from utils.run_restful_chat import start_restful_api, stop_restful_api
from utils.toolkit import encode_text, parse_sse_stream

DEFAULT_PORT = 23333


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num,
        'extra': '--communicator cuda-ipc'
    } for item in get_turbomind_model_list(tp_num)]


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = os.path.join(config.get('model_path'), model)

    pid, startRes = start_restful_api(config, param, model, model_path, 'turbomind', worker_id)

    worker_idx = get_workerid(worker_id)
    api_port = DEFAULT_PORT if worker_idx is None else DEFAULT_PORT + worker_idx

    request.cls.api_port = api_port
    request.node.model_name = model

    yield

    stop_restful_api(pid, startRes, param)


class _TestGenerateComprehensive:

    @pytest.fixture(autouse=True)
    def setup_api(self, request, config):
        self.api_url = f'http://127.0.0.1:{request.cls.api_port}/generate'
        self.headers = {'Content-Type': 'application/json'}

        test_name = request.node.name
        raw_model_name = getattr(request.node, 'model_name', 'unknown_model')
        safe_model_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', raw_model_name)

        log_base = config.get('log_path', './logs')
        self.log_dir = os.path.join(log_base, safe_model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'{test_name}.log')

    def _log_request_response(self, payload, response_data, stream_raw=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': payload,
            'response': response_data,
        }
        if stream_raw is not None:
            log_entry['stream_raw'] = stream_raw

        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f'[LOG WARN] Failed to write {self.log_file}: {e}')

    def _post(self, payload, stream=False):
        resp = requests.post(self.api_url, json=payload, headers=self.headers, stream=stream, timeout=60)
        resp.raise_for_status()

        if stream:
            raw_content = ''
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    raw_content += chunk.decode('utf-8')

            events = parse_sse_stream(raw_content)
            accumulated_text = ''
            output_ids = []
            stream_events_count = 0

            for event in events:
                if event == '[DONE]':
                    break
                try:
                    data_str = event.replace('data: ', '').strip()
                    if not data_str:
                        continue
                    data = json.loads(data_str)
                    delta = data.get('text', '')
                    if isinstance(delta, str):
                        accumulated_text += delta
                    ids = data.get('output_ids')
                    if isinstance(ids, list):
                        output_ids.extend(ids)
                    stream_events_count += 1
                except Exception:
                    continue

            fake_resp = {
                'text': accumulated_text,
                'output_ids': output_ids,
                'meta_info': {
                    'stream_events': stream_events_count
                }
            }
            self._log_request_response(payload, fake_resp, raw_content)

            class MockResp:

                def json(self):
                    return fake_resp

                @property
                def status_code(self):
                    return 200

            return MockResp()

        else:
            data = resp.json()
            self._log_request_response(payload, data)
            return resp

    def test_basic_generation(self):
        resp = self._post({'prompt': 'The sky is', 'max_tokens': 5})
        text = resp.json()['text']
        assert isinstance(text, str) and len(text.strip()) > 0

    def test_input_ids_mode(self, request, config):
        model = getattr(request.node, 'model_name')
        model_path = os.path.join(config.get('model_path'), model)
        try:
            input_ids = encode_text(model_path, 'Hi', add_bos=True)
        except Exception as e:
            pytest.skip(f'Tokenizer failed: {e}')
        resp = self._post({'input_ids': input_ids, 'max_tokens': 5})
        assert isinstance(resp.json()['text'], str) and len(resp.json()['text'].strip()) > 0

    def test_stop_string(self):
        resp = self._post({'prompt': 'Fruits: apple, banana, cherry', 'stop': ['cherry'], 'max_tokens': 20})
        text = resp.json()['text']
        assert 'cherry' not in text

    def test_streaming_mode(self):
        prompt = 'Count: 1, 2,'
        resp = self._post({'prompt': prompt, 'max_tokens': 8, 'stream': True}, stream=True)
        data = resp.json()
        text = data['text']
        assert isinstance(text, str)
        assert len(text.strip()) > 0
        assert '3' in text or '4' in text
        assert len(data['output_ids']) >= 5
        assert data['meta_info']['stream_events'] >= 5

    def test_streaming_incremental_correctness(self):
        prompt = 'A'
        raw_resp = requests.post(self.api_url,
                                 json={
                                     'prompt': prompt,
                                     'max_tokens': 5,
                                     'stream': True
                                 },
                                 headers=self.headers,
                                 stream=True,
                                 timeout=30)
        raw_resp.raise_for_status()

        full_text = ''
        event_count = 0
        for line in raw_resp.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: ') and '[DONE]' not in decoded:
                    try:
                        payload = json.loads(decoded[6:])
                        delta = payload.get('text', '')
                        full_text += delta
                        event_count += 1
                    except Exception:
                        continue

        assert len(full_text.strip()) > 0
        assert event_count >= 3

    def test_return_logprob(self):
        resp = self._post({'prompt': 'Paris is the capital of', 'max_tokens': 2, 'return_logprob': True})
        meta = resp.json()['meta_info']
        assert 'output_token_logprobs' in meta

    def test_same_session_id_allowed(self):
        sid = 9999
        self._post({'prompt': 'X', 'session_id': sid, 'max_tokens': 2})
        self._post({'prompt': 'Y', 'session_id': sid, 'max_tokens': 2})

    def test_empty_prompt_rejected(self):
        with pytest.raises(requests.HTTPError) as exc:
            self._post({'prompt': '', 'max_tokens': 5})
        assert exc.value.response.status_code == 400

    def test_stress_concurrent_requests(self):

        def single_request(idx):
            try:
                resp = requests.post(self.api_url,
                                     json={
                                         'prompt': f'Task {idx}: Hello',
                                         'max_tokens': 10,
                                         'stream': False
                                     },
                                     headers=self.headers,
                                     timeout=45)
                resp.raise_for_status()
                text = resp.json()['text']
                return len(text) > 0
            except Exception as e:
                print(f'[Concurrent Error {idx}]: {e}')
                return False

        success_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_request, i) for i in range(10)]
            for future in as_completed(futures):
                if future.result():
                    success_count += 1

        assert success_count >= 9, f'Too many failures: {10 - success_count}/10'

    def test_stress_long_prompt_and_generation(self):
        long_prompt = 'The quick brown fox jumps over the lazy dog. ' * 200
        resp = self._post({'prompt': long_prompt, 'max_tokens': 512, 'temperature': 0.7})
        text = resp.json()['text']
        assert len(text) > 100
        assert resp.status_code == 200

    def test_stress_max_tokens_boundary(self):
        resp1 = self._post({'prompt': 'Hi', 'max_tokens': 1})
        assert len(resp1.json()['text'].strip()) > 0

        resp2 = self._post({'prompt': 'Once upon a time', 'max_tokens': 1024})
        text2 = resp2.json()['text']
        assert len(text2.split()) >= 100

    def test_stress_streaming_under_load(self):

        def stream_request(idx):
            try:
                resp = requests.post(self.api_url,
                                     json={
                                         'prompt': f'Stream test {idx}',
                                         'max_tokens': 20,
                                         'stream': True
                                     },
                                     headers=self.headers,
                                     stream=True,
                                     timeout=60)
                full = ''
                for line in resp.iter_lines():
                    if line and b'data:' in line:
                        if b'[DONE]' in line:
                            break
                        try:
                            payload = json.loads(line.decode().replace('data: ', ''))
                            full += payload.get('text', '')
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                return len(full.strip()) > 0
            except Exception as e:
                print(f'[Stream Load Error {idx}]: {e}')
                return False

        results = []
        threads = []
        for i in range(5):
            t = threading.Thread(target=lambda i=i: results.append(stream_request(i)))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        success = sum(results)
        assert success >= 4, f'Only {success}/5 streaming succeeded under load'


@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=1), indirect=True)
class TestGenerateComprehensiveMultiModel_tp1(_TestGenerateComprehensive):
    pass


@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=2), indirect=True)
class TestGenerateComprehensiveMultiModel_tp2(_TestGenerateComprehensive):
    pass


@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=4), indirect=True)
class TestGenerateComprehensiveMultiModel_tp4(_TestGenerateComprehensive):
    pass
