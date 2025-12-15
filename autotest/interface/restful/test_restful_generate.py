import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import pytest
import requests
from transformers import AutoTokenizer
from utils.toolkit import encode_text, parse_sse_stream

BASE_HTTP_URL = 'http://127.0.0.1'
DEFAULT_PORT = 23333
MODEL_LIST = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-30B-A3B']
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])


@pytest.mark.parametrize('model_name', MODEL_LIST)
class TestGenerateComprehensive:

    @pytest.fixture(autouse=True)
    def setup_api(self, request, config, model_name):
        self.api_url = f'{BASE_URL}/generate'
        self.headers = {'Content-Type': 'application/json'}
        self.model_name = model_name

        test_name = request.node.name
        safe_test_name = re.sub(r'[^\w\.-]', '_', test_name)
        safe_model_name = self.model_name.replace('/', '_')
        log_base = config.get('log_path', './logs')
        self.log_dir = os.path.join(log_base, safe_model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'{safe_test_name}.log')

    def _log_request_response(self, payload, response_data, stream_raw=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'request': payload,
            'response': response_data,
        }
        if stream_raw is not None:
            log_entry['stream_raw'] = stream_raw

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f'[LOG WARN] Failed to write {self.log_file}: {e}')

    def _post(self, payload, stream=False):
        if 'model' not in payload:
            payload['model'] = self.model_name

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
                except Exception as e:
                    print(f'Error parsing stream event: {e}')
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

    def _validate_generation_response(self,
                                      data: Dict[str, Any],
                                      expected_fields: List[str] = None,
                                      validate_tokens: bool = True,
                                      expect_logprobs: bool = False,
                                      validate_experts: bool = False) -> None:
        assert isinstance(data, dict), f'Response should be a dict, got {type(data)}'

        required_fields = ['text']
        for field in required_fields:
            assert field in data, f'Missing required field: {field}'
            assert data[field] is not None, f'Field {field} should not be None'

        assert isinstance(data['text'], str), \
            f"text should be string, got {type(data['text'])}"

        if validate_experts:
            assert 'routed_experts' in data[
                'meta_info'], "Response should contain 'routed_experts' when validate_experts=True"

            experts_data = data['meta_info']['routed_experts']

            assert isinstance(experts_data, list)
            assert len(experts_data) > 0

            total_steps = len(experts_data)

            for step_idx in range(total_steps):
                token_experts = experts_data[step_idx]

                assert isinstance(token_experts, list)
                assert len(token_experts) > 0

                for layer_idx in range(len(token_experts)):
                    layer_experts = token_experts[layer_idx]

                    assert isinstance(layer_experts, list)
                    assert len(layer_experts) == 8

                    for expert_idx, expert_id in enumerate(layer_experts):
                        assert isinstance(expert_id, int)
                        assert 0 <= expert_id < 256, f'Invalid expert_id: {expert_id}. Must be in [0, 256)'

        if validate_tokens:
            assert 'output_ids' in data, "Response should contain 'output_ids'"
            output_ids = data['output_ids']

            assert isinstance(output_ids, list), \
                f'output_ids should be list, got {type(output_ids)}'
            assert len(output_ids) >= 0, 'output_ids should not be empty'

            for i, token_id in enumerate(output_ids):
                assert isinstance(token_id, int), \
                    f'output_ids[{i}] should be int, got {type(token_id)}'

            if 'meta_info' in data:
                meta = data['meta_info']
                assert isinstance(meta, dict), 'meta_info should be dict'

                if 'completion_tokens' in meta:
                    assert meta['completion_tokens'] == len(output_ids), \
                        f"meta.completion_tokens ({meta['completion_tokens']}) " \
                        f'should equal len(output_ids) ({len(output_ids)})'

        if expect_logprobs:
            assert 'meta_info' in data, \
                "Response should contain 'meta_info' when expecting logprobs"
            meta = data['meta_info']
            assert isinstance(meta, dict)

            assert 'output_token_logprobs' in meta, \
                "meta_info missing 'output_token_logprobs'"
            logprobs_data = meta['output_token_logprobs']

            assert isinstance(logprobs_data, list), \
                'output_token_logprobs should be a list'
            assert len(logprobs_data) > 0, \
                'output_token_logprobs should not be empty'

            if 'output_ids' in data:
                assert len(logprobs_data) == len(data['output_ids']), \
                    f'Logprobs outer list length ({len(logprobs_data)}) != ' \
                    f"Output IDs length ({len(data['output_ids'])})"

            for idx, item in enumerate(logprobs_data):
                assert isinstance(item, list), \
                    f'Logprobs item at index {idx} should be a list, got {type(item)}'
                assert len(item) == 2, \
                    f'Logprobs item at index {idx} should have 2 elements ' \
                    f'[logprob, token_id], got {len(item)}'

                logprob_val = item[0]
                assert isinstance(logprob_val, (float, int)), \
                    f'Logprob value at [{idx}][0] should be number, ' \
                    f'got {type(logprob_val)}'
                assert logprob_val <= 0, \
                    f'Logprob value should be <= 0, got {logprob_val}'

                token_id_in_logprob = item[1]
                assert isinstance(token_id_in_logprob, int), \
                    f'Token ID in logprobs at [{idx}][1] should be int, ' \
                    f'got {type(token_id_in_logprob)}'

                if 'output_ids' in data and idx < len(data['output_ids']):
                    assert token_id_in_logprob == data['output_ids'][idx], \
                        f'Token ID mismatch at index {idx}: output_ids has ' \
                        f"{data['output_ids'][idx]}, but logprobs has " \
                        f'{token_id_in_logprob}'

        if expected_fields:
            for field in expected_fields:
                assert field in data, f'Missing expected field: {field}'

        if 'error' in data:
            assert not data['error'], f"Response contains error: {data['error']}"
        if 'code' in data and data['code'] != 0:
            assert False, f"Response contains error code: {data['code']}"

    def test_basic_generation(self):
        print(f'\n[Model: {self.model_name}] Running basic generation test')
        test_cases = [{
            'name': 'simple prompt',
            'payload': {
                'prompt': 'The sky is',
                'max_tokens': 5
            },
        }, {
            'name': 'prompt with spaces',
            'payload': {
                'prompt': '  Hello world  ',
                'max_tokens': 3
            },
        }, {
            'name': 'unicode prompt',
            'payload': {
                'prompt': 'Hello, world',
                'max_tokens': 3
            },
        }, {
            'name': 'longer generation',
            'payload': {
                'prompt': 'Once upon a time',
                'max_tokens': 10
            },
        }]

        for test_case in test_cases:
            test_name = test_case['name']
            print(f'\n[Test: {test_name}]')

            resp = self._post(test_case['payload'])
            data = resp.json()

            self._validate_generation_response(data=data, validate_tokens=True)

            prompt = test_case['payload']['prompt']
            generated_text = data['text']
            assert generated_text != prompt.strip(), \
                f"Generated text should be different from prompt: '{generated_text}'"

            if 'output_ids' in data:
                output_ids = data['output_ids']
                max_tokens = test_case['payload']['max_tokens']
                max_allowed = max_tokens + 1

                assert len(output_ids) <= max_allowed, \
                    f'Too many tokens generated: {len(output_ids)} > {max_allowed}'

                meta = data.get('meta_info', {})
                finish_type = meta.get('finish_reason', {}).get('type')
                if len(output_ids) >= max_tokens and finish_type != 'length':
                    print(f'[WARN] Generated {len(output_ids)} tokens but '
                          f"finish_reason is not 'length': {finish_type}")

            print(f"  Generated text: '{generated_text[:50]}...'")
            print(f"  Generated tokens: {len(data.get('output_ids', []))}")

    def test_input_ids_mode(self, config):
        print(f'\n[Model: {self.model_name}] Running input_ids mode test')
        model_path = os.path.join(config.get('model_path'), self.model_name)

        test_cases = [{
            'name': 'simple text',
            'text': 'Hello world',
            'max_tokens': 5,
            'expected_min_text': 3
        }, {
            'name': 'question',
            'text': 'What is the meaning of life?',
            'max_tokens': 8,
            'expected_min_text': 5
        }, {
            'name': 'short input',
            'text': 'Yes',
            'max_tokens': 3,
            'expected_min_text': 1
        }]

        for test_case in test_cases:
            test_name = test_case['name']
            print(f'\n[Test: input_ids - {test_name}]')

            try:
                input_ids = encode_text(model_path, test_case['text'])
            except Exception as e:
                pytest.skip(f'Tokenizer failed for {test_case["name"]}: {e}')

            assert isinstance(input_ids, list), \
                f'input_ids should be list, got {type(input_ids)}'
            assert len(input_ids) > 0, 'input_ids should not be empty'
            for i, token_id in enumerate(input_ids):
                assert isinstance(token_id, int), \
                    f'input_ids[{i}] should be int, got {type(token_id)}'
                assert token_id >= 0, \
                    f'input_ids[{i}] should be >= 0, got {token_id}'

            resp = self._post({'input_ids': input_ids, 'max_tokens': test_case['max_tokens']})
            data = resp.json()

            self._validate_generation_response(data=data, validate_tokens=True)

            generated_text = data['text']
            try:
                generated_text.encode('utf-8')
            except UnicodeEncodeError:
                pytest.fail(f'Generated text contains invalid UTF-8 characters: '
                            f'{generated_text[:100]}')

            print(f'  Input tokens: {len(input_ids)}')
            print(f"  Output tokens: {len(data.get('output_ids', []))}")
            print(f"  Generated text: '{generated_text[:50]}...'")

    def test_conflict_prompt_and_input_ids(self):
        print(f'\n[Model: {self.model_name}] Running conflict test')
        test_cases = [{
            'name':
            'both provided',
            'payload': {
                'prompt': 'Hello world',
                'input_ids': [1, 2, 3, 4, 5],
                'max_tokens': 5
            },
            'expected_status':
            400,
            'expected_error_keywords': [
                'conflict', 'both', 'either', 'cannot', 'mutually exclusive', 'specify exactly one', 'prompt',
                'input_ids'
            ]
        }, {
            'name':
            'prompt with empty input_ids',
            'payload': {
                'prompt': 'Test',
                'input_ids': [],
                'max_tokens': 3
            },
            'expected_status':
            400,
            'expected_error_keywords': ['conflict', 'invalid', 'empty', 'specify exactly one', 'prompt', 'input_ids']
        }, {
            'name':
            'empty prompt with input_ids',
            'payload': {
                'prompt': '',
                'input_ids': [100, 200, 300],
                'max_tokens': 3
            },
            'expected_status':
            400,
            'expected_error_keywords': ['conflict', 'empty', 'invalid', 'specify exactly one', 'prompt', 'input_ids']
        }]

        for test_case in test_cases:
            test_name = test_case['name']
            print(f'\n[Test: conflict - {test_name}]')

            try:
                resp = requests.post(self.api_url, json=test_case['payload'], headers=self.headers, timeout=30)

                assert resp.status_code == test_case['expected_status'], \
                    f"Expected status {test_case['expected_status']}, " \
                    f'got {resp.status_code}'

                error_data = resp.json()
                assert 'error' in error_data or 'message' in error_data, \
                    "Error response should contain 'error' or 'message' field"

                error_msg = ''
                if 'error' in error_data:
                    error_msg = str(error_data['error']).lower()
                elif 'message' in error_data:
                    error_msg = str(error_data['message']).lower()

                keywords_found = any(keyword in error_msg for keyword in test_case['expected_error_keywords'])

                if not keywords_found:
                    has_both_fields = ('prompt' in error_msg and 'input_ids' in error_msg)
                    has_exclusivity = any(phrase in error_msg for phrase in [
                        'only one', 'specify exactly', 'cannot both', 'mutually exclusive', 'exactly one',
                        'must specify'
                    ])
                    if has_both_fields and has_exclusivity:
                        keywords_found = True

                assert keywords_found, \
                    f'Error message should indicate conflict between prompt and ' \
                    f'input_ids, got: {error_msg}'

                assert 'text' not in error_data, \
                    "Error response should not contain 'text' field"
                assert 'output_ids' not in error_data, \
                    "Error response should not contain 'output_ids' field"

                print(f'  Got expected error: {error_msg[:100]}...')

            except Exception as e:
                print(f'  Unexpected error: {e}')
                raise

    @pytest.mark.logprob
    def test_input_ids_with_logprob(self, config):
        print(f'\n[Model: {self.model_name}] Running input_ids with logprob test')
        model_path = os.path.join(config.get('model_path'), self.model_name)

        test_cases = [{
            'name': 'basic logprob',
            'text': 'The weather is',
            'max_tokens': 3,
            'expected_min_text': 3
        }, {
            'name': 'single token generation',
            'text': 'Hello',
            'max_tokens': 1,
            'expected_min_text': 1
        }, {
            'name': 'multiple tokens with logprob',
            'text': 'Artificial intelligence is',
            'max_tokens': 5,
            'expected_min_text': 5
        }]

        for test_case in test_cases:
            test_name = test_case['name']
            print(f'\n[Test: logprob - {test_name}]')

            try:
                input_ids = encode_text(model_path, test_case['text'])
            except Exception as e:
                pytest.skip(f'Tokenizer failed for {test_case["name"]}: {e}')

            request_payload = {'input_ids': input_ids, 'max_tokens': test_case['max_tokens'], 'return_logprob': True}

            resp = self._post(request_payload)
            data = resp.json()

            self._validate_generation_response(data=data, validate_tokens=True, expect_logprobs=True)

            assert 'meta_info' in data, \
                "Response should contain 'meta_info' when return_logprob=True"
            meta = data['meta_info']

            assert 'output_token_logprobs' in meta, \
                "meta_info should contain 'output_token_logprobs'"
            logprobs = meta['output_token_logprobs']

            logprob_values = []

            for i, item in enumerate(logprobs):
                logprob_values.append(item[0])

            avg_logprob = sum(logprob_values) / len(logprob_values)
            if avg_logprob < -10.0:
                pytest.fail(f'Generation confidence critically low '
                            f'(Avg: {avg_logprob:.2f})')

            generated_text = data.get('text', '')
            print(f'  Generated tokens: {len(logprob_values)}')
            print(f'  Avg Logprob: {avg_logprob:.3f}')
            print(f"  Generated text: '{generated_text[:50]}...'")

    def test_stop_str_with_include_flag(self):
        print(f'\n[Model: {self.model_name}] Running stop_str with include flag test')
        test_cases = [{
            'name': 'simple stop word',
            'prompt': 'Count: 1, 2, 3, ',
            'stop_word': '6',
            'max_tokens': 10,
        }]

        for test_case in test_cases:
            test_name = test_case['name']
            print(f'\n[Test: stop_str - {test_name}]')

            prompt = test_case['prompt']
            stop_word = test_case['stop_word']
            max_tokens = test_case['max_tokens']

            print('  Testing EXCLUDE mode (include_stop=False)...')
            resp1 = self._post({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'stop': [stop_word],
                'include_stop_str_in_output': False,
                'return_logprob': True
            })

            self._validate_generation_response(resp1.json())
            text_exclude = resp1.json()['text']
            assert stop_word not in text_exclude, \
                f"Stop word '{stop_word}' should NOT be in output when include_stop=False"

            print('  Testing INCLUDE mode (include_stop=True)...')
            resp2 = self._post({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'stop': [stop_word],
                'include_stop_str_in_output': True,
                'return_logprob': True
            })

            self._validate_generation_response(resp2.json())
            text_include = resp2.json()['text']
            assert stop_word in text_include, \
                f"Stop word '{stop_word}' should be in output when include_stop=True"

    def test_streaming_mode(self):
        print(f'\n[Model: {self.model_name}] Running streaming mode test')
        prompt = 'Count: 1, 2,'

        resp = self._post({'prompt': prompt, 'max_tokens': 8, 'stream': True}, stream=True)
        assert resp.status_code == 200
        data = resp.json()

        text = data['text']
        output_ids = data['output_ids']
        meta = data['meta_info']

        assert isinstance(text, str) and len(text.strip()) > 0, \
            'Generated text cannot be empty'
        assert len(output_ids) >= 3, 'Output token count should be reasonable'

        import re
        count_matches = len(re.findall(r'\b[3-9]\b', text))
        assert count_matches >= 2, \
            f'Expected continuation of counting, but not enough numbers found ' \
            f'(found {count_matches})'

        stream_events = meta.get('stream_events', [])
        assert stream_events >= len(output_ids), \
            'Streaming event count should not be less than output token count'

        print(f"  Generated text: '{text}'")
        print(f'  Output tokens: {len(output_ids)}, '
              f'Stream events: {stream_events}')

    def test_streaming_incremental_correctness(self):
        print(f'\n[Model: {self.model_name}] Running streaming incremental correctness test')
        prompt = 'The sky is '

        raw_resp = requests.post(self.api_url,
                                 json={
                                     'prompt': prompt,
                                     'max_tokens': 10,
                                     'stream': True
                                 },
                                 headers=self.headers,
                                 stream=True,
                                 timeout=30)
        raw_resp.raise_for_status()

        full_text_from_delta = ''
        tokens_from_delta = []
        event_count = 0

        print('  Streaming chunks:')
        for line in raw_resp.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: ') and '[DONE]' not in line_str:
                    try:
                        json_str = line_str[6:]
                        payload = json.loads(json_str)

                        delta_text = payload.get('text', '')
                        token_id = payload.get('token_id')

                        full_text_from_delta += delta_text
                        if token_id is not None:
                            tokens_from_delta.append(token_id)

                        event_count += 1
                        if delta_text.strip():
                            print(f"    +'{delta_text}'")

                    except Exception as e:
                        print(f'    [Parse warning]: {e}')
                        continue

        assert len(full_text_from_delta.strip()) > 0, \
            'Assembled text from streaming deltas is empty'
        assert event_count >= 3, \
            f'Too few streaming events received ({event_count}), ' \
            f'connection might be interrupted'

        print(f"  Final assembled text: '{full_text_from_delta}'")
        print(f'  Total events received: {event_count}')

    @pytest.mark.logprob
    def test_return_logprob(self):
        print(f'\n[Model: {self.model_name}] Running return_logprob test')

        resp = self._post({'prompt': 'Paris is the capital of', 'max_tokens': 2, 'return_logprob': True})
        data = resp.json()

        self._validate_generation_response(data, validate_tokens=True, expect_logprobs=True)

        print(f"  Generated text: '{data['text']}'")

    def test_same_session_id_allowed(self):
        print(f'\n[Model: {self.model_name}] Running same session_id test')
        sid = 9999

        resp1 = self._post({'prompt': 'First message:', 'session_id': sid, 'max_tokens': 2})
        resp2 = self._post({'prompt': 'Second message:', 'session_id': sid, 'max_tokens': 2})

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        data1 = resp1.json()
        data2 = resp2.json()

        self._validate_generation_response(data1)
        self._validate_generation_response(data2)

        text1 = data1['text'].strip()
        text2 = data2['text'].strip()
        assert text1 != text2

        print(f"  First response: '{data1['text']}'")
        print(f"  Second response: '{data2['text']}'")

    def test_empty_prompt_rejected(self):
        print(f'\n[Model: {self.model_name}] Running empty prompt test')

        with pytest.raises(requests.HTTPError) as exc:
            self._post({'prompt': '', 'max_tokens': 5})

        assert exc.value.response.status_code == 400

        try:
            error_response = exc.value.response.json()
            print(f'  Error response: {error_response}')
            assert 'error' in error_response or 'message' in error_response
        except json.JSONDecodeError:
            print(f'  Non-JSON error: {exc.value.response.text[:100]}')

    def test_input_ids_rejected(self):
        print(f'\n[Model: {self.model_name}] Running input_ids invalid cases test')

        invalid_cases = [{
            'case': {
                'input_ids': [],
                'max_tokens': 5
            },
            'desc': 'Empty input_ids list'
        }, {
            'case': {
                'input_ids': 'not_a_list',
                'max_tokens': 5
            },
            'desc': 'input_ids is a string, not list'
        }, {
            'case': {
                'max_tokens': 5
            },
            'desc': 'Missing input_ids field'
        }]

        for invalid_case in invalid_cases:
            test_desc = invalid_case['desc']
            payload = invalid_case['case']

            with pytest.raises(requests.HTTPError) as exc_info:
                self._post(payload)

            response = exc_info.value.response
            assert response.status_code in [400, 422], (f"Bad Request for case '{test_desc}', "
                                                        f'but got {response.status_code}')

    def test_stress_concurrent_requests(self):
        print(f'\n[Model: {self.model_name}] Running stress concurrent requests test')

        def single_request(idx):
            start_time = time.time()
            try:
                resp = requests.post(self.api_url,
                                     json={
                                         'prompt': f'Hello, task {idx}',
                                         'max_tokens': 5,
                                         'stream': False
                                     },
                                     headers=self.headers,
                                     timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if 'text' in data and len(data['text'].strip()) > 0:
                    latency = time.time() - start_time
                    return {'success': True, 'latency': latency}
                else:
                    return {'success': False, 'error': 'Empty response'}

            except Exception as e:
                return {'success': False, 'error': str(e)}

        success_count = 0
        total_latency = 0
        failures = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_request, i) for i in range(20)]

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result['success']:
                    success_count += 1
                    total_latency += result['latency']
                    print(f"  Req {i}: ✓ (Latency: {result['latency']:.2f}s)")
                else:
                    failures.append(result['error'])
                    print(f'  Req {i}: ✗')

        success_rate = success_count / 20
        assert success_rate == 1.0, \
            f'Stress test failed: success rate {success_rate*100}% < 80%'

        if success_count > 0:
            avg_latency = total_latency / success_count
            assert avg_latency < 5.0, \
                f'Average latency too high: {avg_latency:.2f}s'
            print(f'  Performance: Avg Latency={avg_latency:.2f}s')

        print(f'  Summary: {success_count}/20 succeeded')

    def test_stress_long_prompt_and_generation(self):
        print(f'\n[Model: {self.model_name}] Running stress long prompt test')

        long_prompt = 'Summarize: The quick brown fox jumps over the lazy dog. ' * 100

        resp = self._post({'prompt': long_prompt, 'max_tokens': 512, 'temperature': 0.7})

        data = resp.json()
        self._validate_generation_response(data=data, validate_tokens=True)

    def test_stress_streaming_under_load(self):
        print(f'\n[Model: {self.model_name}] Running stress streaming under load test')

        def stream_request(idx):
            try:
                resp = requests.post(self.api_url,
                                     json={
                                         'prompt': f'Stream load test {idx}',
                                         'max_tokens': 10,
                                         'stream': True
                                     },
                                     headers=self.headers,
                                     stream=True,
                                     timeout=30)

                assert resp.status_code == 200
                content_type = resp.headers.get('Content-Type', '')
                assert 'text/event-stream' in content_type or \
                    'application/x-ndjson' in content_type

                full_text = ''
                event_count = 0
                for line in resp.iter_lines():
                    if line and line.startswith(b'data:'):
                        event_count += 1
                        if b'[DONE]' in line:
                            break
                        try:
                            payload = json.loads(line.decode().replace('data: ', '', 1))
                            full_text += payload.get('text', '')
                        except Exception:
                            pass

                assert len(full_text) > 0
                assert event_count >= 3

                return True

            except Exception as e:
                print(f'  Stream {idx} error: {e}')
                return False

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(stream_request, i) for i in range(10)]
            results = [f.result() for f in futures]

        success_count = sum(results)

        assert success_count == 10, \
            f'Concurrent streaming test failure rate too high: {success_count}/10'

        print(f'  Streaming under load: {success_count}/10 succeeded')

    def test_temperature_parameter(self):
        print(f'\n[Model: {self.model_name}] Running temperature parameter test')
        prompt = 'The capital of France is'

        resp_low = self._post({'prompt': prompt, 'max_tokens': 10, 'temperature': 0.1, 'stream': False})
        resp_high = self._post({'prompt': prompt, 'max_tokens': 10, 'temperature': 0.9, 'stream': False})

        data_low = resp_low.json()
        data_high = resp_high.json()

        self._validate_generation_response(data=data_low, validate_tokens=True)
        self._validate_generation_response(data=data_high, validate_tokens=True)

        assert 'Paris' in data_low['text'] or \
            'paris' in data_low['text'].lower(), \
            "Low temperature didn't answer correct capital"
        assert data_low['text'] != data_high['text'], \
            'High and low temperature outputs identical, ' \
            'temperature may not be effective'

    def test_top_p_parameter(self):
        print(f'\n[Model: {self.model_name}] Running top_p parameter test')
        prompt = 'The weather today is'

        resp_strict = self._post({'prompt': prompt, 'max_tokens': 20, 'top_p': 0.01, 'stream': False})
        resp_loose = self._post({'prompt': prompt, 'max_tokens': 20, 'top_p': 0.99, 'stream': False})

        text_strict = resp_strict.json()
        text_loose = resp_loose.json()

        self._validate_generation_response(data=text_strict, validate_tokens=True)
        self._validate_generation_response(data=text_loose, validate_tokens=True)

    def test_top_k_parameter(self):
        print(f'\n[Model: {self.model_name}] Running top_k parameter test')
        prompt = 'Artificial intelligence'

        resp_k10 = self._post({'prompt': prompt, 'max_tokens': 10, 'top_k': 10, 'stream': False})
        resp_k50 = self._post({'prompt': prompt, 'max_tokens': 10, 'top_k': 50, 'stream': False})

        text_k10 = resp_k10.json()
        text_k50 = resp_k50.json()

        self._validate_generation_response(data=text_k10, validate_tokens=True)
        self._validate_generation_response(data=text_k50, validate_tokens=True)

    def test_min_p_parameter(self):
        print(f'\n[Model: {self.model_name}] Running min_p parameter test')
        prompt = 'Machine learning is'

        resp = self._post({'prompt': prompt, 'max_tokens': 10, 'min_p': 0.05, 'stream': False})
        data = resp.json()
        self._validate_generation_response(data)

    def test_repetition_penalty(self):
        print(f'\n[Model: {self.model_name}] Running repetition penalty test')
        prompt = 'Repeat repeat repeat repeat'

        resp_no_penalty = self._post({'prompt': prompt, 'max_tokens': 10, 'repetition_penalty': 1.0, 'stream': False})
        resp_penalty = self._post({'prompt': prompt, 'max_tokens': 10, 'repetition_penalty': 1.5, 'stream': False})

        text_no_penalty = resp_no_penalty.json()['text']
        text_penalty = resp_penalty.json()['text']

        def count_repeats(text):
            words = text.lower().split()
            return sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])

        repeats_no_penalty = count_repeats(text_no_penalty)
        repeats_penalty = count_repeats(text_penalty)

        assert repeats_penalty <= repeats_no_penalty, (
            f'High penalty coefficient ({1.5}) repetition count ({repeats_penalty}) '
            f'not less than low penalty ({1.0}) count ({repeats_no_penalty}), '
            f'repetition_penalty ineffective')

    def test_ignore_eos_parameter(self):
        print(f'\n[Model: {self.model_name}] Running ignore_eos parameter test')
        prompt = 'The sky is blue.'

        resp_normal = self._post({'prompt': prompt, 'ignore_eos': False, 'stream': False})
        data_normal = resp_normal.json()
        self._validate_generation_response(data_normal)

        resp_ignore = self._post({'prompt': prompt, 'ignore_eos': True, 'stream': False})
        data_ignore = resp_ignore.json()
        self._validate_generation_response(data_ignore)

        reason_ignore = data_ignore.get('meta_info', {}).get('finish_reason', {}).get('type', 'unknown')

        assert reason_ignore == 'length', \
            f'ignore_eos=True must end due to length, actual: {reason_ignore}'

    def test_skip_special_tokens(self, config):
        print(f'[Model: {self.model_name}] Running skip_special_tokens test')
        model_path = os.path.join(config.get('model_path'), self.model_name)
        user_content = 'Hello [world]! This is a [test].'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        special_tokens_map = tokenizer.special_tokens_map

        special_patterns = list(special_tokens_map.values())
        special_patterns = [
            item for sublist in special_patterns for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        print('Special patterns:', special_patterns)

        print(' Executing skip_special_tokens=True')
        payload_true = {'prompt': user_content, 'max_tokens': 100, 'skip_special_tokens': True, 'stream': False}
        resp_true = self._post(payload_true)
        data_true = resp_true.json()
        self._validate_generation_response(data=data_true, validate_tokens=True)
        generated_text = data_true['text']
        assert not any(pattern in generated_text for pattern in special_patterns), \
            'Expected no special pattern in the generated text but found one.'

    def test_stop_token_ids(self):
        print(f'\n[Model: {self.model_name}] Running stop_token_ids test')
        payload = {'prompt': 'Once upon a time', 'max_tokens': 50, 'stop_token_ids': [11], 'stream': False}

        resp = self._post(payload)
        assert resp.status_code == 200, \
            f'HTTP request failed, status code: {resp.status_code}'

        try:
            data = resp.json()
        except Exception as e:
            pytest.fail(f'Response JSON parsing failed: {e}')

        self._validate_generation_response(data)

        generated_text = data.get('text', '')
        finish_reason = data.get('meta_info', {}).get('finish_reason', {}).get('type', 'unknown')
        actual_length = len(generated_text)

        assert finish_reason in ['stop', 'eos'], \
            f'Expected generation to end due to stop token, ' \
            f'actual reason: {finish_reason}. This may mean stop_token_ids [11] ' \
            f"didn't take effect, or generation was truncated."

        print(f'\n stop_token_ids=[11] generation result: length={actual_length}, '
              f"end reason='{finish_reason}', text='{generated_text[:20]}...'")

    def test_combined_parameters(self):
        print(f'\n[Model: {self.model_name}] Running combined parameters test')
        resp = self._post({
            'prompt': 'The future of AI',
            'max_tokens': 15,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'stream': False
        })

        assert resp.status_code == 200
        data = resp.json()
        self._validate_generation_response(data)

    def test_streaming_with_all_parameters(self):
        print(f'\n[Model: {self.model_name}] Running streaming with all parameters test')
        resp = self._post(
            {
                'prompt': 'Streaming test with parameters',
                'max_tokens': 10,
                'temperature': 0.8,
                'top_p': 0.85,
                'top_k': 30,
                'repetition_penalty': 1.2,
                'stop': ['test'],
                'stream': True
            },
            stream=True)

        assert resp.status_code == 200
        data = resp.json()
        self._validate_generation_response(data)

        stream_events = data['meta_info'].get('stream_events', [])

        assert stream_events == len(data['output_ids']) + 1, \
            'Streaming event count should not be less than generated token count'

    def test_invalid_temperature_values(self):
        print(f'\n[Model: {self.model_name}] Running invalid temperature values test')
        resp1 = self._post({'prompt': 'Test', 'max_tokens': 3, 'temperature': 0.0, 'stream': False})
        assert resp1.status_code == 200, 'temperature=0.0 should be valid'

        with pytest.raises(requests.HTTPError) as exc_info:
            self._post({'prompt': 'Test', 'max_tokens': 3, 'temperature': -0.5, 'stream': False})
        assert exc_info.value.response.status_code in [400, 422]

        print('  Invalid temperature values test passed')

    def test_invalid_top_p_values(self):
        print(f'\n[Model: {self.model_name}] Running invalid top_p values test')
        with pytest.raises(requests.HTTPError) as exc_info:
            self._post({'prompt': 'Test', 'max_tokens': 3, 'top_p': 1.5, 'stream': False})
        assert exc_info.value.response.status_code in [400, 422]

        print('  Invalid top_p values test passed')

    def test_invalid_top_k_values(self):
        print(f'\n[Model: {self.model_name}] Running invalid top_k values test')
        with pytest.raises(requests.HTTPError) as exc_info:
            self._post({'prompt': 'Test', 'max_tokens': 3, 'top_k': -5, 'stream': False})
        assert exc_info.value.response.status_code in [400, 422]

        print('  Invalid top_k values test passed')

    def test_boundary_max_tokens(self):
        print(f'\n[Model: {self.model_name}] Running boundary max_tokens test')
        resp1 = self._post({'prompt': 'Min tokens', 'max_tokens': 1, 'stream': False})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1['meta_info']['completion_tokens'] >= 1

        resp2 = self._post({'prompt': 'Max tokens test', 'max_tokens': 2048, 'stream': False})
        assert resp2.status_code == 200

        with pytest.raises(requests.HTTPError) as exc:
            self._post({'prompt': 'Test', 'max_tokens': -2, 'stream': False})

        assert exc.value.response.status_code == 400

        with pytest.raises(requests.HTTPError) as exc:
            self._post({'prompt': 'Test', 'max_tokens': 0, 'stream': False})

        assert exc.value.response.status_code == 400

        print('  Max tokens boundary test passed')

    def test_parameter_interactions(self):
        print(f'\n[Model: {self.model_name}] Running parameter interactions test')
        resp1 = self._post({
            'prompt': 'Deterministic generation',
            'max_tokens': 10,
            'temperature': 0.0,
            'top_p': 0.5,
            'top_k': 10,
            'stream': False
        })
        assert resp1.status_code == 200
        data1 = resp1.json()

        self._validate_generation_response(data1)

        print('  Parameter interaction (temp=0 with top_p/k) passed')

    def test_session_id_with_all_parameters(self):
        print(f'\n[Model: {self.model_name}] Running session_id with all parameters test')
        session_id = int(time.time()) % 100000

        resp1 = self._post({
            'session_id': session_id,
            'prompt': 'Hello, introduce yourself briefly.',
            'max_tokens': 20,
            'temperature': 0.7,
            'stream': False
        })
        assert resp1.status_code == 200
        data1 = resp1.json()
        self._validate_generation_response(data1)

        resp2 = self._post({
            'session_id': session_id,
            'prompt': 'What was I just talking about?',
            'max_tokens': 20,
            'temperature': 0.7,
            'stream': False
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        self._validate_generation_response(data2)

        assert 'What' in data2['text'] or 'hello' in data2['text'].lower() or \
            len(data2['text']) > 0

        print(f'  Session {session_id} test passed')

    def test_edge_cases_stop_conditions(self):
        print(f'\n[Model: {self.model_name}] Running edge cases stop conditions test')
        resp1 = self._post({'prompt': 'Test with empty stop list', 'max_tokens': 10, 'stop': [], 'stream': False})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1['text']) > 0

        resp2 = self._post({
            'prompt': 'Write a sentence ending with a period. Stop here test.',
            'max_tokens': 50,
            'stop': ['.'],
            'stream': False
        })
        assert resp2.status_code == 200
        data2 = resp2.json()

        text2 = data2['text']
        finish_reason = data2['meta_info']['finish_reason']['type']

        if '.' in text2:
            assert text2.strip().endswith('.'), \
                "Stop token '.' should cause generation to end at period"

        assert finish_reason in ['stop', 'eos'], \
            f'Expected to end due to stop token, actual: {finish_reason}'

        print(f"  Stop at '.': generated '{text2}' (Reason: {finish_reason})")

    def test_spaces_between_special_tokens(self, config):
        print(f'[Model: {self.model_name}] Running spaces_between_special_tokens test')
        model_path = os.path.join(config.get('model_path'), self.model_name)
        user_content = 'Hello [world]! This is a [test].'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        special_tokens_map = tokenizer.special_tokens_map

        special_patterns = list(special_tokens_map.values())
        special_patterns = [
            item for sublist in special_patterns for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        print(' Executing skip_special_tokens=False and checking spaces between special tokens')
        payload_false = {'prompt': user_content, 'max_tokens': 100, 'skip_special_tokens': False, 'stream': False}
        resp_false = self._post(payload_false)
        data_false = resp_false.json()
        self._validate_generation_response(data=data_false, validate_tokens=True)
        generated_text = data_false['text']

        for i in range(len(generated_text) - 1):
            if generated_text[i] in special_patterns and generated_text[i + 1] not in [' ', '\n']:
                assert False, f'Expected space after special token {generated_text[i]} but found none.'

    @pytest.mark.experts
    @pytest.mark.pytorch
    def test_request_returns_experts(self):
        print(f'\n[Model: {self.model_name}] Running request with experts test')
        resp1 = self._post({
            'prompt': 'Deterministic generation',
            'max_tokens': 50,
            'temperature': 0.8,
            'return_routed_experts': True
        })
        assert resp1.status_code == 200
        data1 = resp1.json()

        self._validate_generation_response(data1, validate_experts=True)
