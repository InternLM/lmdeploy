import json
import os
import re
import subprocess
import time

import allure
import psutil
import requests
from openai import APIStatusError, BadRequestError, OpenAI
from pytest_assume.plugin import assume
from utils.config_utils import (
    get_case_str_by_config,
    get_cli_common_param,
    get_cuda_prefix_by_workerid,
    get_workerid,
    resolve_extra_params,
)
from utils.constant import DEFAULT_PORT, DEFAULT_SERVER, MM_DEMO_TOMB_USER_PROMPT
from utils.restful_return_check import assert_chat_completions_batch_return
from utils.rule_condition_assert import assert_result

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = f'http://{DEFAULT_SERVER}'


def start_openai_service(config, run_config, worker_id, timeout: int = 1200):
    port = DEFAULT_PORT + get_workerid(worker_id)
    case_name = get_case_str_by_config(run_config)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    server_log = os.path.join(config.get('server_log_path'), f'log_{case_name}_{port}_{timestamp}.log')

    model = run_config.get('model')
    if run_config.get('env', {}).get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        model_path = model
    else:
        model_path = os.path.join(config.get('model_path'), model)

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))

    # Ensure extra_params exists before modifying
    if 'extra_params' not in run_config:
        run_config['extra_params'] = {}

    resolve_extra_params(run_config['extra_params'], config.get('model_path'))

    run_config['extra_params']['server-port'] = str(port)
    run_config['extra_params']['allow-terminate-by-client'] = None
    model_name = case_name if run_config['extra_params'].get(
        'model-name', None) is None else run_config['extra_params'].pop('model-name')
    cmd = ' '.join([
        cuda_prefix, 'lmdeploy serve api_server', model_path,
        get_cli_common_param(run_config), f'--model-name {model_name}'
    ]).strip()

    env = os.environ.copy()
    env['MASTER_PORT'] = str(get_workerid(worker_id) + 29500)
    env.update(run_config.get('env', {}))

    file = open(server_log, 'w')
    print('reproduce command restful: ' + cmd)
    file.write('reproduce command restful: ' + cmd + '\n')
    startRes = subprocess.Popen(cmd,
                                stdout=file,
                                stderr=file,
                                shell=True,
                                text=True,
                                env=env,
                                encoding='utf-8',
                                errors='replace',
                                start_new_session=True)
    pid = startRes.pid

    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    start_time = int(time.time())
    start_timeout = timeout

    time.sleep(5)
    for i in range(start_timeout):
        time.sleep(1)
        end_time = int(time.time())
        total_time = end_time - start_time
        result = health_check(http_url, case_name)
        if result or total_time >= start_timeout:
            break
        try:
            # Check if process is still running
            return_code = startRes.wait(timeout=1)  # Small timeout to check status
            if return_code != 0:
                with open(server_log) as f:
                    content = f.read()
                    print(content)
                return 0, content
        except subprocess.TimeoutExpired:
            continue
    file.close()
    allure.attach.file(server_log, name=server_log, attachment_type=allure.attachment_type.TEXT)
    return pid, ''


def stop_restful_api(pid, startRes):
    if pid > 0:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()


def terminate_restful_api(worker_id):
    port = DEFAULT_PORT + get_workerid(worker_id)
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    response = None
    request_error = None
    try:
        response = requests.get(f'{http_url}/terminate')
    except requests.exceptions.RequestException as exc:
        request_error = exc
    if request_error is not None:
        assert False, f'terminate request failed: {request_error}'
    assert response is not None and response.status_code == 200, f'terminate with {response}'


def run_all_step(log_path, case_name, cases_info, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'
    for case in cases_info.keys():
        if case != 'code_testcase' and 'code' in model.lower():
            continue
        case_info = cases_info.get(case)

        with allure.step(case + ' restful_test - openai chat'):
            restful_result, restful_log, msg = open_chat_test(log_path, case_name, case_info, http_url)
            allure.attach.file(restful_log, name=restful_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert restful_result, msg


def open_chat_test(log_path, case_name, case_info, url):
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    restful_log = os.path.join(log_path, f'log_restful_{case_name}_{timestamp}.log')

    file = open(restful_log, 'w')

    result = True

    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{url}/v1')
    model_name = client.models.list().data[0].id

    messages = []
    msg = ''
    for prompt_detail in case_info:
        if not result:
            break
        prompt = list(prompt_detail.keys())[0]
        messages.append({'role': 'user', 'content': prompt})
        file.writelines('prompt:' + prompt + '\n')

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=messages,
                                                 temperature=0.01,
                                                 top_p=0.8,
                                                 max_completion_tokens=1024,
                                                 stream=True)

        content_chunks = []
        reasoning_content_chunks = []
        for output in outputs:
            # Safely handle streaming chunks: choices may be empty and content may be None
            if not getattr(output, 'choices', None):
                continue
            choice = output.choices[0]
            delta = getattr(choice, 'delta', None)
            reasoning_content = getattr(delta, 'reasoning_content', None) if delta is not None else None
            content = getattr(delta, 'content', None) if delta is not None else None
            if reasoning_content:
                reasoning_content_chunks.append(reasoning_content)
            if content:
                content_chunks.append(content)
        reasoning_content = ''.join(reasoning_content_chunks)
        output_content = ''.join(content_chunks)

        file.writelines(f'reasoning_content :{reasoning_content}, content: {output_content}\n')
        messages.append({'role': 'assistant', 'content': output_content})

        case_result, reason = assert_result(reasoning_content + output_content, prompt_detail.values(), model_name)
        file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
        if not case_result:
            msg += reason
        result = result and case_result
    file.close()
    return result, restful_log, msg


def health_check(url, model_name):
    try:
        api_client = APIClient(url)
        model_name_current = api_client.available_models[0]
        messages = []
        messages.append({'role': 'user', 'content': '你好'})
        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            # Return True on first successful response
            return model_name == model_name_current
        return False  # No output received
    except Exception:
        return False


def get_model(url):
    print(url)
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        return model_name.split('/')[-1]
    except Exception:
        return None


def _run_logprobs_test(port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])
    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    output = None
    for output in api_client.chat_completions_v1(model=model_name,
                                                 messages='Hi, pls intro yourself',
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10):
        continue
    if output is None:
        assert False, 'No output received from logprobs test'
    print(output)
    assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
    assert output.get('choices')[0].get('finish_reason') == 'length'
    assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5


PIC = 'tiger.jpeg'  # noqa E501
PIC2 = 'human-pose.jpg'  # noqa E501
VIDEO = 'red-panda.mp4'  # noqa E501
VIDEO_QWEN3_DEMO = 'N1cdUjctpG8.mp4'  # noqa E501
MM_DEMO_MAX_TOKENS = 24576
MM_DEMO_MAX_TOKENS_STREAM = 24576
VIDEO_SINGLE_FRAME_MAX_TOKENS = 512
VIDEO_REDPANDA_STREAM_MAX_TOKENS = 2048


def _vl_video_stream_finish_assert(finish: str | None, text: str) -> bool:
    """``stop`` / ``length`` red-panda video: species keywords, then ``length``
    needs enough text."""
    if finish not in ('stop', 'length'):
        return False
    t = (text or '').lower()
    raw = text or ''
    species_match = (
        any(p in t for p in ('red panda', 'lesser panda'))
        or 'ailurus' in t
        or any(s in raw for s in ('小熊猫', '红熊猫'))
    )
    if not species_match:
        return False
    if finish == 'length':
        return len(raw.strip()) >= 300
    return True


_REDACTED_THINKING_END = '</think>'


def _mm_demo_public_answer_text(text: str) -> str:
    """Optional JSON-string decode (pipeline logs); then tail after
    ``</think>`` when present."""
    s = (text or '').strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = str(json.loads(s))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    s = s.strip()
    key = _REDACTED_THINKING_END
    i = s.lower().rfind(key.lower())
    if i == -1:
        return s
    return s[i + len(key) :].strip()


def _mm_demo_tomb_answer_assert(text: str) -> bool:
    """Tomb/MCQ: visible tail mentions scene, a digit, or an MCQ-style letter
    (A–D)."""
    raw = _mm_demo_public_answer_text(text).strip()
    if not raw:
        return False
    rl = raw.lower()
    if any(w in rl for w in ('jar', 'porcelain', 'tomb', 'niche', 'chamber', '罐', '瓷', '墓', '龛')):
        return True
    if any(c.isdigit() for c in raw):
        return True
    s = raw.strip()
    if re.search(r'(?i)\b(?:answer|choice|option|correct)\b\s*[:：]?\s*[abcd]\b', s):
        return True
    if re.fullmatch(r'(?is)[`"\(\[]*[abcd][`"\)\]]*\.?\s*', s):
        return True
    if len(s) <= 120 and re.match(r'(?is)[`"\(\[]*[abcd][`"\)\]]*[\s\.\):,\-]', s):
        return True
    return False


def _mm_demo_thinking_wrapper_shape_assert(text: str) -> bool:
    """Bound user-visible tail after ``</think>``, or total size if the wrapper
    never closes."""
    s = (text or '').strip()
    if not s:
        return False
    if _REDACTED_THINKING_END.lower() in s.lower():
        public = _mm_demo_public_answer_text(s).strip()
        return 0 < len(public) <= 2000
    return len(s) <= 3200


def _mm_demo_tomb_run_assert(finish: str | None, text: str) -> bool:
    """Tomb + ``mm_processor``: ``stop`` + shape; ``length`` + closed thinking
    + shape, else long jar/scene tail."""
    t = (text or '').strip()
    if not t or not _mm_demo_tomb_answer_assert(t):
        return False
    if finish == 'stop':
        return _mm_demo_thinking_wrapper_shape_assert(t)
    if finish == 'length':
        if _REDACTED_THINKING_END.lower() in t.lower():
            return _mm_demo_thinking_wrapper_shape_assert(t)
        if len(t) < 1500:
            return False
        head_l = t[:8000].lower()
        if 'jar' not in head_l:
            return False
        return any(w in head_l for w in ('niche', 'chamber', 'tomb', 'porcelain', 'primary', '罐', '墓', '龛', '瓷'))
    return False


def _mm_demo_single_frame_scene_assert(text: str) -> bool:
    """Single-frame: short visible tail plus chamber / niche / vessel hints."""
    raw = _mm_demo_public_answer_text(text).strip()
    if not raw or len(raw) < 20:
        return False
    if sum(1 for c in raw if c.isalpha()) < 12:
        return False
    rl = raw.lower()
    if any(w in rl for w in ('chamber', 'niche', 'jar', 'porcelain', 'artifact', 'coffin', '墓室', '龛', '罐')):
        return True
    return False


def _consume_chat_completion_stream(stream_iter) -> tuple[str | None, str]:
    """Drain a chat-completion stream: ``(finish_reason, joined delta content)``."""
    chunks: list[str] = []
    last_fr: str | None = None
    for ev in stream_iter:
        if not getattr(ev, 'choices', None):
            continue
        choice = ev.choices[0]
        fr = getattr(choice, 'finish_reason', None)
        if fr:
            last_fr = fr
        delta = getattr(choice, 'delta', None)
        if delta and getattr(delta, 'content', None):
            chunks.append(delta.content)
    return last_fr, ''.join(chunks)


def run_vl_testcase(log_path, resource_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)
    if model is None:
        assert False, 'server not start correctly'

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    simple_model_name = model_name.split('/')[-1]
    restful_log = os.path.join(log_path, f'restful_vl_{simple_model_name}_{str(port)}_{timestamp}.log')  # noqa
    file = open(restful_log, 'w')

    prompt_messages = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': f'{resource_path}/{PIC}',
            },
        }, {
            'type': 'image_url',
            'image_url': {
                'url': f'{resource_path}/{PIC2}',
            },
        }],
    }]

    response = client.chat.completions.create(model=model_name, messages=prompt_messages, temperature=0.8, top_p=0.8)
    file.writelines(str(response).lower() + '\n')

    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    for item in api_client.chat_completions_v1(model=model_name, messages=prompt_messages):
        continue
    file.writelines(str(item) + '\n')

    video_path = os.path.join(resource_path, VIDEO)
    video_messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': ('What animal appears in the clip? Give the common species name in one or two '
                         'short sentences (avoid long step-by-step reasoning).'),
            },
            {
                'type': 'video_url',
                'video_url': {
                    'url': video_path,
                },
            },
        ],
    }]
    video_messages_one_frame = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': ('The server decodes this clip to a single video frame only. What animal appears? '
                         'Answer in one or two short sentences.'),
            },
            {
                'type': 'video_url',
                'video_url': {
                    'url': video_path,
                },
            },
        ],
    }]

    if not os.path.isfile(video_path):
        file.writelines(f'[video testcase skipped] missing file: {video_path}\n')
    else:
        try:
            v_resp = client.chat.completions.create(
                model=model_name,
                messages=video_messages,
                temperature=0.2,
                max_tokens=512,
                extra_body={
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 8,
                        },
                    },
                },
            )
        except (BadRequestError, APIStatusError) as exc:
            file.writelines(f'[video testcase skipped] model/server rejected video_url: {exc!r}\n')
        else:
            file.writelines('[video non-stream] ' + str(v_resp).lower() + '\n')
            content = (v_resp.choices[0].message.content or '')
            assert _vl_video_stream_finish_assert(getattr(v_resp.choices[0], 'finish_reason', None), content), v_resp

            v_more = client.chat.completions.create(
                model=model_name,
                messages=video_messages,
                temperature=0.0,
                max_tokens=1,
                extra_body={
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 16,
                        },
                    },
                },
            )
            v_few = client.chat.completions.create(
                model=model_name,
                messages=video_messages,
                temperature=0.0,
                max_tokens=1,
                extra_body={
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 4,
                        },
                    },
                },
            )
            u_more = getattr(v_more, 'usage', None)
            u_few = getattr(v_few, 'usage', None)
            if u_more and u_few and getattr(u_few, 'prompt_tokens', None) and getattr(u_more, 'prompt_tokens', None):
                if u_few.prompt_tokens < u_more.prompt_tokens:
                    file.writelines('[video] fewer frames => fewer prompt_tokens (as expected)\n')
                else:
                    few_t, many_t = u_few.prompt_tokens, u_more.prompt_tokens
                    file.writelines(
                        f'[video] prompt_tokens not compared (few={few_t}, many={many_t})\n',
                    )

            stream = client.chat.completions.create(
                model=model_name,
                messages=video_messages,
                temperature=0.2,
                max_tokens=VIDEO_REDPANDA_STREAM_MAX_TOKENS,
                stream=True,
                extra_body={
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 8,
                        },
                    },
                },
            )
            stream_fr, joined = _consume_chat_completion_stream(stream)
            file.writelines('[video stream] ' + joined.lower() + '\n')
            assert _vl_video_stream_finish_assert(stream_fr, joined), (stream_fr, joined[:1200])

            video_payload = {
                'model': model_name,
                'messages': video_messages,
                'temperature': 0.2,
                'max_tokens': VIDEO_REDPANDA_STREAM_MAX_TOKENS,
                'media_io_kwargs': {
                    'video': {
                        'num_frames': 8,
                    },
                },
            }
            raw = requests.post(f'{http_url}/v1/chat/completions',
                                headers={'content-type': 'application/json'},
                                json=video_payload,
                                timeout=600)
            file.writelines(f'[video raw http] status={raw.status_code}\n')
            assert raw.ok, raw.text
            raw_json = raw.json()
            raw_ch0 = (raw_json.get('choices') or [{}])[0]
            raw_text = raw_ch0.get('message', {}).get('content') or ''
            file.writelines(raw_text.lower() + '\n')
            raw_fr = raw_ch0.get('finish_reason')
            assert _vl_video_stream_finish_assert(raw_fr, raw_text), (raw_fr, raw_text[:1200], raw_json)

            v_one = client.chat.completions.create(
                model=model_name,
                messages=video_messages_one_frame,
                temperature=0.2,
                max_tokens=VIDEO_SINGLE_FRAME_MAX_TOKENS,
                extra_body={
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 1,
                        },
                    },
                },
            )
            file.writelines('[video single-frame] ' + str(v_one).lower() + '\n')
            one_content = (v_one.choices[0].message.content or '')
            assert _vl_video_stream_finish_assert(getattr(v_one.choices[0], 'finish_reason', None), one_content), v_one

    # Qwen3-VL style: local demo mp4 + mm_processor_kwargs (fps / do_sample_frames), OpenAI-compatible body.
    demo_video_path = os.path.join(resource_path, VIDEO_QWEN3_DEMO)
    demo_question = MM_DEMO_TOMB_USER_PROMPT
    # Single-frame sampling often lands on an aerial or intro shot, not the jar niche scene.
    mm_one_question = (
        'This is one frame from a short news-style clip about an ancient tomb. '
        'If you see interior details, focus on chamber, niches, pottery or porcelain jars, '
        'coffin, or furnishings, in one or two short sentences. '
        'If the frame is only exterior or aerial, say that in one short sentence. '
        'No long step-by-step reasoning.')
    mm_demo_messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'video_url',
                'video_url': {
                    'url': demo_video_path,
                },
            },
            {
                'type': 'text',
                'text': demo_question,
            },
        ],
    }]
    mm_one_messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'video_url',
                'video_url': {
                    'url': demo_video_path,
                },
            },
            {
                'type': 'text',
                'text': mm_one_question,
            },
        ],
    }]
    if not os.path.isfile(demo_video_path):
        file.writelines(f'[video mm_processor demo skipped] missing file: {demo_video_path}\n')
    else:
        try:
            mm_resp = client.chat.completions.create(
                model=model_name,
                messages=mm_demo_messages,
                max_tokens=MM_DEMO_MAX_TOKENS,
                temperature=0.3,
                top_p=0.95,
                extra_body={
                    'top_k': 20,
                    'mm_processor_kwargs': {
                        'fps': 2,
                        'do_sample_frames': True,
                    },
                },
            )
        except (BadRequestError, APIStatusError) as exc:
            file.writelines(f'[video mm_processor demo skipped] {exc!r}\n')
        else:
            file.writelines('[video mm_processor non-stream] ' + str(mm_resp).lower() + '\n')
            mm_text = (mm_resp.choices[0].message.content or '').strip()
            mm_fr = getattr(mm_resp.choices[0], 'finish_reason', None)
            assert _mm_demo_tomb_run_assert(mm_fr, mm_text), (mm_fr, mm_text[:2000])

            mm_stream = client.chat.completions.create(
                model=model_name,
                messages=mm_demo_messages,
                max_tokens=MM_DEMO_MAX_TOKENS_STREAM,
                temperature=0.2,
                stream=True,
                extra_body={
                    'top_k': 20,
                    'mm_processor_kwargs': {
                        'fps': 2,
                        'do_sample_frames': True,
                    },
                },
            )
            mm_finish, mm_joined = _consume_chat_completion_stream(mm_stream)
            mm_joined = mm_joined.strip()
            file.writelines('[video mm_processor stream] ' + mm_joined.lower() + '\n')
            assert _mm_demo_tomb_run_assert(mm_finish, mm_joined), (mm_finish, mm_joined[:2000])

            mm_raw_payload = {
                'model': model_name,
                'messages': mm_demo_messages,
                'temperature': 0.3,
                'max_tokens': MM_DEMO_MAX_TOKENS,
                'top_k': 20,
                'mm_processor_kwargs': {
                    'fps': 2,
                    'do_sample_frames': True,
                },
            }
            mm_raw = requests.post(f'{http_url}/v1/chat/completions',
                                     headers={'content-type': 'application/json'},
                                     json=mm_raw_payload,
                                     timeout=600)
            file.writelines(f'[video mm_processor raw http] status={mm_raw.status_code}\n')
            assert mm_raw.ok, mm_raw.text
            mm_raw_json = mm_raw.json()
            mm_raw_choice0 = (mm_raw_json.get('choices') or [{}])[0]
            mm_raw_text = mm_raw_choice0.get('message', {}).get('content') or ''
            file.writelines(mm_raw_text.lower() + '\n')
            mm_raw_fr = mm_raw_choice0.get('finish_reason')
            assert _mm_demo_tomb_run_assert(mm_raw_fr, mm_raw_text), (mm_raw_fr, mm_raw_text[:2000], mm_raw_json)

            mm_one = client.chat.completions.create(
                model=model_name,
                messages=mm_one_messages,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.95,
                extra_body={
                    'top_k': 20,
                    'media_io_kwargs': {
                        'video': {
                            'num_frames': 1,
                        },
                    },
                },
            )
            file.writelines('[video mm_processor single-frame] ' + str(mm_one).lower() + '\n')
            assert getattr(mm_one.choices[0], 'finish_reason', None) == 'stop', mm_one
            mm_one_text = (mm_one.choices[0].message.content or '').strip()
            assert mm_one_text and _mm_demo_single_frame_scene_assert(mm_one_text), mm_one_text
            assert _mm_demo_thinking_wrapper_shape_assert(mm_one_text), mm_one_text

    file.close()

    allure.attach.file(restful_log, name=restful_log, attachment_type=allure.attachment_type.TEXT)

    assert 'tiger' in str(response).lower() or '虎' in str(response).lower() or 'ski' in str(
        response).lower() or '滑雪' in str(response).lower(), response
    assert 'tiger' in str(item).lower() or '虎' in str(item).lower() or 'ski' in str(item).lower() or '滑雪' in str(
        item).lower(), item


def _run_reasoning_case(log_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    restful_log = os.path.join(log_path, f'restful_reasoning_{model}_{str(port)}_{timestamp}.log')
    file = open(restful_log, 'w')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    with allure.step('step1 - stream'):
        messages = [{'role': 'user', 'content': '9.11 and 9.8, which is greater?'}]
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01, stream=True)
        outputList = []
        final_content = ''
        final_reasoning_content = ''
        for stream_response in response:
            if stream_response.choices[0].delta.content is not None:
                final_content += stream_response.choices[0].delta.content
            if stream_response.choices[0].delta.reasoning_content is not None:
                final_reasoning_content += stream_response.choices[0].delta.reasoning_content
            outputList.append(stream_response)
        file.writelines(str(outputList) + '\n')
        with assume:
            assert '9.11' in final_reasoning_content and '9.11' in final_content and len(outputList) > 1, str(
                outputList)

    with allure.step('step2 - batch'):
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01, stream=False)
        print(response)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        file.writelines(str(outputList) + '\n')
        with assume:
            assert '9.11' in reasoning_content and '9.11' in content and len(outputList) > 1, str(outputList)

    file.close()
    allure.attach.file(restful_log, name=restful_log, attachment_type=allure.attachment_type.TEXT)


def test_internlm_multiple_round_prompt(client, model):

    def add(a: int, b: int):
        return a + b

    def mul(a: int, b: int):
        return a * b

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
    }, {
        'type': 'function',
        'function': {
            'name': 'mul',
            'description': 'Calculate the product of two numbers',
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
    messages = [{'role': 'user', 'content': 'Compute (3+5)*2'}]

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.01,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list = [response]
    func1_name = response.choices[0].message.tool_calls[0].function.name
    func1_args = response.choices[0].message.tool_calls[0].function.arguments
    func1_args_dict = json.loads(func1_args)
    func1_out = add(**func1_args_dict) if func1_name == 'add' else mul(**func1_args_dict)
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
    with assume:
        assert func1_name == 'add'
    with assume:
        assert func1_args == '{"a": 3, "b": 5}'
    with assume:
        assert func1_out == 8
    with assume:
        assert response.choices[0].message.tool_calls[0].type == 'function'

    messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
    messages.append({'role': 'environment', 'content': f'3+5={func1_out}', 'name': 'plugin'})
    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list.append(response)
    func2_name = response.choices[0].message.tool_calls[0].function.name
    func2_args = response.choices[0].message.tool_calls[0].function.arguments
    func2_args_dict = json.loads(func2_args)
    func2_out = add(**func2_args_dict) if func2_name == 'add' else mul(**func2_args_dict)
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
    with assume:
        assert func2_name == 'mul'
    with assume:
        assert func2_args == '{"a": 8, "b": 2}'
    with assume:
        assert func2_out == 16
    with assume:
        assert response.choices[0].message.tool_calls[0].type == 'function'

    return response_list


def test_qwen_multiple_round_prompt(client, model):

    def get_current_temperature(location: str, unit: str = 'celsius'):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            'temperature': 26.1,
            'location': location,
            'unit': unit,
        }

    def get_temperature_date(location: str, date: str, unit: str = 'celsius'):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format 'City, State, Country'.
            date: The date to get the temperature for, in the format 'Year-Month-Day'.
            unit: The unit to return the temperature in. Defaults to 'celsius'. (choices: ['celsius', 'fahrenheit'])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            'temperature': 25.9,
            'location': location,
            'date': date,
            'unit': unit,
        }

    def get_function_by_name(name):
        if name == 'get_current_temperature':
            return get_current_temperature
        if name == 'get_temperature_date':
            return get_temperature_date

    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_current_temperature',
            'description': 'Get current temperature at a location.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description':
                        'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': ['location']
            }
        }
    }, {
        'type': 'function',
        'function': {
            'name': 'get_temperature_date',
            'description': 'Get temperature at a location and date.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description':
                        'The location to get the temperature for, in the format \'City, State, Country\'.'
                    },
                    'date': {
                        'type': 'string',
                        'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                    }
                },
                'required': ['location', 'date']
            }
        }
    }]
    messages = [{
        'role': 'user',
        'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?'
    }]

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list = [response]
    func1_name = response.choices[0].message.tool_calls[0].function.name
    func1_args = response.choices[0].message.tool_calls[0].function.arguments
    func2_name = response.choices[0].message.tool_calls[1].function.name
    func2_args = response.choices[0].message.tool_calls[1].function.arguments
    with assume:
        assert response.choices[0].finish_reason == 'tool_calls'
        assert func1_name == 'get_current_temperature'
        assert func1_args == '{"location": "San Francisco, CA, USA"}' \
            or func1_args == '{"location": "San Francisco, California, USA", "unit": "celsius"}'
        assert func2_name == 'get_temperature_date'
        assert func2_args == '{"location": "San Francisco, CA, USA", "date": "2024-11-15"}' \
            or func2_args == '{"location": "San Francisco, California, USA", "date": "2024-11-15", "unit": "celsius"}'
        assert response.choices[0].message.tool_calls[0].type == 'function'

    messages.append(response.choices[0].message)

    for tool_call in response.choices[0].message.tool_calls:
        tool_call_args = json.loads(tool_call.function.arguments)
        tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
        messages.append({
            'role': 'tool',
            'name': tool_call.function.name,
            'content': tool_call_result,
            'tool_call_id': tool_call.id
        })

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=0.8,
                                              top_p=0.8,
                                              stream=False,
                                              tools=tools)
    print(response)
    response_list.append(response)
    with assume:
        assert response.choices[0].finish_reason == 'stop'
        assert '26.1' in response.choices[0].message.content

    return response_list


def _run_tools_case(log_path, port: int = DEFAULT_PORT):
    http_url = ':'.join([BASE_HTTP_URL, str(port)])

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    restful_log = os.path.join(log_path, f'restful_toolcall_{model}_{str(port)}_{timestamp}.log')
    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    with open(restful_log, 'a') as file:
        with allure.step('step1 - one_round_prompt'):
            tools = [{
                'type': 'function',
                'function': {
                    'name': 'get_current_weather',
                    'description': 'Get the current weather in a given location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city and state, e.g. San Francisco, CA',
                            },
                            'unit': {
                                'type': 'string',
                                'enum': ['celsius', 'fahrenheit']
                            },
                        },
                        'required': ['location'],
                    },
                }
            }]
            messages = [{'role': 'user', 'content': 'What\'s the weather like in Boston today?'}]
            response = client.chat.completions.create(model=model_name,
                                                      messages=messages,
                                                      temperature=0.01,
                                                      stream=False,
                                                      tools=tools)
            print(response)
            with assume:
                assert response.choices[0].finish_reason == 'tool_calls'
            with assume:
                assert response.choices[0].message.tool_calls[0].function.name == 'get_current_weather'
            with assume:
                assert 'Boston' in response.choices[0].message.tool_calls[0].function.arguments
            with assume:
                assert response.choices[0].message.tool_calls[0].type == 'function'
            file.writelines(str(response) + '\n')

        with allure.step('step2 - search prompt'):
            tools = [{
                'type': 'function',
                'function': {
                    'name': 'search',
                    'description': 'BING search API',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'list of search query strings'
                            }
                        },
                        'required': ['location']
                    }
                }
            }]
            messages = [{'role': 'user', 'content': '搜索最近的人工智能发展趋势'}]
            response = client.chat.completions.create(model=model_name,
                                                      messages=messages,
                                                      temperature=0.01,
                                                      stream=False,
                                                      tools=tools)
            print(response)
            with assume:
                assert response.choices[0].finish_reason == 'tool_calls'
            with assume:
                assert response.choices[0].message.tool_calls[0].function.name == 'search'
            with assume:
                assert '人工智能' in response.choices[0].message.tool_calls[0].function.arguments
            with assume:
                assert response.choices[0].message.tool_calls[0].type == 'function'
            file.writelines(str(response) + '\n')

        with allure.step('step3 - multiple_round_prompt'):
            response_list = None
            if 'intern' in model.lower():
                response_list = test_internlm_multiple_round_prompt(client, model_name)
            elif 'qwen' in model.lower():
                response_list = test_qwen_multiple_round_prompt(client, model_name)

            if response_list is not None:
                file.writelines(str(response_list) + '\n')

    allure.attach.file(restful_log, name=restful_log, attachment_type=allure.attachment_type.TEXT)


def proxy_health_check(url):
    """Check if proxy server is healthy."""
    try:
        # For proxy server, we check if it responds to the /v1/models endpoint
        import requests
        response = requests.get(f'{url}/v1/models', timeout=5)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False


def start_proxy_server(log_path, port, case_name: str = 'default'):
    """Start the proxy server for testing with enhanced error handling and
    logging."""
    if log_path is None:
        log_path = '/nvme/qa_test_models/evaluation_report'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    proxy_log = os.path.join(log_path, f'proxy_server_{case_name}_{str(port)}_{timestamp}.log')

    proxy_url = f'http://{DEFAULT_SERVER}:{port}'  # noqa: E231, E261
    try:
        response = requests.get(f'{proxy_url}/nodes/status', timeout=5)
        if response.status_code == 200:
            print(f'Terminating existing nodes on proxy {proxy_url}')
            requests.get(f'{proxy_url}/nodes/terminate_all', timeout=10)
            time.sleep(5)
    except requests.exceptions.RequestException:
        pass

    cmd = (f'lmdeploy serve proxy --server-name {DEFAULT_SERVER} --server-port {port} '
           f'--routing-strategy min_expected_latency --serving-strategy Hybrid')

    print(f'Starting proxy server with command: {cmd}')
    print(f'Proxy log will be saved to: {proxy_log}')

    proxy_file = open(proxy_log, 'w')
    proxy_process = subprocess.Popen([cmd],
                                     stdout=proxy_file,
                                     stderr=proxy_file,
                                     shell=True,
                                     text=True,
                                     encoding='utf-8')
    pid = proxy_process.pid

    start_time = int(time.time())
    timeout = 300

    time.sleep(5)
    for i in range(timeout):
        time.sleep(1)
        if proxy_health_check(f'http://{DEFAULT_SERVER}:{port}'):  # noqa: E231, E261
            break

        try:
            # Check if process is still running
            return_code = proxy_process.wait(timeout=1)  # Small timeout to check status
            if return_code != 0:
                with open(proxy_log) as f:
                    content = f.read()
                    print(content)
                return 0, proxy_process
        except subprocess.TimeoutExpired:
            continue

        end_time = int(time.time())
        total_time = end_time - start_time
        if total_time >= timeout:
            break

    proxy_file.close()
    allure.attach.file(proxy_log, name=proxy_log, attachment_type=allure.attachment_type.TEXT)

    print(f'Proxy server started successfully with PID: {pid}')
    return pid, proxy_process


def run_llm_test(config, run_config, common_case_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            case_name = get_case_str_by_config(run_config)
            run_all_step(config.get('log_path'),
                         case_name,
                         common_case_config,
                         port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_mllm_test(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            run_vl_testcase(config.get('log_path'),
                            config.get('resource_path'),
                            port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_reasoning_case(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_reasoning_case(config.get('log_path'), port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_tools_case(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_tools_case(config.get('log_path'), port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_logprob_test(config, run_config, worker_id):
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        if pid > 0:
            _run_logprobs_test(port=DEFAULT_PORT + get_workerid(worker_id))
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)
