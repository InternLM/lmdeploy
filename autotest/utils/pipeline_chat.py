import os
import subprocess
from subprocess import PIPE

import allure
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from pytest_assume.plugin import assume
from utils.get_run_config import get_model_name, get_tp_num
from utils.rule_condition_assert import assert_result

from lmdeploy import pipeline
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.utils import is_bf16_supported
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64

gen_config = GenerationConfig(max_new_tokens=500)


def run_pipeline_chat_test(config,
                           cases_info,
                           model_case,
                           type,
                           worker_id: str = '',
                           extra: object = None,
                           use_local_model: bool = True):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_name = model_name = get_model_name(model_case)
    model_path = config.get('model_path')
    if use_local_model is True:
        hf_path = model_path + '/' + model_case
    else:
        hf_path = model_case

    if 'pytorch' in type:
        backend_config = PytorchEngineConfig(tp=tp)
        if not is_bf16_supported():
            backend_config.dtype = 'float16'
    else:
        backend_config = TurbomindEngineConfig(tp=tp)

    if 'lora' in type:
        backend_config.adapters = extra.get('adapters')
    if 'kvint' in type:
        backend_config.quant_policy = extra.get('quant_policy')

    # if llava support kvint or awq, this code should refactor
    if 'llava' in model_case:
        backend_config.model_name = 'vicuna'
    if 'w4' in model_case or ('4bits' in model_case or 'awq' in model_case.lower()):
        backend_config.model_format = 'awq'
    if 'gptq' in model_case.lower():
        backend_config.model_format = 'gptq'

    pipe = pipeline(hf_path, backend_config=backend_config)

    config_log = os.path.join(log_path,
                              '_'.join(['pipeline', 'config', type, worker_id,
                                        model_case.split('/')[1] + '.log']))
    file = open(config_log, 'w')
    log_string = '\n'.join([
        'reproduce config info:', 'from lmdeploy import pipeline', 'from lmdeploy.messages import PytorchEngineConfig',
        'from lmdeploy.messages import TurbomindEngineConfig', 'engine_config = ' + str(backend_config),
        'pipe = pipeline("' + hf_path + '",  backend_config=engine_config)', 'res = pipe("Hi, pls introduce shanghai")'
    ])
    file.writelines(log_string)
    print(log_string)
    file.close

    for case in cases_info.keys():
        if ('coder' in model_case or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        case_info = cases_info.get(case)
        pipeline_chat_log = os.path.join(
            log_path, '_'.join(['pipeline', 'chat', type, worker_id,
                                model_case.split('/')[1], case + '.log']))

        file = open(pipeline_chat_log, 'w')

        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            file.writelines('prompt:' + prompt + '\n')

            response = pipe([prompts], gen_config=gen_config, log_level='INFO', max_log_len=10)[0].text

            case_result, reason = assert_result(response, prompt_detail.values(), model_name)
            prompts.append({'role': 'assistant', 'content': response})
            file.writelines('output:' + response + '\n')
            file.writelines('result:' + str(case_result) + ', reason:' + reason + '\n')
        file.close()

    del pipe
    torch.cuda.empty_cache()


def assert_pipeline_chat_log(config, cases_info, model_case, type, worker_id: str = ''):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path,
                              '_'.join(['pipeline', 'config', type, worker_id,
                                        model_case.split('/')[1] + '.log']))

    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)

    for case in cases_info.keys():
        if ('coder' in model_case or 'CodeLlama' in model_case) and 'code' not in case:
            continue
        msg = 'result is empty, please check again'
        result = False
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, '_'.join(['pipeline', 'chat', type, worker_id,
                                    model_case.split('/')[1], case + '.log']))

            allure.attach.file(pipeline_chat_log, attachment_type=allure.attachment_type.TEXT)

            with open(pipeline_chat_log, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if 'result:False, reason:' in line:
                        result = False
                        msg = line
                        break
                    if 'result:True, reason:' in line and not result:
                        result = True
                        msg = ''

            with assume:
                assert result, msg


def save_pipeline_common_log(config, log_name, result, content, msg: str = '', write_type: str = 'w'):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    file = open(config_log, write_type)
    file.writelines(f'result:{result}, reason: {msg}, content: {content}')  # noqa E231
    file.close()


def assert_pipeline_common_log(config, log_name):
    log_path = config.get('log_path')

    config_log = os.path.join(log_path, log_name)
    allure.attach.file(config_log, attachment_type=allure.attachment_type.TEXT)

    msg = 'result is empty, please check again'
    result = False
    with open(config_log, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'result:False, reason:' in line:
                result = False
                msg = line
                break
            if 'result:True, reason:' in line and not result:
                result = True
                msg = ''
    subprocess.run([' '.join(['rm -rf', config_log])],
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8')

    assert result, msg


def assert_pipeline_single_return(output, logprobs_num: int = 0):
    result = assert_pipeline_single_element(output, is_last=True, logprobs_num=logprobs_num)
    if not result:
        return result, 'single_stream_element is wrong'
    return result & (len(output.token_ids) == output.generate_token_len
                     or len(output.token_ids) == output.generate_token_len - 1), 'token_is len is not correct'


def assert_pipeline_batch_return(output, size: int = 1):
    if len(output) != size:
        return False, 'length is not correct'
    for single_output in output:
        result, msg = assert_pipeline_single_return(single_output)
        if not result:
            return result, msg
    return True, ''


def assert_pipeline_single_stream_return(output, logprobs_num: int = 0):
    for i in range(0, len(output) - 1):
        if not assert_pipeline_single_element(output[i], is_stream=True, logprobs_num=logprobs_num):
            return False, f'single_stream_element is false, index is {i}'
    if assert_pipeline_single_element(output[-1], is_stream=True, is_last=True, logprobs_num=logprobs_num) is False:
        return False, 'last single_stream_element is false'
    return True, ''


def assert_pipeline_batch_stream_return(output, size: int = 1):
    for i in range(size):
        output_list = [item for item in output if item.index == i]
        result, msg = assert_pipeline_single_stream_return(output_list)
        if not result:
            return result, msg
    return True, ''


def assert_pipeline_single_element(output, is_stream: bool = False, is_last: bool = False, logprobs_num: int = 0):
    result = True
    result &= output.generate_token_len > 0
    result &= output.input_token_len > 0
    result &= output.index >= 0
    if is_last:
        result &= len(output.text) >= 0
        result &= output.finish_reason in ['stop', 'length']
        if is_stream:
            result &= output.token_ids is None or output.token_ids == []
        else:
            result &= len(output.token_ids) > 0
    else:
        result &= len(output.text) > 0
        result &= output.finish_reason is None
        result &= len(output.token_ids) > 0
    if logprobs_num == 0 or (is_last and is_stream):
        result &= output.logprobs is None
    else:
        if is_stream:
            result &= len(output.logprobs) == 1
        else:
            result &= len(output.logprobs) == output.generate_token_len or len(
                output.logprobs) == output.generate_token_len + 1
        if result:
            for content in output.logprobs:
                result &= len(content.keys()) <= logprobs_num
                for key in content.keys():
                    result &= type(content.get(key)) == float
    return result


PIC1 = 'tiger.jpeg'
PIC2 = 'human-pose.jpg'
PIC_BEIJING = 'Beijing_Small.jpeg'
PIC_CHONGQING = 'Chongqing_Small.jpeg'
PIC_REDPANDA = 'redpanda.jpg'
PIC_PANDA = 'panda.jpg'
DESC = 'What are the similarities and differences between these two images.'
DESC_ZH = '两张图有什么相同和不同的地方.'


def run_pipeline_vl_chat_test(config, model_case, backend, worker_id: str = '', quant_policy: int = None):
    log_path = config.get('log_path')
    tp = get_tp_num(config, model_case)
    model_path = config.get('model_path')
    hf_path = model_path + '/' + model_case
    resource_path = config.get('resource_path')

    if 'pytorch' in backend:
        backend_config = PytorchEngineConfig(tp=tp, session_len=32576)
        if not is_bf16_supported():
            backend_config.dtype = 'float16'
    else:
        backend_config = TurbomindEngineConfig(tp=tp, session_len=32576)

    if 'llava' in model_case:
        backend_config.model_name = 'vicuna'
    if '4bit' in model_case.lower() or 'awq' in model_case.lower():
        backend_config.model_format = 'awq'
    if quant_policy is not None:
        backend_config.quant_policy = quant_policy

    if not is_bf16_supported():
        backend_config.dtype = 'float16'
    pipe = pipeline(hf_path, backend_config=backend_config)

    pipeline_chat_log = os.path.join(log_path, 'pipeline_vl_chat_' + model_case.split('/')[1] + worker_id + '.log')
    file = open(pipeline_chat_log, 'w')

    image = load_image(f'{resource_path}/{PIC1}')

    if 'deepseek' in model_case:
        prompt = f'describe this image{IMAGE_TOKEN}'
    else:
        prompt = 'describe this image'

    log_string = '\n'.join([
        'reproduce config info:', 'from lmdeploy import pipeline', 'from lmdeploy.messages import PytorchEngineConfig',
        'from lmdeploy.messages import TurbomindEngineConfig', 'engine_config = ' + str(backend_config),
        'pipe = pipeline("' + hf_path + '",  backend_config=engine_config)', f'res = pipe((\'{prompt}\', {image}))'
    ])
    file.writelines(log_string)
    print(log_string)
    response = pipe((prompt, image))
    result = 'tiger' in response.text.lower() or '虎' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: simple example tiger not in ' + response.text + '\n')

    prompts = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': prompt
        }, {
            'type': 'image_url',
            'image_url': {
                'url': f'{resource_path}/{PIC1}'
            }
        }]
    }]
    response = pipe(prompts, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'tiger' in response.text.lower() or '虎' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: OpenAI format example: tiger not in ' + response.text + '\n')

    image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
    images = [load_image(img_url) for img_url in image_urls]
    response = pipe((prompt, images))
    result = 'tiger' in response.text.lower() or 'ski' in response.text.lower() or '虎' in response.text.lower(
    ) or '滑雪' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: Multi-images example: tiger or ski not in ' + response.text +
                    '\n')

    image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
    prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = ('ski' in response[0].text.lower() or '滑雪'
              in response[0].text.lower()) and ('tiger' in response[1].text.lower() or '虎' in response[1].text.lower())
    file.writelines('result:' + str(result) + ', reason: Batch example: ski or tiger not in ' + str(response) + '\n')

    image = load_image(f'{resource_path}/{PIC2}')
    sess = pipe.chat((prompt, image))
    result = 'ski' in sess.response.text.lower() or '滑雪' in sess.response.text.lower()
    file.writelines('result:' + str(result) + ', reason: Multi-turn example: ski not in ' + sess.response.text + '\n')
    sess = pipe.chat('What is the woman doing?', session=sess)
    result = 'ski' in sess.response.text.lower() or '滑雪' in sess.response.text.lower()
    file.writelines('result:' + str(result) + ', reason: Multi-turn example: ski not in ' + sess.response.text + '\n')

    if 'internvl' in model_case.lower():
        internvl_vl_testcase(pipe, file, resource_path)
        internvl_vl_testcase(pipe, file, resource_path, 'cn')
    if 'minicpm' in model_case.lower():
        MiniCPM_vl_testcase(pipe, file, resource_path)
    if 'qwen' in model_case.lower():
        Qwen_vl_testcase(pipe, file, resource_path)

    file.close()

    del pipe
    torch.cuda.empty_cache()


def internvl_vl_testcase(pipe, file, resource_path, lang='en'):
    if lang == 'cn':
        description = DESC_ZH
    else:
        description = DESC
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text=f'{IMAGE_TOKEN}{IMAGE_TOKEN}\n{description}'),
                 dict(type='image_url', image_url=dict(max_dynamic_patch=12, url=f'{resource_path}/{PIC_REDPANDA}')),
                 dict(type='image_url', image_url=dict(max_dynamic_patch=12, url=f'{resource_path}/{PIC_PANDA}'))
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images: panda not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images second: panda not in ' + response.text + '\n')

    # multi-image multi-round conversation, separate images
    messages = [
        dict(
            role='user',
            content=[
                dict(
                    type='text',
                    text=f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n' +  # noqa E251,E501
                    description),
                dict(type='image_url', image_url=dict(max_dynamic_patch=12, url=f'{resource_path}/{PIC_REDPANDA}')),
                dict(type='image_url', image_url=dict(max_dynamic_patch=12, url=f'{resource_path}/{PIC_PANDA}'))
            ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower() or 'same' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: separate images: panda not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower() or 'same' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: separate images second: panda not in ' + response.text + '\n')

    # video multi-round conversation
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
        return frame_indices

    def load_video(video_path, bound=None, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        imgs = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            imgs.append(img)
        return imgs

    video_path = f'{resource_path}/red-panda.mp4'
    imgs = load_video(video_path, num_segments=8)

    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

    if lang == 'cn':
        question += '小熊猫在做什么？'
    else:
        question += 'What is the red panda doing?'

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({
            'type': 'image_url',
            'image_url': {
                'max_dynamic_patch': 1,
                'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'  # noqa E231
            }
        })

    messages = [dict(role='user', content=content)]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: video images: red panda not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    if lang == 'cn':
        messages.append(dict(role='user', content='描述视频详情，不要重复'))
    else:
        messages.append(dict(role='user', content='Describe this video in detail. Don\'t repeat.'))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'red panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: video images: red panda not in ' + response.text + '\n')


def llava_vl_testcase(pipe, file, resource_path):
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/{PIC_BEIJING}')),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/{PIC_CHONGQING}'))
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'city' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images: city not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'city' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images second: city not in ' + response.text + '\n')


def MiniCPM_vl_testcase(pipe, file, resource_path):
    # Chat with multiple images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url', image_url=dict(max_slice_nums=9, url=f'{resource_path}/{PIC_REDPANDA}')),
                 dict(type='image_url', image_url=dict(max_slice_nums=9, url=f'{resource_path}/{PIC_PANDA}'))
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: multiple images: panda not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: multiple images second: panda not in ' + response.text + '\n')

    # In-context few-shot learning
    question = 'production date'
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/data1.jpeg')),
             ]),
        dict(role='assistant', content='2021.08.29'),
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/data2.jpeg')),
             ]),
        dict(role='assistant', content='1999.05.15'),
        dict(role='user',
             content=[
                 dict(type='text', text=question),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/data3.jpeg')),
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = '2021' in response.text.lower() or '14' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: in context learning: 2021 or 14 not in ' + response.text +
                    '\n')

    # Chat with video
    MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

    def encode_video(video_path):

        def uniform_sample(length, n):
            gap = len(length) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [length[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames

    video_path = f'{resource_path}red-panda.mp4'
    frames = encode_video(video_path)
    question = 'Describe the video'

    content = [dict(type='text', text=question)]
    for frame in frames:
        content.append(
            dict(type='image_url',
                 image_url=dict(use_image_id=False,
                                max_slice_nums=2,
                                url=f'data:image/jpeg;base64,{encode_image_base64(frame)}')))  # noqa E231

    messages = [dict(role='user', content=content)]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'red panda' in response.text.lower() or '熊猫' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: video example: panda not in ' + response.text + '\n')


def Qwen_vl_testcase(pipe, file, resource_path):
    # multi-image multi-round conversation, combined images
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/{PIC_BEIJING}')),
                 dict(type='image_url', image_url=dict(url=f'{resource_path}/{PIC_CHONGQING}'))
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'city' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images: city not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'city' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: combined images second: city not in ' + response.text + '\n')

    # image resolution for performance boost
    min_pixels = 64 * 28 * 28
    max_pixels = 64 * 28 * 28
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the two images in detail.'),
                 dict(type='image_url',
                      image_url=dict(min_pixels=min_pixels, max_pixels=max_pixels,
                                     url=f'{resource_path}/{PIC_BEIJING}')),
                 dict(type='image_url',
                      image_url=dict(min_pixels=min_pixels,
                                     max_pixels=max_pixels,
                                     url=f'{resource_path}/{PIC_CHONGQING}'))
             ])
    ]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'ski' in response.text.lower() or '滑雪' in response.text.lower()
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: performance boost: buildings not in ' + response.text + '\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    result = 'buildings' in response.text.lower() or '楼' in response.text.lower() or 'skyline' in response.text.lower(
    ) or 'cityscape' in response.text.lower()
    file.writelines('result:' + str(result) + ', reason: performance boost second: buildings not in ' + response.text +
                    '\n')


def assert_pipeline_vl_chat_log(config, model_case, worker_id):
    log_path = config.get('log_path')

    pipeline_chat_log = os.path.join(log_path, 'pipeline_vl_chat_' + model_case.split('/')[1] + worker_id + '.log')

    allure.attach.file(pipeline_chat_log, attachment_type=allure.attachment_type.TEXT)

    msg = 'result is empty, please check again'
    result = False
    with open(pipeline_chat_log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'result:False, reason:' in line:
                result = False
                msg = line
                break
            if 'result:True, reason:' in line and not result:
                result = True
                msg = ''

    with assume:
        assert result, msg
