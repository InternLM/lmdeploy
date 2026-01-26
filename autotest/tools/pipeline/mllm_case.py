import json

import fire
import numpy as np
from PIL import Image

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)

PIC1 = 'tiger.jpeg'
PIC2 = 'human-pose.jpg'
PIC_BEIJING = 'Beijing_Small.jpeg'
PIC_CHONGQING = 'Chongqing_Small.jpeg'
PIC_REDPANDA = 'redpanda.jpg'
PIC_PANDA = 'panda.jpg'
DESC = 'What are the similarities and differences between these two images.'
DESC_ZH = '两张图有什么相同和不同的地方.'


def run_pipeline_mllm_test(model_path, run_config, resource_path, is_pr_test: bool = False):
    backend = run_config.get('backend')
    communicator = run_config.get('communicator')
    quant_policy = run_config.get('quant_policy')
    extra_params = run_config.get('extra_params', {})
    parallel_config = run_config.get('parallel_config', {})

    if 'pytorch' == backend:
        backend_config = PytorchEngineConfig(session_len=32576, quant_policy=quant_policy, cache_max_entry_count=0.6)
    else:
        backend_config = TurbomindEngineConfig(session_len=32576,
                                               communicator=communicator,
                                               quant_policy=quant_policy,
                                               cache_max_entry_count=0.6)

    # quant format
    model_lower = model_path.lower()
    if 'w4' in model_lower or '4bits' in model_lower or 'awq' in model_lower:
        backend_config.model_format = 'awq'
    elif 'gptq' in model_lower:
        backend_config.model_format = 'gptq'

    # Parallel config
    for para_key in ('dp', 'ep', 'cp'):
        if para_key in parallel_config:
            setattr(backend_config, para_key, parallel_config[para_key])
    if 'tp' in parallel_config and parallel_config['tp'] > 1:
        backend_config.tp = parallel_config['tp']

    # Extra params
    for key, value in extra_params.items():
        try:
            setattr(backend_config, key, value)
        except AttributeError:
            print(f"Warning: Cannot set attribute '{key}' on backend_config. Skipping.")

    print('backend_config config: ' + str(backend_config))
    pipe = pipeline(model_path, backend_config=backend_config)

    image = load_image(f'{resource_path}/{PIC1}')

    if 'deepseek' in model_lower:
        prompt = f'describe this image{IMAGE_TOKEN}'
    else:
        prompt = 'describe this image'

    response = pipe((prompt, image)).text
    print('[caseresult single1 start]' + json.dumps(response, ensure_ascii=False) + '[caseresult single1 end]\n')

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
    print('[caseresult single2 start]' + json.dumps(response.text, ensure_ascii=False) + '[caseresult single2 end]\n')

    image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
    images = [load_image(img_url) for img_url in image_urls]
    response = pipe((prompt, images))
    print('[caseresult multi-imagese start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult multi-imagese end]\n')

    image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
    prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print('[caseresult batch-example1 start]' + json.dumps(response[0].text, ensure_ascii=False) +
          '[caseresult batch-example1 end]\n')
    print('[caseresult batch-example2 start]' + json.dumps(response[1].text, ensure_ascii=False) +
          '[caseresult batch-example2 end]\n')

    image = load_image(f'{resource_path}/{PIC2}')
    sess = pipe.chat((prompt, image))
    print('[caseresult multi-turn1 start]' + json.dumps(sess.response.text, ensure_ascii=False) +
          '[caseresult multi-turn1 end]\n')
    sess = pipe.chat('What is the woman doing?', session=sess)
    print('[caseresult multi-turn2 start]' + json.dumps(sess.response.text, ensure_ascii=False) +
          '[caseresult multi-turn2 end]\n')

    if not is_pr_test:
        if 'internvl' in model_path.lower() and 'internvl2-4b' not in model_path.lower():
            internvl_vl_testcase(pipe,
                                 resource_path,
                                 use_decord=run_config.get('extra_params', {}).get('device', 'cuda') == 'cuda')
            internvl_vl_testcase(pipe,
                                 resource_path,
                                 lang='cn',
                                 use_decord=run_config.get('extra_params', {}).get('device', 'cuda') == 'cuda')
        if 'minicpm' in model_path.lower():
            MiniCPM_vl_testcase(pipe, resource_path)
        if 'qwen' in model_path.lower():
            Qwen_vl_testcase(pipe, resource_path)

    pipe.close()


def internvl_vl_testcase(pipe, resource_path, lang='en', use_decord=True):
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
    print(f'[caseresult internvl-combined-images-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-combined-images-{lang} end]\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print(f'[caseresult internvl-combined-images2-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-combined-images2-{lang} end]\n')

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
    print(f'[caseresult internvl-separate-images-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-separate-images-{lang} end]\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print(f'[caseresult internvl-separate-images2-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-separate-images2-{lang} end]\n')

    if use_decord:

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
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
            frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            imgs = []
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
                imgs.append(img)
            return imgs
    else:
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
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f'Cannot open video file: {video_path}')

            max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            imgs = []

            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame).convert('RGB')
                    imgs.append(img)

            cap.release()
            return imgs

    video_path = resource_path + '/red-panda.mp4'
    imgs = load_video(video_path, num_segments=8)

    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

    if lang == 'cn':
        question += '视频里有什么动物，它在做什么？'
    else:
        question += 'What animals are in the video, and what are they doing?'

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
    print(f'[caseresult internvl-video-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-video-{lang} end]\n')

    messages.append(dict(role='assistant', content=response.text))
    if lang == 'cn':
        messages.append(dict(role='user', content='描述视频详情，不要重复'))
    else:
        messages.append(dict(role='user', content='Describe this video in detail. Don\'t repeat.'))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print(f'[caseresult internvl-video2-{lang} start]' + json.dumps(response.text, ensure_ascii=False) +
          f'[caseresult internvl-video2-{lang} end]\n')


def MiniCPM_vl_testcase(pipe, resource_path):
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
    print('[caseresult minicpm-combined-images start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult minicpm-combined-images end]\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print('[caseresult minicpm-combined-images2 start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult minicpm-combined-images2 end]\n')

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
    print('[caseresult minicpm-fewshot start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult minicpm-fewshot end]\n')

    # Chat with video
    MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

    def encode_video(video_path):

        def uniform_sample(length, n):
            gap = len(length) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [length[i] for i in idxs]

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Cannot open video file: {video_path}')

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_fps = round(fps / 1)  # FPS
        frame_idx = [i for i in range(0, total_frames, sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

        frames = []
        for idx in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame.astype('uint8')).convert('RGB'))

        cap.release()
        print('num frames:', len(frames))
        return frames

    video_path = resource_path + '/red-panda.mp4'
    frames = encode_video(video_path)
    question = 'What animals are in the video, and what are they doing?'

    content = [dict(type='text', text=question)]
    for frame in frames:
        content.append(
            dict(type='image_url',
                 image_url=dict(use_image_id=False,
                                max_slice_nums=2,
                                url=f'data:image/jpeg;base64,{encode_image_base64(frame)}')))  # noqa E231

    messages = [dict(role='user', content=content)]
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print('[caseresult minicpm-video start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult minicpm-video end]\n')


def Qwen_vl_testcase(pipe, resource_path):
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
    print('[caseresult qwen-combined-images start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult qwen-combined-images end]\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print('[caseresult qwen-combined-images2 start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult qwen-combined-images2 end]\n')

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
    print('[caseresult qwen-performance-images start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult qwen-performance-images end]\n')

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    print('[caseresult qwen-performance-images2 start]' + json.dumps(response.text, ensure_ascii=False) +
          '[caseresult qwen-performance-images2 end]\n')


if __name__ == '__main__':
    fire.Fire()
