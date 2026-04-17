import json
import os
import sys
from typing import Any

_autotest_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _autotest_root not in sys.path:
    sys.path.insert(0, _autotest_root)

import fire  # noqa: E402
import numpy as np  # noqa: E402
from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline  # noqa: E402
from lmdeploy.vl import encode_image_base64, load_image, load_video  # noqa: E402
from lmdeploy.vl.constants import IMAGE_TOKEN  # noqa: E402
from PIL import Image  # noqa: E402
from utils.constant import MM_DEMO_TOMB_USER_PROMPT  # noqa: E402

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)

PIC1 = 'tiger.jpeg'
PIC2 = 'human-pose.jpg'
PIC_BEIJING = 'Beijing_Small.jpeg'
PIC_CHONGQING = 'Chongqing_Small.jpeg'
PIC_REDPANDA = 'redpanda.jpg'
PIC_PANDA = 'panda.jpg'
DESC = 'What are the similarities and differences between these two images.'
DESC_ZH = '两张图有什么相同和不同的地方.'

DEFAULT_VIDEO_FILENAME = 'red-panda.mp4'
VIDEO_QWEN3_DEMO_FILENAME = 'N1cdUjctpG8.mp4'


def _numpy_video_to_pil_list(frames: np.ndarray) -> list[Image.Image]:
    images: list[Image.Image] = []
    for i in range(int(frames.shape[0])):
        images.append(Image.fromarray(frames[i].astype('uint8')).convert('RGB'))
    return images


def load_video_sampled_pil(video_path: str, num_frames: int, **kwargs: Any) -> tuple[list[Image.Image], dict[str, Any]]:
    frames, meta = load_video(video_path, num_frames=num_frames, **kwargs)
    return _numpy_video_to_pil_list(frames), meta


def run_pipeline_mllm_test(model_path, run_config, resource_path, is_pr_test: bool = False):
    backend = run_config.get('backend')
    communicator = run_config.get('communicator')
    quant_policy = run_config.get('quant_policy')
    extra_params = run_config.get('extra_params', {})
    parallel_config = run_config.get('parallel_config', {})

    if 'pytorch' == backend:
        backend_config = PytorchEngineConfig(session_len=65152, quant_policy=quant_policy, cache_max_entry_count=0.6)
    else:
        backend_config = TurbomindEngineConfig(session_len=65152,
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
    # Map CLI param names to PytorchEngineConfig attribute names
    param_name_map = {'device': 'device_type'}
    for key, value in extra_params.items():
        attr_name = param_name_map.get(key, key)
        try:
            setattr(backend_config, attr_name, value)
        except AttributeError:
            print(f"Warning: Cannot set attribute '{attr_name}' on backend_config. Skipping.")

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
            internvl_vl_testcase(pipe, resource_path)
            internvl_vl_testcase(pipe, resource_path, lang='cn')
        if 'minicpm' in model_path.lower():
            MiniCPM_vl_testcase(pipe, resource_path)
        if 'qwen' in model_path.lower():
            Qwen_vl_testcase(pipe, resource_path)

    pipe.close()


def internvl_vl_testcase(pipe, resource_path, lang='en'):
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

    # video multi-round conversation (uniform ``num_frames`` via lmdeploy.vl.load_video)
    video_path = f'{resource_path}/{DEFAULT_VIDEO_FILENAME}'
    imgs, _ = load_video_sampled_pil(video_path, num_frames=8)

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

    # Chat with video (fixed frame budget; same decoder as REST ``video_url``)
    max_video_frames = 32
    video_path = f'{resource_path}/{DEFAULT_VIDEO_FILENAME}'
    frames, video_meta = load_video_sampled_pil(video_path, num_frames=max_video_frames)
    print('num frames:', len(frames), 'meta:', video_meta.get('frames_indices'))
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

    # Qwen2.5/3-VL: native ``video`` + same knobs as REST ``extra_body`` (top_k / mm_processor_kwargs).
    demo_path = os.path.join(resource_path, VIDEO_QWEN3_DEMO_FILENAME)
    if not os.path.isfile(demo_path):
        print('[caseresult qwen3-demo-video start]' +
              json.dumps('SKIPPED_NO_DEMO_MP4', ensure_ascii=False) + '[caseresult qwen3-demo-video end]\n')
    else:
        try:
            frames, vmeta = load_video(demo_path, num_frames=16, fps=2)
            demo_q = MM_DEMO_TOMB_USER_PROMPT
            vmsg = [{
                'role':
                'user',
                'content': [
                    {
                        'type': 'video',
                        'data': frames,
                        'video_metadata': vmeta,
                    },
                    {
                        'type': 'text',
                        'text': demo_q,
                    },
                ],
            }]
            mm_gen_config = GenerationConfig(
                max_new_tokens=24576,
                min_new_tokens=10,
                top_k=20,
                temperature=0.3,
                top_p=0.95,
            )
            response = pipe(
                vmsg,
                gen_config=mm_gen_config,
                log_level='INFO',
                max_log_len=10,
                mm_processor_kwargs={
                    'fps': 2,
                    'do_sample_frames': True,
                },
            )
            print('[caseresult qwen3-demo-video start]' + json.dumps(response.text, ensure_ascii=False) +
                  '[caseresult qwen3-demo-video end]\n')
        except Exception as exc:
            err = json.dumps(f'PIPELINE_VIDEO_ERROR:{exc!s}', ensure_ascii=False)
            print('[caseresult qwen3-demo-video start]' + err + '[caseresult qwen3-demo-video end]\n')

    rp_video = os.path.join(resource_path, DEFAULT_VIDEO_FILENAME)
    if not os.path.isfile(rp_video):
        print('[caseresult qwen-mixed-image-text-video start]' +
              json.dumps('SKIPPED_NO_RED_PANDA_MP4', ensure_ascii=False) +
              '[caseresult qwen-mixed-image-text-video end]\n')
    else:
        try:
            frames_pil, _vmeta_m = load_video_sampled_pil(rp_video, num_frames=6, fps=1)
            mixed_content = [
                {
                    'type':
                    'text',
                    'text': (
                        'You are given one still image, then several frames from a short video in order. '
                        'In 2-4 sentences: name one thing in the still image, and what animal or activity '
                        'you see in the video frames.'),
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'{resource_path}/{PIC1}',
                    },
                },
            ]
            for frame in frames_pil:
                mixed_content.append(
                    dict(
                        type='image_url',
                        image_url=dict(url=f'data:image/jpeg;base64,{encode_image_base64(frame)}'),
                    ))
            mixed_msg = [{'role': 'user', 'content': mixed_content}]
            response = pipe(
                mixed_msg,
                gen_config=gen_config,
                log_level='INFO',
                max_log_len=10,
            )
            print('[caseresult qwen-mixed-image-text-video start]' +
                  json.dumps(response.text, ensure_ascii=False) + '[caseresult qwen-mixed-image-text-video end]\n')
        except Exception as exc:
            err = json.dumps(f'PIPELINE_MIXED_MM_ERROR:{exc!s}', ensure_ascii=False)
            print('[caseresult qwen-mixed-image-text-video start]' + err +
                  '[caseresult qwen-mixed-image-text-video end]\n')


if __name__ == '__main__':
    fire.Fire()
