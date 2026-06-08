import gc
import json
import os
from typing import Any

import fire
import numpy as np
from PIL import Image

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import encode_image_base64, load_image, load_video
from lmdeploy.vl.constants import IMAGE_TOKEN

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)

PIC1 = 'tiger.jpeg'
PIC2 = 'human-pose.jpg'
PIC_BEIJING = 'Beijing_Small.jpeg'
PIC_CHONGQING = 'Chongqing_Small.jpeg'
PIC_REDPANDA = 'redpanda.jpg'
PIC_PANDA = 'panda.jpg'
DESC = 'What are the similarities and differences between these two images.'
DESC_ZH = '两张图有什么相同和不同的地方.'
_MM_DEMO_TOMB_MCQ_JSON_BLOCK = """{
  "question": "How many porcelain jars were discovered in the niches located in the primary chamber of the tomb?",
  "options": [
    "A. 4.",
    "B. 9.",
    "C. 5.",
    "D. 13."
  ]
}"""
MM_DEMO_TOMB_USER_PROMPT = (
    'You are given a multiple-choice problem as JSON (question and options only; there is no answer field). '
    'Watch the entire video, pick the best option from what you see, then reply briefly with the letter '
    '(A, B, C, or D) first and at most one short sentence. Do not output long step-by-step reasoning; '
    'keep the final reply concise.\n\n' + _MM_DEMO_TOMB_MCQ_JSON_BLOCK)

DEFAULT_VIDEO_FILENAME = 'red-panda.mp4'
VIDEO_QWEN3_DEMO_FILENAME = 'N1cdUjctpG8.mp4'
MM_DEMO_MAX_NEW_TOKENS = 24576


def _is_unsupported_video_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = (
        'unsupported message type: video',
        'unsupported message type',
        'not support video',
        'does not support video',
        'video_url',
    )
    return any(m in msg for m in markers)


def _skip_video_error_text(exc: BaseException) -> str:
    return f'SKIPPED_UNSUPPORTED_VIDEO:{exc!s}'


def _is_input_length_error_text(text: str) -> bool:
    rl = text.lower()
    return 'input_length_error' in rl or 'internal error happened' in rl


def _mm_demo_pipeline_log_payload(response) -> dict[str, Any]:
    return {
        'text': response.text,
        'finish_reason': response.finish_reason,
    }


def _numpy_video_to_pil_list(frames: np.ndarray) -> list[Image.Image]:
    images: list[Image.Image] = []
    for i in range(int(frames.shape[0])):
        images.append(Image.fromarray(frames[i].astype('uint8')).convert('RGB'))
    return images


def load_video_sampled_pil(video_path: str, num_frames: int, **kwargs: Any) -> tuple[list[Image.Image], dict[str, Any]]:
    frames, meta = load_video(video_path, num_frames=num_frames, **kwargs)
    return _numpy_video_to_pil_list(frames), meta


def _is_video_mixed_whitelist_model(model_path: str) -> bool:
    """Only run video/mixed-mm cases for selected model families."""
    m = model_path.lower()
    whitelist = ('qwen3-vl', 'qwen3.5', 'interns2-preview')
    return any(p in m for p in whitelist)


def _log_case_result(case_name: str, payload: Any) -> None:
    dumped = json.dumps(payload, ensure_ascii=False)
    print(f'[caseresult {case_name} start]{dumped}[caseresult {case_name} end]\n')


def _best_effort_runtime_cleanup(pipe=None) -> None:
    if pipe is not None:
        try:
            pipe.close()
        except Exception as exc:
            print(f'Warning: failed to close pipeline cleanly: {exc!s}')
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


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
    pipe = None
    try:
        pipe = pipeline(model_path, backend_config=backend_config, trust_remote_code=True)
        enable_video_mixed = _is_video_mixed_whitelist_model(model_path)
        image = load_image(f'{resource_path}/{PIC1}')

        if 'deepseek' in model_lower:
            prompt = f'describe this image{IMAGE_TOKEN}'
        else:
            prompt = 'describe this image'

        response = pipe((prompt, image)).text
        _log_case_result('single1', response)

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
        _log_case_result('single2', response.text)

        image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
        images = [load_image(img_url) for img_url in image_urls]
        response = pipe((prompt, images))
        _log_case_result('multi-imagese', response.text)

        image_urls = [f'{resource_path}/{PIC2}', f'{resource_path}/{PIC1}']
        prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
        response = pipe(prompts, gen_config=gen_config, log_level='INFO', max_log_len=10)
        _log_case_result('batch-example1', response[0].text)
        _log_case_result('batch-example2', response[1].text)

        image = load_image(f'{resource_path}/{PIC2}')
        sess = pipe.chat((prompt, image))
        _log_case_result('multi-turn1', sess.response.text)
        sess = pipe.chat('What is the woman doing?', session=sess)
        _log_case_result('multi-turn2', sess.response.text)

        if not is_pr_test:
            if 'internvl' in model_path.lower() and 'internvl2-4b' not in model_path.lower():
                internvl_vl_testcase(pipe, resource_path, enable_video_mixed=enable_video_mixed)
                internvl_vl_testcase(pipe, resource_path, lang='cn', enable_video_mixed=enable_video_mixed)
            if 'minicpm' in model_path.lower():
                MiniCPM_vl_testcase(pipe, resource_path, enable_video_mixed=enable_video_mixed)
            if 'qwen' in model_path.lower():
                Qwen_vl_testcase(pipe, resource_path, enable_video_mixed=enable_video_mixed)
    finally:
        _best_effort_runtime_cleanup(pipe)


def internvl_vl_testcase(pipe, resource_path, lang='en', enable_video_mixed: bool = True):
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
    _log_case_result(f'internvl-combined-images-{lang}', response.text)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    _log_case_result(f'internvl-combined-images2-{lang}', response.text)

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
    _log_case_result(f'internvl-separate-images-{lang}', response.text)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=description))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    _log_case_result(f'internvl-separate-images2-{lang}', response.text)

    if not enable_video_mixed:
        _log_case_result(f'internvl-video-{lang}', 'SKIPPED_VIDEO_GATED_MODEL')
        _log_case_result(f'internvl-video2-{lang}', 'SKIPPED_VIDEO_GATED_MODEL')
    else:
        try:
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
            _log_case_result(f'internvl-video-{lang}', response.text)

            messages.append(dict(role='assistant', content=response.text))
            if lang == 'cn':
                messages.append(dict(role='user', content='描述视频详情，不要重复'))
            else:
                messages.append(dict(role='user', content='Describe this video in detail. Don\'t repeat.'))
            response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
            _log_case_result(f'internvl-video2-{lang}', response.text)
        except Exception as exc:
            if not _is_unsupported_video_error(exc):
                raise
            skipped = _skip_video_error_text(exc)
            _log_case_result(f'internvl-video-{lang}', skipped)
            _log_case_result(f'internvl-video2-{lang}', skipped)


def MiniCPM_vl_testcase(pipe, resource_path, enable_video_mixed: bool = True):
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
    _log_case_result('minicpm-combined-images', response.text)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    _log_case_result('minicpm-combined-images2', response.text)

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
    _log_case_result('minicpm-fewshot', response.text)


    if not enable_video_mixed:
        _log_case_result('minicpm-video', 'SKIPPED_VIDEO_GATED_MODEL')
    else:
        try:
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
            _log_case_result('minicpm-video', response.text)
        except Exception as exc:
            if not _is_unsupported_video_error(exc):
                raise
            _log_case_result('minicpm-video', _skip_video_error_text(exc))


def Qwen_vl_testcase(pipe, resource_path, enable_video_mixed: bool = True):
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
    _log_case_result('qwen-combined-images', response.text)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    _log_case_result('qwen-combined-images2', response.text)

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
    _log_case_result('qwen-performance-images', response.text)

    messages.append(dict(role='assistant', content=response.text))
    messages.append(dict(role='user', content=DESC))
    response = pipe(messages, gen_config=gen_config, log_level='INFO', max_log_len=10)
    _log_case_result('qwen-performance-images2', response.text)


    demo_path = os.path.join(resource_path, VIDEO_QWEN3_DEMO_FILENAME)
    if not enable_video_mixed:
        _log_case_result('qwen3-demo-video', 'SKIPPED_VIDEO_GATED_MODEL')
    elif not os.path.isfile(demo_path):
        _log_case_result('qwen3-demo-video', 'SKIPPED_NO_DEMO_MP4')
    else:
        try:
            demo_q = MM_DEMO_TOMB_USER_PROMPT
            vmsg = [{
                'role':
                'user',
                'content': [
                    {
                        'type': 'video_url',
                        'video_url': {
                            'url': demo_path,
                        },
                    },
                    {
                        'type': 'text',
                        'text': demo_q,
                    },
                ],
            }]
            mm_gen_config = GenerationConfig(
                max_new_tokens=MM_DEMO_MAX_NEW_TOKENS,
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
            if _is_input_length_error_text(response.text):
                _log_case_result('qwen3-demo-video', 'SKIPPED_INPUT_LENGTH_ERROR')
            else:
                _log_case_result('qwen3-demo-video', _mm_demo_pipeline_log_payload(response))
        except Exception as exc:
            if _is_unsupported_video_error(exc):
                err = _skip_video_error_text(exc)
            else:
                err = f'PIPELINE_VIDEO_ERROR:{exc!s}'
            _log_case_result('qwen3-demo-video', err)

    rp_video = os.path.join(resource_path, DEFAULT_VIDEO_FILENAME)
    if not enable_video_mixed:
        _log_case_result('qwen-mixed-image-text-video', 'SKIPPED_VIDEO_GATED_MODEL')
    elif not os.path.isfile(rp_video):
        _log_case_result('qwen-mixed-image-text-video', 'SKIPPED_NO_RED_PANDA_MP4')
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
            _log_case_result('qwen-mixed-image-text-video', response.text)
        except Exception as exc:
            _log_case_result('qwen-mixed-image-text-video', f'PIPELINE_MIXED_MM_ERROR:{exc!s}')


if __name__ == '__main__':
    fire.Fire()
