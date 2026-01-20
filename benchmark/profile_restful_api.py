# Modify from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py  # noqa
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py  # noqa
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py  # noqa
"""Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate-range 1,2,4,8,16,32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --multi
"""  # noqa
import argparse
import asyncio
import csv
import io
import json
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pybase64
import requests
from PIL import Image
from tqdm.asyncio import tqdm
from transformers import (AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=None)

_timeout_value = os.getenv('AIOHTTP_TIMEOUT', None)
if _timeout_value is not None:
    try:
        _timeout_value = int(_timeout_value)
        if _timeout_value < 0:
            raise ValueError('AIOHTTP_TIMEOUT cannot be negative.')
        AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=_timeout_value * 60 * 60)
    except ValueError as e:
        print(f'Invalid AIOHTTP_TIMEOUT: {e}.')
        AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=None)

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ''
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ''


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


# trt llm not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith('generate_stream')

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            'accumulate_tokens': True,
            'text_input': request_func_input.prompt,
            'temperature': 0.000001,
            'top_p': 1.0,
            'max_tokens': request_func_input.output_len,
            'stream': True,
            'min_length': request_func_input.output_len,
            'end_id': 1048576,
            **request_func_input.extra_request_body,
        }
        if args.disable_ignore_eos:
            del payload['min_length']
            del payload['end_id']
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data:')

                        data = json.loads(chunk)
                        output.generated_text += data['text_output']
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ''
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith('completions'), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            'model': request_func_input.model,
            'prompt': prompt,
            'temperature': 0.0,
            'best_of': 1,
            'max_tokens': request_func_input.output_len,
            'stream': not args.disable_stream,
            'ignore_eos': not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {'Authorization': f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ''
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data: ')
                        latency = time.perf_counter() - st
                        if chunk == '[DONE]':
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data['choices'][0]['text']:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data['choices'][0]['text']

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ''
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith('chat/completions'), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    if request_func_input.image_data:
        # Build multi-image content: a list of image_url entries followed by the text
        content_items = [{
            'type': 'image_url',
            'image_url': {
                'url': img_url
            },
        } for img_url in request_func_input.image_data]
        content_items.append({'type': 'text', 'text': request_func_input.prompt})
        messages = [
            {
                'role': 'user',
                'content': content_items,
            },
        ]
    else:
        messages = [{'role': 'user', 'content': request_func_input.prompt}]

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            'model': request_func_input.model,
            'messages': messages,
            'temperature': 0.0,
            'max_completion_tokens': request_func_input.output_len,
            'stream': not args.disable_stream,
            'ignore_eos': not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {'Authorization': f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ''
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    if args.disable_stream:
                        # Non-streaming response
                        response_json = await response.json()
                        output.generated_text = response_json['choices'][0]['message']['content']
                        output.success = True
                        output.latency = time.perf_counter() - st
                        output.ttft = (output.latency)  # For non-streaming, TTFT = total latency
                        output.output_len = response_json.get('usage', {}).get('completion_tokens', output_len)
                    else:
                        # Streaming response
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data: ')
                            latency = time.perf_counter() - st
                            if chunk == '[DONE]':
                                pass
                            else:
                                data = json.loads(chunk)

                                # Check if this chunk contains content
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')

                                if content:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += content

                                # Check for usage info in final chunk
                                output_len = (data.get('usage') or {}).get('completion_tokens', output_len)

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len
                else:
                    output.error = ((response.reason or '') + ': ' + (await response.text()))
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_sglang_generate(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            'text': prompt,
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': request_func_input.output_len,
                'ignore_eos': not args.disable_ignore_eos,
            },
            'stream': not args.disable_stream,
            **request_func_input.extra_request_body,
        }
        headers = {}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ''
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        # print(chunk_bytes)

                        chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data: ')
                        latency = time.perf_counter() - st
                        if chunk == '[DONE]':
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data['text']:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data['text']

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ''
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_gserver(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError()


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('SGLANG_USE_MODELSCOPE', 'False').lower() == 'true':
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=['.*.pt', '.*.safetensors', '.*.bin'],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(pretrained_model_name_or_path: str, ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path.endswith('.json') or pretrained_model_name_or_path.endswith('.model'):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)


def get_processor(pretrained_model_name_or_path: str, ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    assert (pretrained_model_name_or_path is not None and pretrained_model_name_or_path != '')
    if pretrained_model_name_or_path.endswith('.json') or pretrained_model_name_or_path.endswith('.model'):
        from sglang.srt.utils.hf_transformers_utils import get_processor

        return get_processor(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)


ASYNC_REQUEST_FUNCS = {
    'sglang': async_request_sglang_generate,
    'sglang-native': async_request_sglang_generate,
    'sglang-oai': async_request_openai_completions,
    'sglang-oai-chat': async_request_openai_chat_completions,
    'vllm': async_request_openai_completions,
    'vllm-chat': async_request_openai_chat_completions,
    'lmdeploy': async_request_openai_completions,
    'lmdeploy-chat': async_request_openai_chat_completions,
    'trt': async_request_trt_llm,
    'gserver': async_request_gserver,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_input_text: int
    total_input_vision: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float


SHAREGPT_URL = 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json'  # noqa


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join('/tmp', url.split('/')[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f'Downloading from {url} to {filename}')

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0


def sample_sharegpt_requests(dataset_path: str,
                             num_requests: int,
                             tokenizer: PreTrainedTokenizerBase,
                             fixed_output_len: Optional[int] = None) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError('output_len too small')

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (len(completion_token_ids) if fixed_output_len is None else fixed_output_len)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or (prompt_len + output_len > 2048 and fixed_output_len is None):
            # Prune too long sequences.
            continue

        filtered_dataset.append(DatasetRow(
            prompt=prompt,
            prompt_len=prompt_len,
            output_len=output_len,
        ))

    print(f'#Input tokens: {sum(x.prompt_len for x in filtered_dataset)}')
    print(f'#Output tokens: {sum(x.output_len for x in filtered_dataset)}')
    return filtered_dataset


def compute_random_lens(full_len: int, range_ratio: float, num: int):
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    )


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> List[DatasetRow]:

    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    # Sample token ids from ShareGPT and repeat/truncate them to
    # satisfy the input_lens

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
    # remove the empty prompt
    dataset = [(query, answer) for query, answer in dataset if len(query) > 0]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    input_requests: List[DatasetRow] = []
    origin_output_lens: List[int] = []
    for i in range(num_prompts):
        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        origin_output_lens.append(len(completion_token_ids))

        if prompt_len > input_lens[i]:
            input_ids = prompt_token_ids[:input_lens[i]]
        else:
            ratio = (input_lens[i] + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[:input_lens[i]]
        prompt = tokenizer.decode(input_ids)
        input_requests.append(DatasetRow(
            prompt=prompt,
            prompt_len=int(input_lens[i]),
            output_len=int(output_lens[i]),
        ))

    print(f'#Input tokens: {sum(x.prompt_len for x in input_requests)}')
    print(f'#Output tokens: {sum(x.output_len for x in input_requests)}')
    return input_requests


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '1080p', '720p', '360p'. And custom 'heightxwidth' format (e.g., '1080x1920' means height=1080,
    width=1920) will be parsed into (width, height).
    """
    resolution_to_size = {
        '4k': (3840, 2160),
        '1080p': (1920, 1080),
        '720p': (1280, 720),
        '360p': (640, 360),
    }
    if image_resolution in resolution_to_size:
        return resolution_to_size[image_resolution]

    res = image_resolution.strip().lower()
    if 'x' in res:
        parts = res.split('x')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(f'Unsupported image resolution: {image_resolution}. '
                     "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920).")


def gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Generate a random prompt of specified token length using tokenizer
    vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    if image_pad_id:
        all_available_tokens.remove(image_pad_id)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def create_mm_data_row(text_prompt, images: list, images_base64, output_len, processor, backend):
    try:
        content_items = [{'type': 'image', 'image': {'url': image_base64}} for image_base64 in images_base64]
        content_items.append({'type': 'text', 'text': text_prompt})
        prompt_str = processor.apply_chat_template(
            [{
                'role': 'user',
                'content': content_items
            }],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        # Note (Xinyuan): This is a workaround for an issue where some tokenizers
        # do not support content as a list. (e.g. InternVL)
        print(f'Error applying chat template: {e}, fallback to <image> tag')
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        prompt_str = f'<image>{text_prompt}'

    # Calculate total tokens (text + vision)
    prompt_len = processor(
        text=[prompt_str],
        images=images,
        padding=False,
        return_tensors='pt',
    )['input_ids'].numel()

    # Calculate text-only tokens
    try:
        # Create text-only version of the prompt
        text_only_prompt = processor.apply_chat_template(
            [{
                'role': 'user',
                'content': text_prompt
            }],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors='pt',
        )['input_ids'].numel()
    except Exception:
        # Fallback: just tokenize the text prompt directly
        tokenizer_to_use = (processor.tokenizer if hasattr(processor, 'tokenizer') else processor)
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    # Vision tokens = total tokens - text tokens
    vision_prompt_len = prompt_len - text_prompt_len

    use_raw_prompt = backend in [
        'sglang',
        'sglang-oai',
        'sglang-oai-chat',
        'vllm',
        'vllm-chat',
        'lmdeploy',
        'lmdeploy-chat',
    ]
    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )


def sample_image_requests(
    num_requests: int,
    image_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    image_content: str,
    image_format: str,
    image_resolution: str,
    backend: str,
) -> List[DatasetRow]:
    """Generate requests with images.

    - Each request includes ``image_count`` images.
    - Supported resolutions: 4k (3840x2160), 1080p (1920x1080), 720p (1280x720), 360p (640x360),
      or custom 'heightxwidth' (e.g., 1080x1920).
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      only counts text tokens and excludes image data.
    """

    # Parse resolution (supports presets and 'heightxwidth')
    width, height = parse_image_resolution(image_resolution)

    # Check for potentially problematic combinations and warn user
    if width * height >= 1920 * 1080 and image_count * num_requests >= 100:
        warnings.warn(
            f'High resolution ({width}x{height}) with {image_count * num_requests} total images '
            f'may take a long time. Consider reducing resolution or image count.',
            UserWarning,
            stacklevel=2,
        )

    # Sample text lengths
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_requests,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_requests,
    )

    def _gen_random_image_data_uri(width: int = width, height: int = height) -> Tuple[Image.Image, str, int]:
        if image_content == 'blank':
            # Generate blank white image
            arr = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            # Generate random colored image
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format=image_format, quality=85)
        encoded = pybase64.b64encode(buf.getvalue()).decode('utf-8')
        image_data = f'data:image/{image_format};base64,{encoded}'  # noqa
        image_bytes = len(image_data.encode('utf-8'))
        return img, image_data, image_bytes

    dataset: List[DatasetRow] = []
    total_image_bytes = 0
    for i in range(num_requests):
        # Generate text prompt
        text_prompt = gen_mm_prompt(
            processor.tokenizer,
            processor.image_token_id if hasattr(processor, 'image_token_id') else None,
            int(input_lens[i]),
        )

        # Generate image list
        images, images_base64, images_bytes = zip(*[_gen_random_image_data_uri() for _ in range(image_count)])
        total_image_bytes += sum(list(images_bytes))

        data_row = create_mm_data_row(
            text_prompt,
            list(images),
            list(images_base64),
            int(output_lens[i]),
            processor,
            backend,
        )

        dataset.append(data_row)
    avg_image_bytes = total_image_bytes // num_requests if num_requests > 0 else 0

    print(f'#Input tokens: {np.sum([x.prompt_len for x in dataset])}')
    print(f'#Output tokens: {np.sum([x.output_len for x in dataset])}')
    print(f'\nCreated {len(dataset)} {image_content} {image_format} images \
            with average {avg_image_bytes} bytes per request')  # noqa
    return dataset


async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float('inf'):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[DatasetRow],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    total_input_text = 0
    total_input_vision = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(tokenizer.encode(outputs[i].generated_text, add_special_tokens=False))
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i].prompt_len
            total_input_text += input_requests[i].text_prompt_len
            total_input_vision += input_requests[i].vision_prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            'All requests failed. This is likely due to a misconfiguration '
            'on the benchmark arguments.',
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_input_text=total_input_text,
        total_input_vision=total_input_vision,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[DatasetRow],
    request_rate: float,
    disable_tqdm: bool,
    extra_request_body: Dict[str, Any],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f'Unknown backend: {backend}')

    if not args.disable_warmup:
        print('Starting initial single prompt test run...')
        start_warmup = time.perf_counter()
        test_request = input_requests[0]
        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_request.prompt,
            api_url=api_url,
            prompt_len=test_request.prompt_len,
            output_len=test_request.output_len,
            extra_request_body=extra_request_body,
            image_data=test_request.image_data,
        )
        test_output = await request_func(request_func_input=test_input)
        if not test_output.success:
            raise ValueError('Initial test run failed - Please make sure benchmark arguments '
                             f'are correctly specified. Error: {test_output.error}')
        else:
            print('Initial test run completed. Starting main benchmark run...')
        end_warmup = time.perf_counter()
        print(f'warmup time: {end_warmup - start_warmup:.2f}s')
        time.sleep(1.5)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            image_data=request.image_data,
            extra_request_body=extra_request_body,
        )
        tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print('\n{s:{c}^{n}}'.format(s=' Serving Benchmark Result ', n=50, c='='))
    print('{:<40} {:<10}'.format('Backend:', backend))
    print('{:<40} {:<10}'.format('Traffic request rate:', request_rate))
    print('{:<40} {:<10}'.format('Successful requests:', metrics.completed))
    print('{:<40} {:<10.2f}'.format('Benchmark duration (s):', benchmark_duration))
    print('{:<40} {:<10}'.format('Total input tokens:', metrics.total_input))
    print('{:<40} {:<10}'.format('Total input text tokens:', metrics.total_input_text))
    print('{:<40} {:<10}'.format('Total input vision tokens:', metrics.total_input_vision))
    print('{:<40} {:<10}'.format('Total generated tokens:', metrics.total_output))
    print('{:<40} {:<10}'.format('Total generated tokens (retokenized):', metrics.total_output_retokenized))
    print('{:<40} {:<10.2f}'.format('Request throughput (req/s):', metrics.request_throughput))
    print('{:<40} {:<10.2f}'.format('Input token throughput (tok/s):', metrics.input_throughput))
    print('{:<40} {:<10.2f}'.format('Output token throughput (tok/s):', metrics.output_throughput))
    print('{s:{c}^{n}}'.format(s='End-to-End Latency', n=50, c='-'))
    print('{:<40} {:<10.2f}'.format('Mean E2E Latency (ms):', metrics.mean_e2e_latency_ms))
    print('{:<40} {:<10.2f}'.format('Median E2E Latency (ms):', metrics.median_e2e_latency_ms))
    print('{s:{c}^{n}}'.format(s='Time to First Token', n=50, c='-'))
    print('{:<40} {:<10.2f}'.format('Mean TTFT (ms):', metrics.mean_ttft_ms))
    print('{:<40} {:<10.2f}'.format('Median TTFT (ms):', metrics.median_ttft_ms))
    print('{:<40} {:<10.2f}'.format('P99 TTFT (ms):', metrics.p99_ttft_ms))
    print('{s:{c}^{n}}'.format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
    print('{:<40} {:<10.2f}'.format('Mean TPOT (ms):', metrics.mean_tpot_ms))
    print('{:<40} {:<10.2f}'.format('Median TPOT (ms):', metrics.median_tpot_ms))
    print('{:<40} {:<10.2f}'.format('P99 TPOT (ms):', metrics.p99_tpot_ms))
    print('{s:{c}^{n}}'.format(s='Inter-token Latency', n=50, c='-'))
    print('{:<40} {:<10.2f}'.format('Mean ITL (ms):', metrics.mean_itl_ms))
    print('{:<40} {:<10.2f}'.format('Median ITL (ms):', metrics.median_itl_ms))
    print('{:<40} {:<10.2f}'.format('P99 ITL (ms):', metrics.p99_itl_ms))
    print('=' * 50)

    if (metrics.median_ttft_ms is not None and metrics.mean_itl_ms is not None
            and metrics.output_throughput is not None):
        FIELD_NAMES = [
            'backend', 'dataset_name', 'sharegpt_output_len', 'random_input_len', 'random_output_len',
            'random_range_ratio', 'request_rate', 'completed', 'total_input_tokens', 'total_output_tokens', 'duration',
            'request_throughput', 'input_throughput', 'output_throughput', 'mean_e2e_latency_ms', 'mean_ttft_ms',
            'mean_tpot_ms', 'mean_itl_ms'
        ]
        result = {
            'backend': args.backend,
            'dataset_name': args.dataset_name,
            'request_rate': request_rate,
            'total_input_tokens': metrics.total_input,
            'total_output_tokens': metrics.total_output,
            'mean_e2e_latency_ms': metrics.mean_e2e_latency_ms,
            'output_throughput': metrics.output_throughput,
            'sharegpt_output_len': args.sharegpt_output_len,
            'random_input_len': args.random_input_len,
            'random_output_len': args.random_output_len,
            'random_range_ratio': args.random_range_ratio,
            'duration': benchmark_duration,
            'completed': metrics.completed,
            'request_throughput': metrics.request_throughput,
            'input_throughput': metrics.input_throughput,
            'mean_ttft_ms': metrics.mean_ttft_ms,
            'mean_tpot_ms': metrics.mean_tpot_ms,
            'mean_itl_ms': metrics.mean_itl_ms,
        }
    else:
        print(f'Error running benchmark for request rate: {request_rate}')
        print('-' * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime('%m%d')
        if args.dataset_name == 'random':
            output_file_name = f'{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl'  # noqa
        else:
            output_file_name = f'{args.backend}_{now}_{args.num_prompts}_sharegpt.csv'  # noqa

    # Append results to a CSV file
    file_exists = os.path.isfile(output_file_name)
    with open(output_file_name, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    result = {
        'duration': benchmark_duration,
        'completed': metrics.completed,
        'total_input_tokens': metrics.total_input,
        'total_output_tokens': metrics.total_output,
        'total_output_tokens_retokenized': metrics.total_output_retokenized,
        'request_throughput': metrics.request_throughput,
        'input_throughput': metrics.input_throughput,
        'output_throughput': metrics.output_throughput,
        'mean_ttft_ms': metrics.mean_ttft_ms,
        'median_ttft_ms': metrics.median_ttft_ms,
        'std_ttft_ms': metrics.std_ttft_ms,
        'p99_ttft_ms': metrics.p99_ttft_ms,
        'mean_tpot_ms': metrics.mean_tpot_ms,
        'median_tpot_ms': metrics.median_tpot_ms,
        'std_tpot_ms': metrics.std_tpot_ms,
        'p99_tpot_ms': metrics.p99_tpot_ms,
        'mean_itl_ms': metrics.mean_itl_ms,
        'median_itl_ms': metrics.median_itl_ms,
        'std_itl_ms': metrics.std_itl_ms,
        'p99_itl_ms': metrics.p99_itl_ms,
        'input_lens': [output.prompt_len for output in outputs],
        'output_lens': output_lens,
        'ttfts': [output.ttft for output in outputs],
        'itls': [output.itl for output in outputs],
        'generated_texts': [output.generated_text for output in outputs],
        'errors': [output.error for output in outputs],
        'mean_e2e_latency_ms': metrics.mean_e2e_latency_ms,
        'median_e2e_latency_ms': metrics.median_e2e_latency_ms,
    }
    return result


def parse_request_rate_range(request_rate_range):
    if len(request_rate_range.split(',')) == 3:
        start, stop, step = map(int, request_rate_range.split(','))
        return list(range(start, stop, step))
    else:
        return list(map(int, request_rate_range.split(',')))


def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return 'chat_template' in tokenizer.init_kwargs
    except Exception as e:
        print(f'Fail to load tokenizer config with error={e}')
        return False


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set url
    if args.port is None:
        args.port = {
            'sglang': 30000,
            'sglang-native': 30000,
            'sglang-oai': 30000,
            'sglang-oai-chat': 30000,
            'lmdeploy': 23333,
            'lmdeploy-chat': 23333,
            'vllm': 8000,
            'vllm-chat': 8000,
            'trt': 8000,
            'gserver': 9988,
        }.get(args.backend, 30000)

    model_url = (f'{args.base_url}/v1/models' if args.base_url else f'http://{args.host}:{args.port}/v1/models')

    if args.backend in ['sglang', 'sglang-native']:
        api_url = (f'{args.base_url}/generate' if args.base_url else f'http://{args.host}:{args.port}/generate')
    elif args.backend in ['sglang-oai', 'vllm', 'lmdeploy']:
        api_url = (f'{args.base_url}/v1/completions'
                   if args.base_url else f'http://{args.host}:{args.port}/v1/completions')
    elif args.backend in ['lmdeploy-chat', 'vllm-chat', 'sglang-oai-chat']:
        api_url = (f'{args.base_url}/v1/chat/completions'
                   if args.base_url else f'http://{args.host}:{args.port}/v1/chat/completions')
    elif args.backend == 'trt':
        api_url = (
            f'{args.base_url}/v2/models/ensemble/generate_stream'
            if args.base_url else f'http://{args.host}:{args.port}/v2/models/ensemble/generate_stream'  # noqa
        )
        if args.model is None:
            print('Please provide a model using `--model` when using '
                  '`trt` backend.')
            sys.exit(1)
    elif args.backend == 'gserver':
        api_url = args.base_url if args.base_url else \
            f'{args.host}:{args.port}'
        args.model = args.model or 'default'

    # Get model name
    if args.model is None:
        try:
            response = requests.get(model_url)
            model_list = response.json().get('data', [])
            args.model = model_list[0]['id'] if model_list else None
        except Exception as e:
            print(f'Failed to fetch model from {model_url}. Error: {e}')
            print('Please specify the correct host and port using '
                  '`--host` and `--port`.')
            sys.exit(1)

    # Read dataset
    backend = args.backend
    model_id = args.model
    model_path = args.model_path if args.model_path is not None else args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model_path

    if args.model is None:
        print('No model specified or found. Please provide a model '
              'using `--model`.')
        sys.exit(1)

    if not check_chat_template(args.model_path):
        print('\nWARNING It is recommended to use the `Chat` or `Instruct` '
              'model for benchmarking.\n'
              'Because when the tokenizer counts the output tokens, if '
              'there is gibberish, it might count incorrectly.\n')

    print(f'{args}\n')

    tokenizer = get_tokenizer(tokenizer_id)

    if args.dataset_name == 'sharegpt':
        assert args.random_input_len is None and args.random_output_len is None
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
    elif args.dataset_name == 'random':
        assert args.random_input_len is not None and \
            args.random_output_len is not None
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_name == 'image':
        processor = get_processor(model_path)
        input_requests = sample_image_requests(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            processor=processor,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    if not args.multi:
        return asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                extra_request_body=extra_request_body,
            ))
    else:
        # Benchmark multiple rps.
        # TODO: use a fixed duration to compute num_prompts
        request_rates = parse_request_rate_range(args.request_rate_range)

        for rate in request_rates:
            asyncio.run(
                benchmark(
                    backend=backend,
                    api_url=api_url,
                    model_id=model_id,
                    tokenizer=tokenizer,
                    input_requests=input_requests,
                    request_rate=rate,
                    disable_tqdm=args.disable_tqdm,
                    extra_request_body=extra_request_body,
                ))


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f'Fail to set RLIMIT_NOFILE: {e}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Benchmark the online serving throughput.')
    parser.add_argument(
        '--backend',
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default='sglang',
        help='Must specify a backend, depending on the LLM Inference Engine.',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Server or API base url if not using http host and port.',
    )
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Default host is 0.0.0.0.')
    parser.add_argument(
        '--port',
        type=int,
        help='If not set, the default port is configured according to its '
        'default value for different LLM Inference Engines.',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='sharegpt',
        choices=['sharegpt', 'random', 'image'],
        help='Name of the dataset to benchmark on.',
    )
    parser.add_argument('--dataset-path', type=str, default='', help='Path to the dataset.')
    parser.add_argument(
        '--model',
        type=str,
        help='Name or path of the model. If not set, the default model will '
        'request /v1/models for conf.',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to the model. If not set, the default model will be model',
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Name or path of the tokenizer. If not set, using the model '
        'conf.',
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=1000,
        help='Number of prompts to process. Default is 1000.',
    )
    parser.add_argument(
        '--sharegpt-output-len',
        type=int,
        default=None,
        help='Output length for each request. Overrides the output length '
        'from the ShareGPT dataset.',
    )
    parser.add_argument(
        '--random-input-len',
        type=int,
        help='Number of input tokens per request, used only for random '
        'dataset.',
    )
    parser.add_argument(
        '--random-output-len',
        type=int,
        help='Number of output tokens per request, used only for random '
        'dataset.',
    )
    parser.add_argument(
        '--random-range-ratio',
        type=float,
        default=0.0,
        help='Range of sampled ratio of input/output length, '
        'used only for random dataset.',
    )
    # image dataset args
    parser.add_argument(
        '--image-count',
        type=int,
        default=1,
        help='Number of images per request (only available with the image dataset)',
    )
    parser.add_argument(
        '--image-resolution',
        type=str,
        default='1080p',
        help=('Resolution of images for image dataset. '
              "Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920)."),
    )
    parser.add_argument(
        '--image-format',
        type=str,
        default='jpeg',
        help=('Format of images for image dataset. '
              'Supports jpeg and png.'),
    )
    parser.add_argument(
        '--image-content',
        type=str,
        default='random',
        help=('Content for images for image dataset. '
              'Supports random and blank.'),
    )
    parser.add_argument(
        '--request-rate',
        type=float,
        default=float('inf'),
        help='Number of requests per second. If this is inf, then all the '
        'requests are sent at time 0. Otherwise, we use Poisson process to '
        'synthesize the request arrival times. Default is inf.',
    )
    parser.add_argument('--seed', type=int, default=1, help='The random seed.')
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Use request rate range rather than single value.',
    )
    parser.add_argument(
        '--request-rate-range',
        type=str,
        default='2,34,2',
        help='Range of request rates in the format start,stop,step. Default '
        'is 2,34,2. It also supports a list of request rates, requiring '
        'the parameters to not equal three.',
    )
    parser.add_argument('--output-file', type=str, help='Output JSONL file name.')
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Specify to disable tqdm progress bar.',
    )
    parser.add_argument(
        '--disable-stream',
        action='store_true',
        help='Disable streaming mode.',
    )
    parser.add_argument(
        '--disable-ignore-eos',
        action='store_true',
        help='Disable ignoring EOS.',
    )
    parser.add_argument(
        '--extra-request-body',
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help='Append given JSON object to the request payload. You can use '
        'this to specify additional generate params like sampling params.',
    )
    parser.add_argument(
        '--disable-warmup',
        action='store_true',
        default=None,
        help='Disable a warmup request before the benchmark. ',
    )
    args = parser.parse_args()
    run_benchmark(args)
