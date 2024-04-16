# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import PIL

from lmdeploy.model import BaseModel
from lmdeploy.utils import get_hf_config_content
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64, load_image

VLPromptType = Union[str, Tuple[str, PIL.Image.Image],
                     Tuple[str, List[PIL.Image.Image]]]


class VLChatTemplateWrapper:
    """vl chat template wrapper."""

    def __init__(self, chat_template: BaseModel):
        self.chat_template = chat_template

    def prompt_to_messages(self, prompt: VLPromptType):
        """convert prompt to GTP4V format."""
        messages = {
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': '',
            }]
        }
        if isinstance(prompt, str):
            messages['content'][0]['text'] = prompt
        else:
            prompt, images = prompt
            if not isinstance(images, list):
                images = [images]
            messages['content'][0]['text'] = prompt
            for image in images:
                if isinstance(image, str):
                    image = load_image(image)
                image_base64_data = encode_image_base64(image)
                item = {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image_base64_data}'
                    }
                }
                messages['content'].append(item)

        return [messages]

    async def async_collect_pil_images(
            self, messages: Dict) -> List[PIL.Image.Image]:
        """collect image from messages."""
        images = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                continue
            for item in content:
                if item['type'] != 'image_url':
                    continue
                url = item['image_url']['url']
                images.append(url)

        def _inner_call(i, images):
            url = images[i]
            images[i] = load_image(url)

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(
                None, _inner_call, i, images) for i in range(len(images))
        ])

        return images

    def append_image_token(self, prompt, num_images: int):
        """append image token to user prompt."""
        return IMAGE_TOKEN * num_images + '\n' + prompt

    def convert_messages(self, messages, sequence_start=True):
        """convert GPT4V message format to GPT4 text format."""
        new_messages = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                new_messages.append(message)
                continue
            num_images = 0
            for item in content:
                if item['type'] == 'image_url':
                    num_images += 1
                elif item['type'] == 'text':
                    prompt = item['text']
            new_item = {
                'role': 'user',
                'content': self.append_image_token(prompt, num_images)
            }
            new_messages.append(new_item)
        return new_messages

    def messages2prompt(self, messages, sequence_start=True) -> str:
        """convert messages to decorated prompt."""
        if isinstance(messages, str):
            return self.chat_template.messages2prompt(messages, sequence_start)
        new_messages = self.convert_messages(messages, sequence_start)
        return self.chat_template.messages2prompt(new_messages, sequence_start)


class LlavaVLChatTemplateWrapper(VLChatTemplateWrapper):
    """Llava vl chat template."""
    pass


class YiVLChatTemplateWrapper(VLChatTemplateWrapper):
    """Yi vl chat template."""
    pass


class InternVLChatTemplateWrapper(VLChatTemplateWrapper):
    """InternVL chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        # not sure whether support multi images.
        return f'<img>{IMAGE_TOKEN}</img>\n' * num_images + prompt


class DeepSeekVLChatTemplateWrapper(VLChatTemplateWrapper):
    """DeepSeek vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        if num_images == 1:
            return f'{IMAGE_TOKEN}{prompt}'
        res = ''
        for i in range(num_images):
            res += f'{IMAGE_TOKEN} is Figure {str(i)}.\n'
        res = res + prompt
        return res


class QwenVLChatTemplateWrapper(VLChatTemplateWrapper):
    """Qwen vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        res = ''
        for i in range(num_images):
            res += f'Picture {str(i)}:{IMAGE_TOKEN}\n'
        res = res + prompt
        return res


class MiniGeminiLlamaTempateWrapper(VLChatTemplateWrapper):
    """Qwen vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        res = f'{IMAGE_TOKEN}\n'
        assert num_images <= 1, 'MiniGeminiLlama accepts 1 input image'
        res = res + prompt
        return res


def get_vl_prompt_template(model_path: str, chat_template: BaseModel,
                           model_name: str) -> VLChatTemplateWrapper:
    """get vision language prompt template."""
    if model_name == 'yi-vl':
        return YiVLChatTemplateWrapper(chat_template)

    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVLChatTemplateWrapper(chat_template)
    elif arch == 'LlavaLlamaForCausalLM':
        return LlavaVLChatTemplateWrapper(chat_template)
    elif arch == 'MultiModalityCausalLM':  # deepseek-vl
        return DeepSeekVLChatTemplateWrapper(chat_template)
    elif arch == 'InternVLChatModel':
        return InternVLChatTemplateWrapper(chat_template)
    elif arch == 'MiniGeminiLlamaForCausalLM':
        return MiniGeminiLlamaTempateWrapper(chat_template)
    raise ValueError(f'unsupported vl_prompt_template with arch {arch}')
