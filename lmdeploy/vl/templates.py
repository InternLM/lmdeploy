# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple

import PIL

from lmdeploy.model import BaseModel
from lmdeploy.turbomind.utils import get_hf_config_content
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64, load_image_from_url

VLPromptType = Tuple[str, List[PIL.Image.Image]]


class VLChatTemplateWrapper:

    def __init__(self, chat_template: BaseModel):
        self.chat_template = chat_template

    def prompt_to_messages(self, prompt: VLPromptType):
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
            messages['content'][0]['text'] = prompt
            for image in images:
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
            images[i] = load_image_from_url(url)

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(
                None, _inner_call, i, images) for i in range(len(images))
        ])

        return images

    def messages2prompt(self, messages, sequence_start=True):
        raise NotImplementedError()


class LlavaVLChatTemplateWrapper(VLChatTemplateWrapper):

    def append_image_token(self, prompt, num_images: int):
        return IMAGE_TOKEN * num_images + prompt

    def messages2prompt(self, messages, sequence_start=True):
        new_messages = []
        num_images = 0
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                new_messages.append(message)
            for item in content:
                if item['type'] == 'image_url':
                    num_images += 1
                elif item['type'] == 'text':
                    prompt = item['text']
            new_item = {'role': 'user', 'content': prompt}
            new_messages.append(new_item)

        return IMAGE_TOKEN * num_images + self.chat_template.messages2prompt(
            new_messages, sequence_start)


class QwenVLChatTemplateWrapper(VLChatTemplateWrapper):

    def append_image_token(self, prompt, num_images: int):
        res = ''
        for i in range(num_images):
            res += f'Picture {str(i)}:{IMAGE_TOKEN}\n'
        res = res + prompt
        return res

    def messages2prompt(self, messages, sequence_start=True):
        new_messages = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                new_messages.append(message)
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

        return self.chat_template.messages2prompt(new_messages, sequence_start)


def get_vl_prompt_template(model_path: str,
                           chat_template: BaseModel) -> VLChatTemplateWrapper:

    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVLChatTemplateWrapper(chat_template)
    elif arch == 'LlavaLlamaForCausalLM':
        return LlavaVLChatTemplateWrapper(chat_template)
    raise ValueError(f'unsupported vl_prompt_template with arch {arch}')
