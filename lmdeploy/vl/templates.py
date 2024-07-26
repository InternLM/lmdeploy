# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import PIL
import PIL.Image

from lmdeploy.model import BaseModel
from lmdeploy.utils import get_hf_config_content, get_logger
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64, load_image

logger = get_logger('lmdeploy')

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
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if isinstance(image, str):
                    image_base64_data = encode_image_base64(image)
                    if image_base64_data == '':
                        logger.error(f'failed to load file {image}')
                        continue
                    item = {
                        'type': 'image_url',
                        'image_url': {
                            'url':
                            f'data:image/jpeg;base64,{image_base64_data}'
                        }
                    }
                elif isinstance(image, PIL.Image.Image):
                    item = {
                        'type': 'image_data',
                        'image_data': {
                            'data': image
                        }
                    }
                else:
                    raise ValueError(
                        'image should be a str(url/path) or PIL.Image.Image')

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
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if item['type'] == 'image_url':
                    url = item['image_url']['url']
                    images.append(url)
                elif item['type'] == 'image_data':
                    data = item['image_data']['data']
                    images.append(data)

        def _inner_call(i, images):
            url_or_data = images[i]
            images[i] = load_image(url_or_data)

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(
                None, _inner_call, i, images) for i in range(len(images))
        ])

        return images

    def append_image_token(self, prompt, num_images: int):
        """append image token to user prompt."""
        return (IMAGE_TOKEN + '\n') * num_images + prompt

    def convert_messages(self, messages, sequence_start=True):
        """convert GPT4V message format to GPT4 text format."""
        new_messages = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                if isinstance(content, list):
                    text = content[0]['text']
                    message = {'role': role, 'content': text}
                new_messages.append(message)
                continue
            num_images = 0
            for item in content:
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if item['type'] == 'image_url':
                    num_images += 1
                elif item['type'] == 'image_data':
                    num_images += 1
                elif item['type'] == 'text':
                    prompt = item['text']
            # if IMAGE_TOKEN in user prompt, use user custom prompt instead
            # of adding IMAGE_TOKEN to user prompt
            if IMAGE_TOKEN not in prompt and num_images > 0:
                prompt = self.append_image_token(prompt, num_images)
            new_item = {'role': 'user', 'content': prompt}
            new_messages.append(new_item)
        return new_messages

    def messages2prompt(self, messages, sequence_start=True, **kwargs) -> str:
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
        return f'<img>{IMAGE_TOKEN * num_images}</img>\n' + prompt


class DeepSeekVLChatTemplateWrapper(VLChatTemplateWrapper):
    """DeepSeek vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        logger.error(
            f'for deepseek-vl model, the user should insert the {IMAGE_TOKEN} '
            'to user prompt manually, please read https://lmdeploy.readthedocs'
            '.io/en/latest/inference/vl_pipeline.html for more details.')
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


class CogVLMChatTemplateWrapper(VLChatTemplateWrapper):
    """cogvlm chat template wrapper."""

    def __init__(self, chat_template: BaseModel):
        from lmdeploy.model import Vicuna
        self.chat_template = chat_template
        self.llm_chat_template = Vicuna(eoa=chat_template.eoa,
                                        stop_words=chat_template.stop_words)

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
                elif item['type'] == 'image_data':
                    num_images += 1
                elif item['type'] == 'text':
                    prompt = item['text']

            new_item = {
                'role': 'user',
                'content': prompt,
                'num_images': num_images
            }
            new_messages.append(new_item)
        return new_messages

    def messages2prompt(self, messages, sequence_start=True, **kwargs) -> str:
        """convert messages to decorated prompt."""
        if isinstance(messages, str):
            return self.chat_template.messages2prompt(messages, sequence_start)
        new_messages = self.convert_messages(messages, sequence_start)
        prompt = ''
        for i, msg in enumerate(new_messages):
            num_images = msg.pop('num_images', 0)
            if num_images == 0:
                role = msg['role']
                msg = self.llm_chat_template.messages2prompt([msg],
                                                             sequence_start
                                                             and i == 0)
                msg = dict(role=role, content=msg)
            prompt_i = self.chat_template.messages2prompt([msg], sequence_start
                                                          and i == 0)
            if num_images > 0:
                prompt_i = (IMAGE_TOKEN * num_images) + prompt_i
            prompt += prompt_i
        return prompt


class InternLMXComposer2TemplateWrapper(VLChatTemplateWrapper):
    """InternLM-XComposer2 chat template."""

    def append_image_token(self, prompt, num_images: int):
        logger.warning(f'auto append {IMAGE_TOKEN} at the beginning, '
                       'the user can manually insert the token to prompt')
        return ' '.join([IMAGE_TOKEN] * num_images) + prompt


class MiniGeminiLlamaTempateWrapper(VLChatTemplateWrapper):
    """Qwen vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        if num_images == 0:
            return prompt
        res = f'{IMAGE_TOKEN}\n'
        assert num_images <= 1, 'MiniGeminiLlama accepts 1 input image'
        res = res + prompt
        return res


class MiniCPMVTempateWrapper(VLChatTemplateWrapper):
    """MiniCPMV chat template."""

    def append_image_token(self, prompt, num_images: int):
        return f'<image>{IMAGE_TOKEN}</image>\n' * num_images + prompt

    def update_image_token(self, prompt, features):
        _features = []
        _prompt = []
        segs = prompt.split(f'<image>{IMAGE_TOKEN}</image>\n')
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                _feat = features[i - 1]['embeddings'].split(1)
                _feat = [x.squeeze() for x in _feat]
                _features.extend(_feat)
                _seg = f'<image>{IMAGE_TOKEN}</image>'
                if len(_feat) > 1:
                    grid = features[i - 1]['grid']
                    if grid is not None:
                        _slice = '\n'.join(
                            [f'<image>{IMAGE_TOKEN}</image>' * grid[0]] *
                            grid[1])
                        _seg = f'{_seg}<slice>{_slice}</slice>\n'
                _prompt.append(_seg)
            _prompt.append(seg)
        _prompt = ''.join(_prompt)
        return _prompt, _features


class GLM4VChatTemplateWrapper(VLChatTemplateWrapper):
    """glm-4v chat template."""
    pass


def get_vl_prompt_template(model_path: str, chat_template: BaseModel,
                           model_name: str) -> VLChatTemplateWrapper:
    """get vision language prompt template."""
    assert type(chat_template) != type(BaseModel()), 'failed to match ' \
        'chat template, please explicit set chat_template_config' # noqa E721
    if model_name == 'yi-vl':
        return YiVLChatTemplateWrapper(chat_template)

    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVLChatTemplateWrapper(chat_template)
    elif arch in [
            'LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM',
            'LlavaForConditionalGeneration',
            'LlavaNextForConditionalGeneration', 'Phi3VForCausalLM'
    ]:
        return LlavaVLChatTemplateWrapper(chat_template)
    elif arch == 'MultiModalityCausalLM':  # deepseek-vl
        return DeepSeekVLChatTemplateWrapper(chat_template)
    elif arch == 'CogVLMForCausalLM':
        return CogVLMChatTemplateWrapper(chat_template)
    elif arch in ['InternLMXComposer2ForCausalLM', 'InternLM2ForCausalLM']:
        return InternLMXComposer2TemplateWrapper(chat_template)
    elif arch == 'InternVLChatModel':
        return InternVLChatTemplateWrapper(chat_template)
    elif arch in ['MiniGeminiLlamaForCausalLM', 'MGMLlamaForCausalLM']:
        return MiniGeminiLlamaTempateWrapper(chat_template)
    elif arch == 'MiniCPMV':
        return MiniCPMVTempateWrapper(chat_template)
    elif arch == 'ChatGLMModel':
        return GLM4VChatTemplateWrapper(chat_template)
    raise ValueError(f'unsupported vl_prompt_template with arch {arch}')
