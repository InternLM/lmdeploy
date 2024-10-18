# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import PIL
import PIL.Image

from lmdeploy.archs import get_model_arch
from lmdeploy.model import BaseModel
from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import load_image

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
                    image = load_image(image)
                    item = {
                        'type': 'image_data',
                        'image_data': {
                            'data': image
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
            self, messages: Dict) -> List[Tuple[PIL.Image.Image, Dict]]:
        """collect image from messages."""
        images_with_kwargs = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role != 'user' or isinstance(content, str):
                continue
            for item in content:
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if item['type'] == 'image_url':
                    item_copy = item['image_url'].copy()
                    try:
                        url = item_copy.pop('url')
                        images_with_kwargs.append([url, item_copy])
                    except KeyError:
                        logger.error(f'invalid format {message}')
                elif item['type'] == 'image_data':
                    item_copy = item['image_data'].copy()
                    try:
                        data = item_copy.pop('data')
                        images_with_kwargs.append([data, item_copy])
                    except KeyError:
                        logger.error(f'invalid format {message}')

        def _inner_call(i, images):
            url_or_data = images[i][0]
            images[i][0] = load_image(url_or_data)

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(None, _inner_call, i,
                                                     images_with_kwargs)
            for i in range(len(images_with_kwargs))
        ])

        return images_with_kwargs

    def append_image_token(self, prompt, num_images: int):
        """append image token to user prompt."""
        if IMAGE_TOKEN in prompt:
            return prompt
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
            if num_images > 0:
                # add IMAGE_TOKEN to user prompt
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
        # lmdeploy uses <IMAGET_TOKEN> as image token
        # internvl uses special tags
        if IMAGE_TOKEN in prompt and f'<img>{IMAGE_TOKEN}' not in prompt:
            prompt = prompt.replace(f'{IMAGE_TOKEN}',
                                    f'<img>{IMAGE_TOKEN}</img>')
            prompt = prompt.replace('</img><img>', '')
            prompt = prompt.replace('<img><img>', '<img>')
            prompt = prompt.replace('</img></img>', '</img>')
        elif IMAGE_TOKEN not in prompt:
            prompt = f'<img>{IMAGE_TOKEN * num_images}</img>\n' + prompt
        return prompt


class DeepSeekVLChatTemplateWrapper(VLChatTemplateWrapper):
    """DeepSeek vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        if IMAGE_TOKEN in prompt:
            return prompt
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
        if IMAGE_TOKEN in prompt:
            return prompt
        res = ''
        for i in range(num_images):
            res += f'Picture {str(i)}:{IMAGE_TOKEN}\n'
        res = res + prompt
        return res


class Qwen2VLChatTemplateWrapper(VLChatTemplateWrapper):
    """qwen2 vl."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        if IMAGE_TOKEN in prompt and '<|vision_start|>' not in prompt:
            prompt = prompt.replace(
                IMAGE_TOKEN, f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>')
        else:
            # Qwen2-VL-2B-Instruct will concat image and user prompt according
            #   to their order in the content list
            # we insert image token before user prompt by default. The user can
            #   use custom image token position if they want the same decorated
            #   prompt as Qwen2-VL
            prompt = f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>' * \
                num_images + prompt
        return prompt

    def get_mrope_info(self,
                       seq_len: int,
                       grid_thws: List[Tuple[int, int, int]] = None,
                       embedding_ranges: List[Tuple[int, int]] = None):
        import torch
        if grid_thws is None:
            mrope_position_ids = torch.arange(seq_len).expand(3, -1)
            mrope_position_delta = torch.tensor([0], dtype=torch.long)
        else:
            mrope_position_ids = [
                torch.arange(embedding_ranges[0][0]).expand(3, -1)
            ]
            st_idx = embedding_ranges[0][0]
            for i, (grid_thw, embedding_range) in enumerate(
                    zip(grid_thws, embedding_ranges)):
                llm_grid_t, llm_grid_h, llm_grid_w = grid_thw
                llm_grid_h //= 2
                llm_grid_w //= 2
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()
                mrope_position_ids.append(
                    torch.stack([t_index, h_index, w_index]) + st_idx)
                st_idx += max(llm_grid_h, llm_grid_w)
                if i < len(embedding_ranges) - 1:
                    text_len = embedding_ranges[i +
                                                1][0] - embedding_ranges[i][1]
                else:
                    text_len = seq_len - embedding_range[1]
                mrope_position_ids.append(
                    torch.arange(text_len).expand(3, -1) + st_idx)
                st_idx += text_len
            mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
            mrope_position_delta = torch.tensor([st_idx - seq_len],
                                                dtype=torch.long)

        return mrope_position_ids, mrope_position_delta


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
        if IMAGE_TOKEN in prompt:
            return prompt
        logger.warning(f'auto append {IMAGE_TOKEN} at the beginning, '
                       'the user can manually insert the token to prompt')
        return ' '.join([IMAGE_TOKEN] * num_images) + prompt


class MiniGeminiLlamaTempateWrapper(VLChatTemplateWrapper):
    """Qwen vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        if num_images == 0:
            return prompt
        if IMAGE_TOKEN in prompt:
            return prompt
        res = f'{IMAGE_TOKEN}\n'
        assert num_images <= 1, 'MiniGeminiLlama accepts 1 input image'
        res = res + prompt
        return res


class MiniCPMVTempateWrapper(VLChatTemplateWrapper):
    """MiniCPM-Llama3-V-2_5 chat template."""

    def append_image_token(self, prompt, num_images: int):
        if IMAGE_TOKEN in prompt:
            return prompt
        prompt = f'{IMAGE_TOKEN}\n' * num_images + prompt
        return prompt

    def update_image_token(self, prompt, features):
        _features = []
        _prompt = []
        segs = prompt.split(f'{IMAGE_TOKEN}\n')
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


class MiniCPMV26TempateWrapper(MiniCPMVTempateWrapper):
    """MiniCPM-V-2_6 chat template."""

    def update_image_token(self, prompt, features):
        _features = []
        _prompt = []
        segs = prompt.split(f'{IMAGE_TOKEN}\n')
        idx = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                _feat = features[i - 1]['embeddings'].split(1)
                _feat = [x.squeeze() for x in _feat]
                _features.extend(_feat)
                _seg = f'<image>{IMAGE_TOKEN}</image>'
                if features[i - 1].get('use_image_id', False):
                    _seg = f'<image_id>{idx}</image_id>' + _seg
                    idx += 1
                if len(_feat) > 1:
                    grid = features[i - 1]['grid']
                    if grid is not None:
                        _slice = '\n'.join(
                            [f'<slice>{IMAGE_TOKEN}</slice>' * grid[0]] *
                            grid[1])
                        _seg = _seg + _slice
                _seg += '\n'
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
        'chat template, please explicit set chat_template_config'  # noqa E721
    if model_name == 'yi-vl':
        return YiVLChatTemplateWrapper(chat_template)
    arch, cfg = get_model_arch(model_path)
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
        version_map = {
            '2.5': MiniCPMVTempateWrapper,
            '2.6': MiniCPMV26TempateWrapper
        }
        version = str(getattr(cfg, 'version', '2.5'))
        return version_map[version](chat_template)
    elif arch == 'ChatGLMModel':
        return GLM4VChatTemplateWrapper(chat_template)
    elif arch == 'Qwen2VLForConditionalGeneration':
        return Qwen2VLChatTemplateWrapper(chat_template)
    raise ValueError(f'unsupported vl_prompt_template with arch {arch}')
