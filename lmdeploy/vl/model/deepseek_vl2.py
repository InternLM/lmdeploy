# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import redirect_stdout
from typing import Dict, List

import torch
from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel

logger = get_logger('lmdeploy')


def check_deepseek_vl2_install():
    """check deepseek_vl install."""
    try:
        import deepseek_vl2  # noqa: F401
    except ImportError:
        raise ImportError('To use DeepSeek-VL2, please install deepseek_vl2 by '
                          '`pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git'
                          ' --no-deps`')


@VISION_MODELS.register_module()
class DeepSeek2VisionModel(VisonModel):
    """DeepSeek2 vision model."""

    _arch = 'DeepseekV2ForCausalLM'

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        if hasattr(config, 'language_config') and hasattr(config, 'vision_config'):
            arch = config.language_config.get('architectures', [None])[0]
            return arch == cls._arch
        return False

    def build_preprocessor(self):
        check_deepseek_vl2_install()
        from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor

        # suppress deepseek-vl2 processor initialization print logs
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                self.image_processor = DeepseekVLV2Processor.from_pretrained(self.model_path,
                                                                             image_token='<IMAGE_TOKEN>')

    def build_model(self):
        """build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        # TODO, implement for tubomind engine
        raise NotImplementedError()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to the spec of `super.preprocess()"""
        images = self.collect_images(messages)

        # convert to upstream api formats
        images = [img_parameter[0] for img_parameter in images]
        formatted_messages = []
        for msg in messages:
            crt_role = msg['role']

            text_content = ''
            image_content = []
            for ele in msg['content']:
                if ele['type'] == 'text':
                    text_content = ele['text']
                elif ele['type'] == 'image':
                    image_content.append(ele['image'])

            formatted_messages.append({'role': crt_role, 'content': text_content, 'images': image_content})

        # NOTE: DeepseekVLV2Processor inputs
        # conversations (List[Dict]): conversations with a list of messages;
        # images (List[ImageType]): the list of images;
        # force_batchify (bool): force batchify the inputs;
        # inference_mode (bool): if True, then remove the last eos token;
        prepare = self.image_processor(conversations=formatted_messages,
                                       images=images,
                                       force_batchify=False,
                                       inference_mode=False)

        messages.append(
            dict(role='preprocess',
                 content=[
                     dict(pixel_values=prepare.images,
                          image_tokens=prepare.num_image_tokens[0],
                          image_token_id=0,
                          image_size=self.image_processor.image_size,
                          images_spatial_crop=prepare.images_spatial_crop,
                          images_seq_mask=prepare.images_seq_mask)
                 ]))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        # TODO, implement for turbomind engine
        raise NotImplementedError()

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [x['text'] for x in message['content'] if x['type'] == 'text']
            content = content[0]
            n_image = sum([1 for x in message['content'] if x['type'] == 'image'])
            n_placeholder = content.count(IMAGE_TOKEN)
            if n_placeholder == 0:
                logger.warning(f"""for deepseek-vl model, the user should insert the {IMAGE_TOKEN}
                    to user prompt manually, please read https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html
                    for more details.""")  # noqa
            if n_placeholder != 0 and n_placeholder != n_image:
                logger.error(f'unmatched placeholder and image: {n_placeholder} vs '
                             f'{n_image}. Ignore the placeholder')
                content = content.replace(IMAGE_TOKEN, '')
                n_placeholder = 0
            if n_placeholder == 0:
                if n_image == 1:
                    content = f'{IMAGE_TOKEN}{content}'
                else:
                    content = ''.join([f'{IMAGE_TOKEN} is Figure {str(i)}.\n' for i in range(n_image)]) + content
            prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
