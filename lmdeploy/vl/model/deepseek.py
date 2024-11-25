# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def check_deepseek_vl_install():
    """check deepseek_vl install."""
    try:
        import deepseek_vl  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use DeepSeekVLModel, please install deepseek_vl by '
            '`pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git'
            ' --no-deps`')


@VISION_MODELS.register_module()
class DeepSeekVisionModel(VisonModel):
    """Qwen vision model."""

    _arch = 'MultiModalityCausalLM'

    def build_preprocessor(self):
        check_deepseek_vl_install()
        from deepseek_vl.models import VLChatProcessor
        self.image_processor = VLChatProcessor.from_pretrained(
            self.model_path).image_processor

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['Block'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['Block'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        same_device_keys = [
            ('vision_model.vision_tower_high.vision_tower.pos_embed',
             'vision_model.vision_tower_high.vision_tower.patch_embed'),
            ('vision_model.vision_tower_low.vision_tower.pos_embed',
             'vision_model.vision_tower_low.vision_tower.patch_embed')
        ]
        for (a, b) in same_device_keys:
            if a in device_map and b in device_map:
                device_map[b] = device_map[a]
        downsamples = []
        ka = 'vision_model.vision_tower_high.vision_tower.downsamples'
        kb = 'vision_model.vision_tower_high.vision_tower.hd_alpha_downsamples'  # noqa: E501
        for k in device_map:
            if k.startswith(ka):
                downsamples.append(k)
        if len(downsamples) == 1:
            device_map[ka] = device_map[kb]
        elif len(downsamples) > 1:
            numbers = [int(x[len(ka) + 1:]) for x in downsamples]
            device_map[f'{ka}.{numbers[-1]}'] = device_map[kb]

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                dtype=torch.half)

        self.vision_model = model.vision_model.eval()
        self.aligner = model.aligner.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to the spec of `super.preprocess()"""
        images = super().collect_images(messages)
        outputs = []
        for image, _ in images:
            image = image.convert('RGB')
            pixel_values = self.image_processor(
                [image], return_tensors='pt').pixel_values
            outputs.append(
                dict(
                    pixel_values=pixel_values,
                    image_size=image.size,
                    image_tokens=576,  # TODO
                    image_token_id=0))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict]) -> List[Dict]:
        """extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        pixel_values = [x['pixel_values'] for x in inputs]
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(device=next(
            self.vision_model.parameters()).device,
                                       dtype=torch.float16)
        # [b x n_images, T2, D]
        images_embeds = self.aligner(self.vision_model(pixel_values))
        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        messages.append(dict(role='forward', content=outputs))
        return messages

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [
                x['text'] for x in message['content'] if x['type'] == 'text'
            ]
            content = content[0]
            if IMAGE_TOKEN not in content:
                logger.warning(
                    f"""for deepseek-vl model, the user should insert the {IMAGE_TOKEN}
                    to user prompt manually, please read https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html
                    for more details.""")  # noqa
                n_images = len(
                    [1 for x in message['content'] if x['type'] == 'image'])
                if n_images == 1:
                    content = f'{IMAGE_TOKEN}{content}'
                else:
                    content = ''.join([
                        f'{IMAGE_TOKEN} is Figure {str(i)}.\n'
                        for i in range(n_images)
                    ]) + content
            else:
                logger.error('TODO deepseek-vl')
            prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_turbomind_aux(messages, prompt, IMAGE_TOKEN,
                                        tokenizer, sequence_start)
