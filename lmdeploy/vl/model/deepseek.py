# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def check_deepseek_vl_install():
    """Check deepseek_vl install."""
    try:
        import deepseek_vl  # noqa: F401
    except ImportError:
        raise ImportError('To use DeepSeekVLModel, please install deepseek_vl by '
                          '`pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git'
                          ' --no-deps`')


@VISION_MODELS.register_module()
class DeepSeekVisionModel(VisionModel):
    """Qwen vision model."""

    _arch = 'MultiModalityCausalLM'

    def build_preprocessor(self):
        check_deepseek_vl_install()
        from deepseek_vl.models import VLChatProcessor
        vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
        tokenizer = vl_chat_processor.tokenizer
        self.image_token_id = tokenizer.vocab.get(vl_chat_processor.image_tag)
        self.image_processor = vl_chat_processor.image_processor

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.vl_model = model
            if not self.with_llm:
                del model.language_model

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['Block'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['Block'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        same_device_keys = [('vision_model.vision_tower_high.vision_tower.pos_embed',
                             'vision_model.vision_tower_high.vision_tower.patch_embed'),
                            ('vision_model.vision_tower_low.vision_tower.pos_embed',
                             'vision_model.vision_tower_low.vision_tower.patch_embed')]
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
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map=device_map if not self.with_llm else {'': 'cpu'},
                                         dtype=torch.half)

        self.model = model.eval()
        self.vision_model = model.vision_model.eval()
        self.aligner = model.aligner.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to the spec of `super.preprocess()"""
        images = self.collect_images(messages)
        outputs = []
        for image, _ in images:
            image = image.convert('RGB')
            pixel_values = self.image_processor([image], return_tensors='pt').pixel_values
            outputs.append(
                dict(
                    pixel_values=pixel_values,
                    image_size=image.size,
                    # refer to https://github.com/deepseek-ai/DeepSeek-VL/blob/main/deepseek_vl/models/processing_vlm.py  # noqa
                    # which is hardcoded 576
                    image_tokens=576,
                    image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(device=next(self.vision_model.parameters()).device, dtype=torch.float16)
            # [b x n_images, T2, D]
            logger.info(f'vision forward shape: {pixel_values.shape}')
            feats = self.aligner(self.vision_model(pixel_values))
            feats = torch.split(feats, 1, dim=0)
            outputs.extend([x.squeeze() for x in feats])
        messages.append(dict(role='forward', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
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

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
