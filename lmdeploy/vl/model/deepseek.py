# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import numpy as np
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
            'pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git'
            ' --no-deps')


@VISION_MODELS.register_module()
class DeepSeekVisionModel(VisonModel):
    """Qwen vision model."""

    _arch = 'MultiModalityCausalLM'

    def build_preprocessor(self):
        check_deepseek_vl_install()
        # empty init
        from accelerate import init_empty_weights
        from deepseek_vl.models import VLChatProcessor
        with init_empty_weights():
            warnings.simplefilter('ignore')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            if not self.with_llm:
                del self.model.language_model
            else:
                self.vl_model = self.model
        self.image_processor = VLChatProcessor.from_pretrained(
            self.model_path).image_processor

    def build_model(self):
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(self.model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['Block'])
        device_map = infer_auto_device_map(self.model,
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
                model=self.model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                dtype=torch.half)

        self.vision_model = self.model.vision_model.eval()
        self.aligner = self.model.aligner.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """get images and their corresponding preprocess parameters from
        messages, and perform preprocessing."""
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                pixel_values = self.image_processor(
                    image, return_tensors='pt').pixel_values
                outputs.append(dict(pixel_values=pixel_values))
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        """forward."""
        assert all(x.get('pixel_values') is not None for x in inputs)
        pixel_values = [x['pixel_values'] for x in inputs]
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(device=next(
            self.vision_model.parameters()).device,
                                       dtype=torch.float16)
        # [b x n_images, T2, D]
        images_embeds = self.aligner(self.vision_model(pixel_values))
        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
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

        # collect all preprocessing result from messages
        preps = [
            message.pop('preprocess') for message in messages
            if 'preprocess' in message.keys()
        ]
        segs = prompt.split(IMAGE_TOKEN)
        # flatten the list
        preps = list(itertools.chain(*preps))
        assert len(segs) == len(preps) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(preps)}')

        return prompt, segs, preps

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, preps = self.proc_messages(messages, chat_template,
                                                 sequence_start)

        # calculate the image token offset for each image
        input_ids = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = 0  # TODO
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_tokens)
            token_ids = tokenizer.encode(seg,
                                         add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, features = self.proc_messages(messages, chat_template,
                                                    sequence_start)
        features = [x.cpu().numpy() for x in features]

        # tokenizer prompt, and get input_embeddings and input_embedding_ranges
        input_ids = []
        begins = []
        ends = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
            seg_ids = tokenizer.encode(seg,
                                       add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()
        return dict(prompt=prompt,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges)
