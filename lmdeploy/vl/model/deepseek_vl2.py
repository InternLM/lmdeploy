# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/deepseek.py

import warnings
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class DeepSeekVisionModel2(VisonModel):
    """DeepSeek vision model."""

    _arch = 'MultiModalityCausalLM'

    def build_preprocessor(self):
        from lmdeploy.vl.model.deepseek_vl2_processors import DeepseekVLV2Processor
        self.image_processor = DeepseekVLV2Processor.from_pretrained(self.model_path)

    def build_model(self):
        """build the vision part of a VLM model when backend is turbomind, or
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
        """refers to the spec of `super.preprocess()"""
        images = self.collect_images(messages)
        outputs = []
        for image, _ in images:
            image = image.convert('RGB')
            pixel_values, num_image_tokens, images_spatial_crop = self.image_processor.process_single_image(
                image=image, cropping=len(images) <= 2)

            outputs.append(
                dict(pixel_values=pixel_values,
                     image_tokens=num_image_tokens,
                     image_token_id=0,
                     image_size=image.size,
                     images_spatial_crop=images_spatial_crop))
        messages.append(dict(role='preprocess', content=outputs))
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
        # TODO, add deepseek-vl2 model support for turbomind engine
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

    # https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/processing_deepseek_vl_v2.py#L523
    def get_images_seq_mask(self, prompt, messages, bos=True, eos=True, inference_mode=True) -> None:
        for message in messages:
            if message['role'] not in ['preprocess']:
                continue

            num_image_tokens = message['content'][0]['image_tokens']
            if isinstance(num_image_tokens, int):
                num_image_tokens = [num_image_tokens]

            images_seq_mask = []

            IMAGE_TOKEN = '<IMAGE_TOKEN>'
            text_splits = prompt.split(IMAGE_TOKEN)
            assert len(text_splits) == len(num_image_tokens) + 1

            for sep_idx, text_sep in enumerate(text_splits):
                tokenized_sep = self.image_processor.encode(text_sep, bos=False, eos=False)
                images_seq_mask += [False] * len(tokenized_sep)

                if sep_idx < len(num_image_tokens):
                    images_seq_mask += [True] * num_image_tokens[sep_idx]
            """add the bos and eos tokens"""
            if bos:
                images_seq_mask = [False] + images_seq_mask
            if eos:
                images_seq_mask = images_seq_mask + [False]

            if inference_mode:
                # remove eos token
                images_seq_mask = images_seq_mask[:-1]

            # add images_seq_mask into messages, used in vision embedding extractions
            message['content'][0]['images_seq_mask'] = images_seq_mask

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        self.get_images_seq_mask(prompt, messages)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
