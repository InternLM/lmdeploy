# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MiniCPMVModel(VisonModel):
    """MiniCPMV vision model."""

    _arch = 'MiniCPMV'

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        if not hasattr(self.hf_config, 'version'):
            raise ValueError('Can not find `version` in config.json. '
                             'Please checkout the latest model')
        version = str(self.hf_config.version)
        if version not in ['2.5', '2.6']:
            raise ValueError(
                f'Only support v2.5 and v2.6, but got version {version}')
        self.version = version

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.image_processor = self.processor.image_processor
        self._preprocess_func = (self._preprocess_v2_5 if self.version == '2.5'
                                 else self._preprocess_v2_6)

    def build_model(self):
        """build model & load weights."""
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            config = self.hf_config
            assert config.slice_mode is True, 'only support slice mode'
            config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
        if not self.with_llm:
            del model.llm
        else:
            self.vl_model = model

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=[
                    'Idefics2EncoderLayer', 'Resampler', 'SiglipEncoderLayer'
                ],
                dtype=torch.half)

        model.resampler.pos_embed = model.resampler.pos_embed.to(
            device=model.resampler.proj.device)
        self.config = config
        self.model = model.eval()

    def _get_slice_image(self, image: Image):
        slice_images = []
        source_image, patches, best_grid = self.image_processor.slice_image(
            image)
        slice_images.append(source_image)
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])
        return slice_images, best_grid

    def _reshape_by_patch(self, slice_images):
        tgt_sizes = []
        patches = []
        for slice_image in slice_images:
            slice_image = self.model.transform(slice_image)
            H, W = slice_image.shape[1:]
            slice_image = slice_image.numpy()
            slice_image = self.image_processor.reshape_by_patch(slice_image)
            slice_image = torch.from_numpy(slice_image)
            patches.append(slice_image)
            H //= self.config.patch_size
            W //= self.config.patch_size
            tgt_sizes.append(torch.Tensor([H, W]).type(torch.int32))
        return patches, tgt_sizes

    def _preprocess_v2_5(self, image: Image, params: Dict = None) -> Dict:
        """image preprocessing for MiniCPM-Llama3-V-2_5."""
        slice_images, best_grid = self._get_slice_image(image)
        # pixel_values, tgt_sizes are list of torch tensors
        pixel_values, tgt_sizes = self._reshape_by_patch(slice_images)
        num_patches = len(pixel_values)
        return dict(pixel_values=pixel_values,
                    tgt_sizes=tgt_sizes,
                    best_grid=best_grid,
                    num_patches=num_patches,
                    image_tokens=1,
                    image_token_id=0)

    def _preprocess_v2_6(self, image: Image, params: Dict = None) -> Dict:
        """image preprocessing for MiniCPM-V-2_6."""
        max_slice_nums = self.image_processor.max_slice_nums
        use_image_id = self.image_processor.use_image_id
        max_slice_nums = params.get('max_slice_nums', max_slice_nums)
        use_image_id = params.get('use_image_id', use_image_id)
        outputs = self.image_processor(image, max_slice_nums=max_slice_nums)
        pixel_values = outputs['pixel_values'][0]
        num_patches = len(pixel_values)
        pixel_values = [torch.as_tensor(x) for x in pixel_values]
        tgt_sizes = outputs['tgt_sizes'][0]
        tgt_sizes = [torch.as_tensor(x) for x in tgt_sizes]
        grid = self.image_processor.get_sliced_grid(
            image_size=image.size, max_slice_nums=max_slice_nums)

        return dict(
            pixel_values=pixel_values,  # a list
            tgt_sizes=tgt_sizes,  # a list
            best_grid=grid,
            num_patches=num_patches,
            image_tokens=1,
            image_token_id=0,
            use_image_id=use_image_id)

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to `super().preprocess() for spec."""
        outputs = []
        for item in messages[-1]['content']:
            if item['type'] == 'image':
                image = item['image'].convert('RGB')
                params = {
                    k: v
                    for k, v in item.items() if k not in {'type', 'image'}
                }
                result = self._preprocess_func(image, params)
                outputs.append(result)
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        """forward for MiniCPM-Llama3-V-2_5.

        Args:
            inputs(List[Dict]): the preprocessing result, each dict is
                the value returned by `_preprocess_v2_5`
        """
        tgt_sizes = [x['tgt_sizes'] for x in inputs]
        pixel_values = [x['pixel_values'] for x in inputs]
        # flatten the list
        tgt_sizes = list(itertools.chain(*tgt_sizes))
        pixel_values = list(itertools.chain(*pixel_values))
        pixel_values = [
            x.to(dtype=torch.half, device=self.model.device)
            for x in pixel_values
        ]
        pixel_values = [
            x.flatten(end_dim=1).permute(1, 0) for x in pixel_values
        ]
        pixel_values = torch.nn.utils.rnn.pad_sequence(pixel_values,
                                                       batch_first=True,
                                                       padding_value=0.0)
        B, L, _ = pixel_values.shape
        pixel_values = pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=self.model.device)
        for i in range(B):
            patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
        if self.version == '2.5':
            embeddings = self.model.vpm(
                pixel_values.type(torch.half),
                patch_attention_mask=patch_attn_mask).last_hidden_state
        else:
            embeddings = self.model.vpm(pixel_values.type(torch.half),
                                        patch_attention_mask=patch_attn_mask,
                                        tgt_sizes=tgt_sizes).last_hidden_state
        embeddings = self.model.resampler(embeddings, tgt_sizes)
        return embeddings

    def proc_messages(self, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        idx = 0
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            for x in message['preprocess']:
                prompt = f'<image>{IMAGE_TOKEN}</image>'
                if x.get('use_image_id', False):
                    prompt = f'<image_id>{idx}</image_id>' + prompt
                    idx += 1
                grid = x['best_grid']
                if grid is not None:
                    if self.version == '2.5':
                        slice = '\n'.join(
                            [f'<image>{IMAGE_TOKEN}</image>' * grid[0]] *
                            grid[1])
                        prompt = f'{prompt}<slice>{slice}</slice>\n'
                    elif self.version == '2.6':
                        slice = '\n'.join(
                            [f'<slice>{IMAGE_TOKEN}</slice>' * grid[0]] *
                            grid[1])
                        prompt = prompt + slice
                prompt += '\n'
            content = [
                x['text'] for x in message['content'] if x['type'] == 'text'
            ]
            prompt += content[0]
            prompt_messages.append(dict(role='user', content=prompt))
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
        features = []
        for message in messages:
            if 'preprocess' not in message.keys():
                continue
            inputs = message.pop('preprocess', None)
            embeddings = message.pop('forward', None)
            num_patches = [x['num_patches'] for x in inputs]
            embeddings = torch.split(embeddings, num_patches, 0)
            embeddings = [emmbedding.split(1) for emmbedding in embeddings]
            embeddings = list(itertools.chain(*embeddings))
            features.extend(embeddings)

        # flatten the list
        features = list(itertools.chain(*features))
        features = [x.cpu().numpy() for x in features]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(features) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(features)}')

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
