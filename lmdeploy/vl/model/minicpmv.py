# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MiniCPMVModel(VisionModel):
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
            raise ValueError(f'Only support v2.5 and v2.6, but got version {version}')
        self.version = version

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.image_processor = self.processor.image_processor
        self._preprocess_func = (self._preprocess_v2_5 if self.version == '2.5' else self._preprocess_v2_6)

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            config = self.hf_config
            assert config.slice_mode is True, 'only support slice mode'
            config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self.vl_model = model
        if not self.with_llm:
            del model.llm

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['Idefics2EncoderLayer', 'Resampler', 'SiglipEncoderLayer'],
                dtype=torch.half)

        model.resampler.pos_embed = model.resampler.pos_embed.to(device=model.resampler.proj.device)
        self.config = config
        self.model = model.eval()

    def _get_slice_image(self, image: Image):
        slice_images = []
        source_image, patches, best_grid = self.image_processor.slice_image(image)
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
        """Image preprocessing for MiniCPM-Llama3-V-2_5."""
        slice_images, best_grid = self._get_slice_image(image)
        # pixel_values, tgt_sizes are list of torch tensors
        pixel_values, tgt_sizes = self._reshape_by_patch(slice_images)
        num_patches = len(pixel_values)
        return dict(
            pixel_values=pixel_values,  # a list
            tgt_sizes=tgt_sizes,  # a list
            best_grid=best_grid,
            num_patches=num_patches,
            image_tokens=1,
            image_token_id=self.image_token_id)

    def _preprocess_v2_6(self, image: Image, params: Dict = None) -> Dict:
        """Image preprocessing for MiniCPM-V-2_6."""
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
        grid = self.image_processor.get_sliced_grid(image_size=image.size, max_slice_nums=max_slice_nums)
        return dict(
            pixel_values=pixel_values,  # a list
            tgt_sizes=tgt_sizes,  # a list
            best_grid=grid,
            num_patches=num_patches,
            image_tokens=1,
            image_token_id=self.image_token_id,
            use_image_id=use_image_id)

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess() for spec."""
        outputs = []
        for i, message in enumerate(messages):
            if message['role'] != 'user' or not isinstance(message['content'], List):
                continue
            for item in message['content']:
                if item['type'] == 'image':
                    image = item['image'].convert('RGB')
                    params = {k: v for k, v in item.items() if k not in {'type', 'image'}}
                    result = self._preprocess_func(image, params)
                    outputs.append(result)
            messages[i].update(dict(preprocess=outputs))
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
        # collect preprocess results into a list
        inputs = []
        inputs = [x['preprocess'] for x in messages if 'preprocess' in x.keys()]
        # flatten the list
        inputs = list(itertools.chain(*inputs))
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            tgt_sizes = [x['tgt_sizes'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            num_patches = [x['num_patches'] for x in inputs[idx:idx + max_batch_size]]
            # flatten the list
            tgt_sizes = list(itertools.chain(*tgt_sizes))
            pixel_values = list(itertools.chain(*pixel_values))
            pixel_values = [x.to(dtype=torch.half, device=self.model.device) for x in pixel_values]
            pixel_values = [x.flatten(end_dim=1).permute(1, 0) for x in pixel_values]
            pixel_values = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = pixel_values.shape
            pixel_values = pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=self.model.device)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            if self.version == '2.5':
                for j in range(B):
                    patch_attn_mask[j, :tgt_sizes[j][0] * tgt_sizes[j][1]] = True
                embeddings = self.model.vpm(pixel_values.type(torch.half),
                                            patch_attention_mask=patch_attn_mask).last_hidden_state
            else:
                for j in range(B):
                    patch_attn_mask[j, 0, :tgt_sizes[j][0] * tgt_sizes[j][1]] = True
                embeddings = self.model.vpm(pixel_values.type(torch.half),
                                            patch_attention_mask=patch_attn_mask,
                                            tgt_sizes=tgt_sizes).last_hidden_state

            embeddings = self.model.resampler(embeddings, tgt_sizes)
            embeddings = torch.split(embeddings, num_patches, 0)
            for embedding in embeddings:
                embedding = embedding.split(1, dim=0)
                outputs.extend([x.squeeze() for x in embedding])
        messages.append(dict(role='forward', content=outputs))
        return messages

    def proc_messages(self, messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        idx = 0
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            if 'preprocess' not in message.keys():
                continue
            prompts = []
            for x in message['preprocess']:
                prompt = f'<image>{IMAGE_TOKEN}</image>'
                if x.get('use_image_id', False):
                    prompt = f'<image_id>{idx}</image_id>' + prompt
                    idx += 1
                grid = x['best_grid']
                if grid is not None:
                    if self.version == '2.5':
                        slice = '\n'.join([f'<image>{IMAGE_TOKEN}</image>' * grid[0]] * grid[1])
                        prompt = f'{prompt}<slice>{slice}</slice>\n'
                    elif self.version == '2.6':
                        slice = '\n'.join([f'<slice>{IMAGE_TOKEN}</slice>' * grid[0]] * grid[1])
                        prompt = prompt + slice
                        prompt += '\n'
                else:
                    prompt = (prompt + '\n' if self.version == '2.6' else prompt)
                prompts.append(prompt)
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = ''.join(prompts) + content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
