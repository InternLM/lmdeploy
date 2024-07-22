# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MiniCPMVModel(VisonModel):
    """MiniCPMV vision model."""

    _arch = 'MiniCPMV'

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
                no_split_module_classes=['Idefics2EncoderLayer', 'Resampler'],
                dtype=torch.half)

        model.resampler.pos_embed = model.resampler.pos_embed.to(
            device=model.resampler.proj.device)
        self.config = config
        self.model = model.eval()

        if hasattr(config, 'vision_config'):
            self._forward_func = self._forward_v2_5
        else:
            self._forward_func = self._forward_v2

    def _get_slice_image(self, image: Image):
        slice_images = []
        source_image, patches, best_grid = self.model.slice_image(image)
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
            patches.append(self.model.reshape_by_patch(slice_image))
            H //= self.config.patch_size
            W //= self.config.patch_size
            tgt_sizes.append(torch.Tensor([H, W]).type(torch.int32))
        return patches, tgt_sizes

    def _forward_v2(self, images: List[Image]):
        """forward for MiniCPM-V-2."""
        raise NotImplementedError

    def _forward_v2_5(self, images: List[Image]):
        """forward for MiniCPM-Llama3-V-2_5."""
        patches = []
        tgt_sizes = []
        best_grids = []
        num_patches = []
        for image in images:
            slice_images, best_grid = self._get_slice_image(image)
            _patches, _tgt_sizes = self._reshape_by_patch(slice_images)
            num_patches.append(len(_patches))
            patches.extend(_patches)
            tgt_sizes.extend(_tgt_sizes)
            best_grids.append(best_grid)

        patches = [
            x.to(dtype=torch.half, device=self.model.device) for x in patches
        ]
        patches = [x.flatten(end_dim=1).permute(1, 0) for x in patches]
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(patches,
                                                           batch_first=True,
                                                           padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=self.model.device)
        for i in range(B):
            patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
        vision_embedding = self.model.vpm(
            all_pixel_values.type(torch.half),
            patch_attention_mask=patch_attn_mask).last_hidden_state
        vision_embedding = self.model.resampler(vision_embedding, tgt_sizes)
        vision_embedding = torch.split(vision_embedding, num_patches, 0)
        outputs = []
        for embeddings, grid in zip(vision_embedding, best_grids):
            embeddings = embeddings.cpu()  # n x d x h
            outputs.append(dict(embeddings=embeddings, grid=grid))

        return outputs

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        images = [x.convert('RGB') for x in images]
        return self._forward_func(images)
