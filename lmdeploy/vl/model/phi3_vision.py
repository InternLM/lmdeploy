# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


# from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py # noqa E501
def _process_image_embedding(self, pixel_values: torch.Tensor,
                             image_sizes: torch.Tensor):
    """process image embedding."""
    img_embeds = pixel_values
    img_sizes = image_sizes
    target_device = pixel_values.device
    target_dtype = pixel_values.dtype
    if self.use_hd_transform and img_sizes is not None and len(img_sizes):
        assert img_embeds.ndim == 5, f'img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'  # noqa E501
        # img_embeds: (num_images, max_num_crops, 3, H, W)
        # img_sizes: (num_images, 2).view(1, -1)

        bs = img_embeds.shape[0]
        # Nx(HW)xC
        img_features = self.get_img_features(img_embeds.flatten(0, 1))
        base_feat_height = base_feat_width = int(img_features.shape[1]**0.5)

        assert base_feat_height == 24 and base_feat_width == 24, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect 24x24 features for hd transform'  # noqa E501

        # bs x max_num_crops x (24x24) x C
        img_features = img_features.view(bs, -1,
                                         base_feat_height * base_feat_width,
                                         self.image_dim_out)
        C = self.image_dim_out
        H = base_feat_height

        output_imgs = []
        output_len = []
        # training is tensor, inference is list
        if isinstance(img_sizes, torch.Tensor):
            img_sizes = img_sizes.view(-1, 2)
        for _bs in range(bs):
            h, w = img_sizes[_bs]
            h = h // 336
            w = w // 336
            B_ = h * w

            # 1 x (24x24) x 1024
            global_img_feature = img_features[_bs, :1]

            # 1 x 12 x 12 x 4096
            glb_img = global_img_feature.reshape(1, H, H, C).reshape(
                1, H // 2, 2, H // 2, 2,
                C).contiguous().permute(0, 1, 3, 2, 4,
                                        5).reshape(1, H // 2, H // 2,
                                                   4 * C).contiguous()
            temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)

            # 1 x 156 x 4096
            glb_img = torch.cat([glb_img, temp_glb_GN],
                                dim=2).reshape(1, -1, 4 * C)

            # (max_num_crops-1) x (12x12) x C
            sub_img = img_features[_bs, 1:]
            # 16x574x1024
            # get rid of padding sub_img
            sub_img = sub_img[:B_]

            # (num_crops, 12, 2, 12, 2, 1024)->(num_crops, 12, 12, 2, 2, 1024)
            # -> (num_crops, 12*12, 4*1024)
            sub_img = sub_img.reshape(B_, H, H, C).reshape(
                B_, H // 2, 2, H // 2, 2,
                C).contiguous().permute(0, 1, 3, 2, 4,
                                        5).reshape(B_, -1, 4 * C).contiguous()
            sub_img = sub_img.reshape(1, h, w, 12, 12, -1).permute(
                0, 1, 3, 2, 4, 5).reshape(1, h * 12, w * 12, 4 * C)
            temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)
            sub_img = torch.cat([sub_img, temp_sub_GN],
                                dim=2).reshape(1, -1, 4 * C)
            # (1, num_img_tokens, 1024*4)

            # glb + sub
            if self.hd_transform_order == 'glb_sub':
                output_imgs.append(
                    torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
            elif self.hd_transform_order == 'sub_glb':
                output_imgs.append(
                    torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
            else:
                raise NotImplementedError(
                    f'hd_transform_order = {self.hd_transform_order}'
                )  # noqa E501

            temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
            assert temp_len == output_imgs[-1].shape[
                1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'  # noqa E501
            output_len.append(temp_len)

        img_set_tensor = []
        for _output_img in output_imgs:
            img_feature_proj = self.img_projection(
                _output_img.to(target_device).to(target_dtype))
            img_set_tensor.append(img_feature_proj)
    elif img_embeds.ndim == 4:
        tt = (self.get_img_features(img_embeds).to(target_device).to(
            target_dtype).reshape(-1, self.image_dim_out))
        img_set_tensor = self.img_projection(tt)  # adapted visual features.
    elif img_embeds.ndim == 3:
        tt = (img_embeds.to(target_device).to(target_dtype).view(
            -1, self.image_dim_out))
        img_set_tensor = self.img_projection(tt)  # adapted visual features.
    else:
        raise NotImplementedError
    return img_set_tensor


@VISION_MODELS.register_module()
class Phi3VisionModel(VisonModel):
    """Llava hf vision model."""

    _arch = 'Phi3VForCausalLM'

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(self.hf_config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.lm_head
                del model.model.layers
                del model.model.norm
                del model.model.embed_tokens
                del model.model.vision_embed_tokens.wte
            else:
                self.vl_model = model

        no_split_module_classes = ['CLIPEncoderLayer']
        max_memory = get_balanced_memory(
            model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            max_memory=max_memory,
            dtype=torch.half)
        same_device_keys = [('model.vision_embed_tokens.img_projection',
                             'model.vision_embed_tokens.sub_GN',
                             'model.vision_embed_tokens.glb_GN')]
        for keys in same_device_keys:
            keys = [k for k in keys if k in device_map]
            if len(keys) <= 1:
                continue
            for k in keys[1:]:
                device_map[k] = device_map[keys[0]]

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=no_split_module_classes,
                dtype=torch.half)

        model.eval()
        self.model = model
        # processor
        processor = AutoProcessor.from_pretrained(self.model_path,
                                                  trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.prtokenizer = None
        self.processor = processor.image_processor
        self.processor = processor

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        process_outputs = self.processor.image_processor(
            images, return_tensors='pt').to(device=self.model.device,
                                            dtype=self.model.dtype)
        pixel_values = process_outputs['pixel_values']
        image_sizes = process_outputs['image_sizes']
        image_features = _process_image_embedding(
            self.model.model.vision_embed_tokens,
            pixel_values=pixel_values,
            image_sizes=image_sizes)
        outputs = [x.squeeze() for x in image_features]
        return outputs
