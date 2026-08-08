# Copyright (c) OpenMMLab. All rights reserved.

from transformers import AutoConfig, AutoProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

from lmdeploy.vl.model.internvl import VISION_MODELS, InternVLVisionModel


class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        'text_kwargs': {
            'padding': False,
        },
        'images_kwargs': {
            'crop_to_patches': True,
        },
        'videos_kwargs': {},
    }


@VISION_MODELS.register_module()
class InternVL3VisionModel(InternVLVisionModel):
    """Internvl3 vision model."""

    _arch = ['InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration']
    _turbomind_native_vision = True

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = '',
                 trust_remote_code: bool = False):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend, trust_remote_code=trust_remote_code)
        self.arch = self.hf_config.architectures[0]

    def build_preprocessor(self, trust_remote_code: bool = False):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.context_image_token_id
        self.image_tokens_per_patch = self.processor.image_seq_length
        self.tokenizer_init_kwargs = tokenizer.init_kwargs

    def build_model(self, trust_remote_code: bool = False):
        """InternVL3 does not support quantization."""
        raise NotImplementedError('Quantization is not supported for InternVL3VisionModel.')

    def preprocess(self, messages: list[dict]) -> list[dict]:
        """Refers to `super.preprocess() for spec."""
        from transformers.image_utils import make_flat_list_of_images
        output_kwargs = self.processor._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer_init_kwargs,
            **{
                'return_tensors': 'pt',
                'add_special_tokens': False
            },
        )
        images = self.collect_multimodal_items(messages)
        images = [image for modality, image, _ in images]
        num_image = len(images)
        images = make_flat_list_of_images(images)
        image_inputs = self.processor.image_processor(images, **output_kwargs['images_kwargs'])
        image_num_patches = image_inputs.pop('num_patches').cpu().numpy().tolist()
        image_pixel_values = image_inputs.pop('pixel_values')
        outputs = []
        cum_num_patches = 0
        for idx in range(num_image):
            cur_num_patches = image_num_patches[idx]
            pixel_values = image_pixel_values[cum_num_patches:cum_num_patches + cur_num_patches, ...]
            cum_num_patches += cur_num_patches
            data = dict(pixel_values=pixel_values.to(self.mm_feature_dtype),
                        image_tokens=self.image_tokens_per_patch * cur_num_patches,
                        image_token_id=self.image_token_id)
            outputs.append(data)

        messages.append(dict(role='preprocess', content=outputs))
        return messages
