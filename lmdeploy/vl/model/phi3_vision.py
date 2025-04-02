# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

from transformers import AutoProcessor

from lmdeploy.vl.model.llava_hf import VISION_MODELS, LlavaHfVisionModel
from lmdeploy.vl.utils import hash_multimodal_data

@VISION_MODELS.register_module()
class Phi3VisionModel(LlavaHfVisionModel):
    """Phi3-vision model."""

    _arch = 'Phi3VForCausalLM'

    def build_preprocessor(self):
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.tokenizer = None
        self.processor = processor

    def build_model(self):
        if self.with_llm:
            from transformers import AutoModelForCausalLM
            self.vl_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                 device_map='cpu',
                                                                 trust_remote_code=True)
        else:
            raise NotImplementedError('turbomind has not supported phi3v yet')

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            hash_value = None
            if self.enable_prefix_caching:
                hash_value = hash_multimodal_data(model_id=self.model_path, image=image, params=params)
            result = self.processor.image_processor([image], return_tensors='pt')
            image_tokens = result['num_img_tokens']
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id, hash_value=hash_value))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages
