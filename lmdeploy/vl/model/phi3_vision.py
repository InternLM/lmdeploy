# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

from transformers import AutoProcessor

from lmdeploy.vl.model.llava_hf import VISION_MODELS, LlavaHfVisionModel


@VISION_MODELS.register_module()
class Phi3VisionModel(LlavaHfVisionModel):
    """Phi3-vision model."""

    _arch = 'Phi3VForCausalLM'

    def build_preprocessor(self):
        processor = AutoProcessor.from_pretrained(self.model_path,
                                                  trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.tokenizer = None
        self.processor = processor

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            result = self.processor.image_processor(image, return_tensors='pt')
            h = result['image_sizes'][0][0].item() // 336
            w = result['image_sizes'][0][1].item() // 336
            image_tokens = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
            result.update(
                dict(image_size=image.size,
                     image_tokens=image_tokens,
                     image_token_id=0))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages
