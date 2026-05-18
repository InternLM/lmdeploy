# Copyright (c) OpenMMLab. All rights reserved.


from transformers import AutoProcessor

from lmdeploy.vl.model.llava_hf import VISION_MODELS, LlavaHfVisionModel


@VISION_MODELS.register_module()
class Phi3VisionModel(LlavaHfVisionModel):
    """Phi3-vision model."""

    _arch = 'Phi3VForCausalLM'

    def build_preprocessor(self, trust_remote_code: bool = False):
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.tokenizer = None
        self.processor = processor

    def build_model(self, trust_remote_code: bool = False):
        if self.with_llm:
            from transformers import AutoModelForCausalLM
            self.vl_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                 device_map='cpu',
                                                                 trust_remote_code=trust_remote_code)
        else:
            raise NotImplementedError('turbomind has not supported phi3v yet')

    def preprocess(self, messages: list[dict]) -> list[dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_multimodal_items(messages)
        outputs = []
        for modality, image, params in images:
            result = self.processor.image_processor([image], return_tensors='pt')
            image_tokens = result['num_img_tokens']
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages
