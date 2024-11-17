# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_qwen_vl_deps_install():
    """check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install qwen_vl_utils by pip install qwen_vl_utils'  # noqa: E501
        )
    try:
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen2VLModel(VisonModel):
    """Qwen2VL model."""

    _arch = 'Qwen2VLForConditionalGeneration'

    def build_proprocessor(self):
        check_qwen_vl_deps_install()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            # disable accelerate check_tied_parameters_in_config
            # for Qwen2-VL-2B-Instruct
            config.tie_word_embeddings = False

            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration._from_config(config)
            if not self.with_llm:
                del model.model
                del model.lm_head
            else:
                self.vl_model = model
            model.half()

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=['Qwen2VLVisionBlock'],
                dtype=torch.half)

        self.model = model.eval()

    def preprocess(self, messages: List[Dict]) -> Dict:
        """preprocess multimodal data in the messages, of which only the last
        item includes the mulitmodal data.

        Args:
            message(Dict): multimodal data in a dict, which is as follows:
            [
                {'role': 'user', 'content': 'user prompt'},
                {'role': 'assisant', 'content': 'AI reponse'},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'string',
                        },
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            key1: value1,
                            ...
                        },
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            key1: value1,
                            ...
                        },
                        ...
                    ]
                }
            ]
        Returns:
            the preprocessing results in a dict
        """
        assert isinstance(messages, List)
        assert isinstance(messages[-1]['content'], List)
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=False)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        outputs = self.processor(text=[text],
                                                 images=image_inputs,
                                                 videos=video_inputs,
                                                 return_tensors='pt')
        outputs.pop('attention_mask')
        outputs.update(prompt=text)
        return outputs

    @torch.no_grad()
    def forward(self, preprocess_output) -> List[torch.Tensor]:
        pixel_values = preprocess_output['pixel_values'].to(
            self.model.visual.get_device())
        image_grid_thw = preprocess_output['image_grid_thw'].to(
            self.model.visual.get_device())
        pixel_values = pixel_values.type(self.model.visual.get_dtype())
        image_embeds = self.model.visual(pixel_values,
                                         grid_thw=image_grid_thw).cpu()
        merge_length = self.processor.image_processor.merge_size**2
        split_size = preprocess_output['image_grid_thw'].prod(
            dim=1) // merge_length
        image_embeds = image_embeds.split(split_size.tolist())

        outputs = []
        for i, embeddings in enumerate(image_embeds):
            outputs.append(
                dict(embeddings=embeddings,
                     grid_thw=preprocess_output['image_grid_thw'][i].tolist()))
        return outputs

    def to_pytorch(self, messages, chat_template, sequence_start):
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        input_ids = [
            torch.squeeze(message['preprocess']['input_ids'])
            for message in messages if 'preprocess' in message
        ]
        # inputs_ids[1:] contains bos_id which should be removed
        input_ids = [
            input_id if i == 0 else input_id[1:]
            for i, input_id in enumerate(input_ids)
        ]
        input_ids = torch.concat(input_ids, dim=-1)
        pixel_values = [
            message['preprocess']['pixel_values'] for message in messages
            if 'preprocess' in message
        ]
        pixel_values = torch.concat(pixel_values, dim=0)
        image_grid_thw = [
            message['preprocess']['image_grid_thw'] for message in messages
            if 'preprocess' in message
        ]
        image_grid_thw = torch.concat(image_grid_thw, dim=0)

        return dict(prompt=text,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw)

    def to_turbomind(self, messages, chat_template, sequence_start):
        pass
