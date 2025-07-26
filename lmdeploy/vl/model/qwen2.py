# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_qwen_vl_deps_install():
    """Check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        raise ImportError('please install qwen_vl_utils by `pip install qwen_vl_utils`'  # noqa: E501
                          )
    try:
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen2VLModel(VisonModel):
    """Qwen2VL model."""

    _arch = ['Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration']

    def build_preprocessor(self):
        check_qwen_vl_deps_install()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(image_token)[-1]

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        from qwen_vl_utils import process_vision_info

        images = self.collect_images(messages)
        optional_keys = {'resized_height', 'resized_width', 'min_pixels', 'max_pixels'}
        outputs = []
        for image, params in images:
            image = image.convert('RGB')

            item = dict(type='image', image=image)
            item.update({key: params[key] for key in params.keys() if key in optional_keys})
            image_inputs, _ = process_vision_info([dict(content=[item])])
            result = self.processor.image_processor(images=image_inputs, videos=None, return_tensors='pt')
            merge_length = self.processor.image_processor.merge_size**2
            image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def build_model(self):
        check_qwen_vl_deps_install()
        arch = self.hf_config.architectures[0]
        if arch == 'Qwen2VLForConditionalGeneration':
            from transformers import Qwen2VLForConditionalGeneration as AutoModelCls
        elif arch == 'Qwen2_5_VLForConditionalGeneration':
            from transformers import Qwen2_5_VLForConditionalGeneration as AutoModelCls
        else:
            raise ValueError(f'Unsupported arch={arch}')

        if self.with_llm:
            self.vl_model = AutoModelCls.from_pretrained(self.model_path, device_map='cpu')
        else:
            from accelerate import init_empty_weights
            with init_empty_weights():
                config = self.hf_config
                # disable accelerate check_tied_parameters_in_config for Qwen2-VL-2B-Instruct
                config.tie_word_embeddings = False
                model = AutoModelCls._from_config(config)
                if hasattr(AutoModelCls, 'visual'):
                    # transformers >= 4.52.0 modified model structure
                    # https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1791-L1800
                    model.visual = model.model.visual
                del model.model
                del model.lm_head
                model.half()

            from accelerate import load_checkpoint_and_dispatch
            with disable_logging():
                load_checkpoint_and_dispatch(model=model,
                                             checkpoint=self.model_path,
                                             device_map='auto' if not self.with_llm else {'': 'cpu'},
                                             max_memory=self.max_memory,
                                             no_split_module_classes=['Qwen2VLVisionBlock', 'Qwen2_5_VLVisionBlock'],
                                             dtype=torch.half)
            self.model = model.eval()

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
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        dtype = torch.half
        device = next(self.model.visual.parameters()).device
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'].type(dtype) for x in inputs[idx:idx + max_batch_size]]
            image_grid_thw = [x['image_grid_thw'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0).to(device)
            image_grid_thw = torch.cat(image_grid_thw, dim=0).to(device)
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            merge_length = self.processor.image_processor.merge_size**2
            split_size = image_grid_thw.prod(dim=1) // merge_length
            image_embeds = image_embeds.split(split_size.tolist())
            outputs.extend(image_embeds)
        messages.append(dict(role='forward', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [item['text'] for item in message['content'] if item['type'] == 'text']
            prompt = content[0]
            if IMAGE_TOKEN in prompt and '<|vision_start|>' not in prompt:
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>')
            else:
                # Qwen2-VL-2B-Instruct will concat image and user prompt
                # according to their order in the content list
                # we insert image token before user prompt by default. The
                # user can use custom image token position if they want the
                # same decorated prompt as Qwen2-VL
                prompt = f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>' * \
                    n_images + prompt
            prompt_messages.append(dict(role=message['role'], content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    @staticmethod
    def get_mrope_info(seq_len: int,
                       grid_thws: List[Tuple[int, int, int]] = None,
                       ranges: List[Tuple[int, int]] = None):
        mrope_position_ids = [torch.arange(ranges[0][0]).expand(3, -1)]
        st_idx = ranges[0][0]
        for i, (grid_thw, embedding_range) in enumerate(zip(grid_thws, ranges)):
            llm_grid_t, llm_grid_h, llm_grid_w = grid_thw
            llm_grid_h //= 2
            llm_grid_w //= 2
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            mrope_position_ids.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            st_idx += max(llm_grid_h, llm_grid_w)
            if i < len(ranges) - 1:
                text_len = ranges[i + 1][0] - ranges[i][1]
            else:
                text_len = seq_len - embedding_range[1]
            mrope_position_ids.append(torch.arange(text_len).expand(3, -1) + st_idx)
            st_idx += text_len
        mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
        mrope_position_delta = torch.tensor([st_idx - seq_len], dtype=torch.long)
        return mrope_position_ids, mrope_position_delta

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        info = super().to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        grid_thws = [x['image_grid_thw'].tolist()[0] for x in inputs]
        seq_len = len(info['input_ids'])
        ranges = info['input_embedding_ranges']
        mrope_position_ids, mrope_position_delta = self.get_mrope_info(seq_len, grid_thws, ranges)
        meta = dict(mrope_position_ids=mrope_position_ids, mrope_position_delta=mrope_position_delta)
        info.update(dict(input_meta=meta))
        return info
