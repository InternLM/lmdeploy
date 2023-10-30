# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

from lmdeploy.model import Qwen7BChat

XMODELS = Registry('xmodel', locations=['lmdeploy.xmodel'])


@XMODELS.register_module(name='qwen-7b')
class QwenVLChat(Qwen7BChat):

    def __init__(self,
                 session_len=8192,
                 top_p=0.5,
                 top_k=40,
                 temperature=1.0,
                 im_start='<|im_start|>',
                 im_end='<|im_end|>',
                 system='You are a helpful assistant.',
                 stop_words=['<|im_end|>'],
                 capability='chat',
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.im_start = im_start
        self.im_end = im_end
        self.system = system
        self.stop_words = stop_words
        self.capability = capability

    def get_prompt(self, prompt, images=None, sequence_start=True):
        assert self.capability == 'chat'
        return self.decorate_prompt(prompt, images, sequence_start)

    def decorate_prompt(self, prompt, images=None, sequence_start=True):
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if images is not None:
            res = ''
            for i in range(len(images)):
                res += f'Picture {str(i)}:<img>placeholder</img>\n'
            prompt = res + prompt
        return super().decorate_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, image_embs=None, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template."""
        if isinstance(messages, str):
            return self.get_prompt(messages, image_embs, sequence_start)
        system, users, assistants = self._translate_messages(messages)
        system = self.system if not system else system
        ret = f'{self.im_start}system\n{system}{self.im_end}'
        for user, img_emb, assistant in zip(users, image_embs, assistants):
            prompt_img = ''
            for i in range(len(img_emb)):
                prompt_img += f'Picture {str(i)}:<img>placeholder</img>\n'
            if assistant:
                ret += f'\n{self.im_start}user\n{prompt_img}{user}{self.im_end}' \
                       f'\n{self.im_start}assistant\n{assistant}'  # noqa: E501
            else:
                ret += f'\n{self.im_start}user\n{prompt_img}{user}{self.im_end}' \
                       f'\n{self.im_start}assistant\n'  # noqa: E501
        return ret

    @property
    def image_seq_length(self):
        return 256

    @property
    def image_start_id(self):
        return 151857

    @property
    def image_end_id(self):
        return 151858
