# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

XMODELS = Registry('xmodel', locations=['lmdeploy.xmodel'])


@XMODELS.register_module(name='qwen-7b')
class QwenVLChat:

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
        if sequence_start:
            return f'{self.im_start}system\n{self.system}{self.im_end}' \
                   f'\n{self.im_start}user\n{prompt}{self.im_end}' \
                   f'\n{self.im_start}assistant\n'

        return f'\n{self.im_start}user\n{prompt}{self.im_end}' \
               f'\n{self.im_start}assistant\n'

    @property
    def image_seq_length(self):
        return 256

    @property
    def image_start_id(self):
        return 151857

    @property
    def image_end_id(self):
        return 151858
