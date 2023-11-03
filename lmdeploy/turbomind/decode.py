# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import torch

from lmdeploy import turbomind as tm
from lmdeploy.tokenizer import Tokenizer

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def main(model_path, inputs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        inputs (str): the path of text file contatin input text lines
    """
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path, eos_id=tokenizer.eos_token_id)
    generator = tm_model.create_instance()

    with open(inputs, 'r') as f:
        lines = f.readlines()

    input_ids = [tokenizer.encode(x) for x in lines]

    logits = generator.decode(input_ids)

    top_1 = torch.argmax(logits, -1)

    print(top_1)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
