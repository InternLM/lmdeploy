# Copyright (c) OpenMMLab. All rights reserved.
import fire

from .lite import SubCliLite


class CLI(object):
    """LMDeploy Command Line Interface.

    The CLI provides a unified API for converting, compressing and deploying
    large language models.
    """

    def deploy(self,
               model_name: str,
               model_path: str,
               model_format: str = None,
               tokenizer_path: str = None,
               dst_path: str = './workspace',
               tp: int = 1,
               quant_path: str = None,
               group_size: int = 0):
        """deploy llama family models via turbomind.

        Args:
            model_name (str): the name of the to-be-deployed model, such as
                llama-7b, llama-13b, vicuna-7b and etc
            model_path (str): the directory path of the model
            model_format (str): the format of the model, fb or hf. 'fb' stands
                for META's llama format, and 'hf' means huggingface format
            tokenizer_path (str): the path of tokenizer model
            dst_path (str): the destination path that saves outputs
            tp (int): the number of GPUs used for tensor parallelism, should
                be 2^n
            quant_path (str): path of the quantized model, which can be None
            group_size (int): a parameter used in AWQ to quantize fp16 weights
                to 4 bits
        """
        from lmdeploy.serve.turbomind.deploy import main as deploy
        deploy(model_name,
               model_path,
               model_format=model_format,
               tokenizer_path=tokenizer_path,
               dst_path=dst_path,
               tp=tp,
               quant_path=quant_path,
               group_size=group_size)


def run():
    cli = CLI()
    cli.lite = SubCliLite()
    fire.Fire(cli, name='lmdeploy')
