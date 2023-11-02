# Copyright (c) OpenMMLab. All rights reserved.
import fire

from .chat import SubCliChat
from .lite import SubCliLite
from .serve import SubCliServe


class CLI(object):
    """LMDeploy Command Line Interface.

    The CLI provides a unified API for converting, compressing and deploying
    large language models.
    """

    def convert(self,
                model_name: str,
                model_path: str,
                model_format: str = None,
                tokenizer_path: str = None,
                dst_path: str = './workspace',
                tp: int = 1,
                quant_path: str = None,
                group_size: int = 0):
        """Convert LLMs to lmdeploy format.

        Args:
            model_name (str): The name of the to-be-deployed model, such as
                llama-7b, llama-13b, vicuna-7b and etc.
            model_path (str): The directory path of the model
            model_format (str): the format of the model, should choose from
                ['llama', 'hf', 'awq', None]. 'llama' stands for META's llama
                format, 'hf' means huggingface llama format, and 'awq' means
                llama(hf) model quantized by lmdeploy/lite/quantization/awq.py.
                the default value is None, which means the model_format will be
                inferred based on model_name
            tokenizer_path (str): The path of tokenizer model.
            dst_path (str): The destination path that saves outputs.
            tp (int): The number of GPUs used for tensor parallelism, which
                should be 2^n.
            quant_path (str): Path of the quantized model, which can be None.
            group_size (int): A parameter used in AWQ to quantize fp16 weights
                to 4 bits.
        """
        from lmdeploy.turbomind.deploy.converter import main as convert

        convert(model_name,
                model_path,
                model_format=model_format,
                tokenizer_path=tokenizer_path,
                dst_path=dst_path,
                tp=tp,
                quant_path=quant_path,
                group_size=group_size)


def run():
    """The entry point of running LMDeploy CLI."""

    cli = CLI()
    cli.lite = SubCliLite()
    cli.chat = SubCliChat()
    cli.serve = SubCliServe()

    fire.Fire(cli, name='lmdeploy')
