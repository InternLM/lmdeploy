# Copyright (c) OpenMMLab. All rights reserved.

import argparse
from typing import List


class DefaultsAndTypesHelpFormatter(argparse.HelpFormatter):
    """Formatter to output default value and type in help information."""

    def _get_help_string(self, action):
        """Add default and type info into help."""
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if (action.option_strings or action.nargs
                        in defaulting_nargs) and 'default' not in help.lower():
                    help += '. Default: %(default)s'
                if action.type:
                    help += '. Type: %(type)s'
        return help


def convert_args(args):
    """Convert args to dict format."""
    special_names = ['run', 'command']
    kwargs = {
        k[0]: k[1]
        for k in args._get_kwargs() if k[0] not in special_names
    }
    return kwargs


def get_lora_adapters(adapters: List[str]):
    """Parse lora adapers from cli input.

    Args:
        adapters (List[str]): CLI input string of lora adapter path(s).

    Returns:
        Dict[str,str] or None: Parsed lora adapter path(s).
    """
    if not adapters:
        return None
    n = len(adapters)
    output = {}
    if n == 1:
        name = 'default'
        path = adapters[0].strip()
        if '=' in path:
            name, path = path.split('=', 1)
        output[name] = path
    else:
        for pair in adapters:
            assert '=' in pair, f'Multiple lora paths must in format of ' \
                                 f'xxx=yyy. But given: {pair}'
            name, path = pair.strip().split('=', 1)
            assert name not in output, f'Multiple lora paths with ' \
                                       f'repeated lora name: {name}'
            output[name] = path
    return output


class ArgumentHelper:
    """Helper class to add unified argument."""

    @staticmethod
    def model_name(parser):
        """Add argument model_name to parser."""

        return parser.add_argument(
            '--model-name',
            type=str,
            default=None,
            help='The name of the to-be-deployed model, such as'
            ' llama-7b, llama-13b, vicuna-7b and etc. You '
            'can run `lmdeploy list` to get the supported '
            'model names')

    @staticmethod
    def model_format(parser):
        """Add argument model_format to parser."""

        return parser.add_argument(
            '--model-format',
            type=str,
            default=None,
            choices=['hf', 'llama', 'awq'],
            help='The format of input model. `hf` meaning `hf_llama`, `llama` '
            'meaning `meta_llama`, `awq` meaning the quantized model by awq')

    @staticmethod
    def tp(parser):
        """Add argument tp to parser."""

        return parser.add_argument(
            '--tp',
            type=int,
            default=1,
            help='GPU number used in tensor parallelism. Should be 2^n')

    @staticmethod
    def session_id(parser):
        """Add argument session_id to parser."""

        return parser.add_argument('--session-id',
                                   type=int,
                                   default=1,
                                   help='The identical id of a session')

    @staticmethod
    def session_len(parser):
        """Add argument session_len to parser."""

        return parser.add_argument('--session-len',
                                   type=int,
                                   default=None,
                                   help='The max session length of a sequence')

    @staticmethod
    def max_batch_size(parser):
        """Add argument max_batch_size to parser."""

        return parser.add_argument('--max-batch-size',
                                   type=int,
                                   default=128,
                                   help='Maximum batch size')

    @staticmethod
    def quant_policy(parser):
        """Add argument quant_policy to parser."""

        return parser.add_argument('--quant-policy',
                                   type=int,
                                   default=0,
                                   help='Whether to use kv int8')

    @staticmethod
    def rope_scaling_factor(parser):
        """Add argument rope_scaling_factor to parser."""

        return parser.add_argument('--rope-scaling-factor',
                                   type=float,
                                   default=0.0,
                                   help='Rope scaling factor')

    @staticmethod
    def use_logn_attn(parser):
        """Add argument use_logn_attn to parser."""

        return parser.add_argument(
            '--use-logn-attn',
            action='store_true',
            default=False,
            help='Whether to use logn attention scaling')

    @staticmethod
    def block_size(parser):
        """Add argument block_size to parser."""

        return parser.add_argument('--block-size',
                                   type=int,
                                   default=64,
                                   help='The block size for paging cache')

    @staticmethod
    def top_p(parser):
        """Add argument top_p to parser."""

        return parser.add_argument(
            '--top-p',
            type=float,
            default=0.8,
            help='An alternative to sampling with temperature,'
            ' called nucleus sampling, where the model '
            'considers the results of the tokens with '
            'top_p probability mass')

    @staticmethod
    def top_k(parser):
        """Add argument top_k to parser."""

        return parser.add_argument(
            '--top-k',
            type=int,
            default=1,
            help='An alternative to sampling with temperature, '
            'where the model considers the top_k tokens '
            'with the highest probability')

    @staticmethod
    def temperature(parser):
        """Add argument temperature to parser."""

        return parser.add_argument('--temperature',
                                   type=float,
                                   default=0.8,
                                   help='Sampling temperature')

    @staticmethod
    def repetition_penalty(parser):
        """Add argument repetition_penalty to parser."""

        return parser.add_argument('--repetition-penalty',
                                   type=float,
                                   default=1.0,
                                   help='Parameter to penalize repetition')

    @staticmethod
    def cap(parser):
        """Add argument cap to parser."""

        return parser.add_argument(
            '--cap',
            type=str,
            default='chat',
            choices=['completion', 'infilling', 'chat', 'python'],
            help='The capability of a model. For example, codellama has the '
            'ability among ["completion", "infilling", "chat", "python"]')

    @staticmethod
    def log_level(parser):
        """Add argument log_level to parser."""

        import logging
        return parser.add_argument('--log-level',
                                   type=str,
                                   default='ERROR',
                                   choices=list(logging._nameToLevel.keys()),
                                   help='Set the log level')

    @staticmethod
    def backend(parser):
        """Add argument backend to parser."""

        return parser.add_argument('--backend',
                                   type=str,
                                   default='turbomind',
                                   choices=['pytorch', 'turbomind'],
                                   help='Set the inference backend')

    @staticmethod
    def engine(parser):
        """Add argument engine to parser."""

        return parser.add_argument('--engine',
                                   type=str,
                                   default='turbomind',
                                   choices=['pytorch', 'turbomind'],
                                   help='Set the inference backend')

    @staticmethod
    def stream_output(parser):
        """Add argument stream_output to parser."""

        return parser.add_argument(
            '--stream-output',
            action='store_true',
            help='Indicator for streaming output or not')

    @staticmethod
    def calib_dataset(parser):
        """Add argument calib_dataset to parser."""

        return parser.add_argument('--calib-dataset',
                                   type=str,
                                   default='ptb',
                                   help='The calibration dataset name')

    @staticmethod
    def calib_samples(parser):
        """Add argument calib_samples to parser."""

        return parser.add_argument(
            '--calib-samples',
            type=int,
            default=128,
            help='The number of samples for calibration')

    @staticmethod
    def calib_seqlen(parser):
        """Add argument calib_seqlen to parser."""

        return parser.add_argument('--calib-seqlen',
                                   type=int,
                                   default=2048,
                                   help='The sequence length for calibration')

    @staticmethod
    def device(parser):
        """Add argument device to parser."""

        return parser.add_argument('--device',
                                   type=str,
                                   default='cuda',
                                   choices=['cuda', 'cpu'],
                                   help='Device type of running')

    @staticmethod
    def meta_instruction(parser):
        """Add argument meta_instruction to parser."""

        return parser.add_argument('--meta-instruction',
                                   type=str,
                                   default=None,
                                   help='System prompt for ChatTemplateConfig')

    @staticmethod
    def cache_max_entry_count(parser):
        """Add argument cache_max_entry_count to parser."""

        return parser.add_argument(
            '--cache-max-entry-count',
            type=float,
            default=0.8,
            help='The percentage of gpu memory occupied by the k/v cache')

    @staticmethod
    def adapters(parser):
        """Add argument adapters to parser."""

        return parser.add_argument(
            '--adapters',
            nargs='*',
            type=str,
            default=None,
            help='Used to set path(s) of lora adapter(s). One can input '
            'key-value pairs in xxx=yyy format for multiple lora '
            'adapters. If only have one adapter, one can only input '
            'the path of the adapter.')

    @staticmethod
    def work_dir(parser):
        """Add argument work_dir to parser."""

        return parser.add_argument(
            '--work-dir',
            type=str,
            default='./work_dir',
            help='The working directory to save results')
