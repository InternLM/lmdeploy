# Copyright (c) OpenMMLab. All rights reserved.

import argparse

from mmengine.config import DictAction


class DefaultsAndTypesHelpFormatter(argparse.HelpFormatter):

    def _get_help_string(self, action):
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
    kwargs = {k[0]: k[1] for k in args._get_kwargs() if k[0] not in ['run']}
    return kwargs


class ArgumentHelper:
    """Helper class to add unified argument."""

    @staticmethod
    def model_name(parser):
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
        return parser.add_argument(
            '--model-format',
            type=str,
            default=None,
            choices=['hf', 'llama', 'awq'],
            help='The format of input model. `hf` meaning `hf_llama`, `llama` '
            'meaning `meta_llama`, `awq` meaning the quantized model by awq')

    @staticmethod
    def tp(parser):
        return parser.add_argument(
            '--tp',
            type=int,
            default=1,
            help='GPU number used in tensor parallelism. Should be 2^n')

    @staticmethod
    def session_id(parser):
        return parser.add_argument('--session-id',
                                   type=int,
                                   default=1,
                                   help='The identical id of a session')

    @staticmethod
    def session_len(parser):
        return parser.add_argument('--session-len',
                                   type=int,
                                   default=None,
                                   help='The max session length of a sequence')

    @staticmethod
    def max_batch_size(parser):
        return parser.add_argument('--max-batch-size',
                                   type=int,
                                   default=128,
                                   help='Maximum batch size')

    @staticmethod
    def quant_policy(parser):
        return parser.add_argument('--quant-policy',
                                   type=int,
                                   default=0,
                                   help='Whether to use kv int8')

    @staticmethod
    def rope_scaling_factor(parser):
        return parser.add_argument('--rope-scaling-factor',
                                   type=float,
                                   default=0.0,
                                   help='Rope scaling factor')

    @staticmethod
    def use_logn_attn(parser):
        return parser.add_argument(
            '--use-logn-attn',
            action='store_true',
            default=False,
            help='Whether to use logn attention scaling')

    @staticmethod
    def block_size(parser):
        return parser.add_argument('--block-size',
                                   type=int,
                                   default=64,
                                   help='The block size for paging cache')

    @staticmethod
    def top_p(parser):
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
        return parser.add_argument(
            '--top-k',
            type=int,
            default=1,
            help='An alternative to sampling with temperature, '
            'where the model considers the top_k tokens '
            'with the highest probability')

    @staticmethod
    def temperature(parser):
        return parser.add_argument('--temperature',
                                   type=float,
                                   default=0.8,
                                   help='Sampling temperature')

    @staticmethod
    def repetition_penalty(parser):
        return parser.add_argument('--repetition-penalty',
                                   type=float,
                                   default=1.0,
                                   help='Parameter to penalize repetition')

    @staticmethod
    def cap(parser):
        return parser.add_argument(
            '--cap',
            type=str,
            default='chat',
            choices=['completion', 'infilling', 'chat', 'python'],
            help='The capability of a model. For example, codellama has the '
            'ability among ["completion", "infilling", "chat", "python"]')

    @staticmethod
    def log_level(parser):
        import logging
        return parser.add_argument('--log-level',
                                   type=str,
                                   default='ERROR',
                                   choices=list(logging._nameToLevel.keys()),
                                   help='Set the log level')

    @staticmethod
    def backend(parser):
        return parser.add_argument('--backend',
                                   type=str,
                                   default='turbomind',
                                   choices=['pytorch', 'turbomind'],
                                   help='Set the inference backend')

    @staticmethod
    def engine(parser):
        return parser.add_argument('--engine',
                                   type=str,
                                   default='turbomind',
                                   choices=['pytorch', 'turbomind'],
                                   help='Set the inference backend')

    @staticmethod
    def stream_output(parser):
        return parser.add_argument(
            '--stream-output',
            action='store_true',
            help='Indicator for streaming output or not')

    @staticmethod
    def calib_dataset(parser):
        return parser.add_argument(
            '--calib-dataset',
            type=str,
            default='c4',
            help='The calibration dataset name. defaults to "c4"')

    @staticmethod
    def calib_samples(parser):
        return parser.add_argument(
            '--calib-samples',
            type=int,
            default=128,
            help='The number of samples for calibration. defaults to 128')

    @staticmethod
    def calib_seqlen(parser):
        return parser.add_argument(
            '--calib-seqlen',
            type=int,
            default=2048,
            help='The sequence length for calibration. defaults to 2048')

    @staticmethod
    def device(parser):
        return parser.add_argument('--device',
                                   type=str,
                                   default='cuda',
                                   choices=['cuda', 'cpu'],
                                   help='Device type of running')

    @staticmethod
    def meta_instruction(parser):
        return parser.add_argument('--meta-instruction',
                                   type=str,
                                   default=None,
                                   help='System prompt for ChatTemplateConfig')

    @staticmethod
    def cache_max_entry_count(parser):
        return parser.add_argument(
            '--cache-max-entry-count',
            type=float,
            default=0.5,
            help='The percentage of gpu memory occupied by the k/v cache')

    @staticmethod
    def adapters(parser):
        return parser.add_argument(
            '--adapters',
            default=None,
            action=DictAction,
            help='Used key-values pairs in xxx=yyy format'
            ' to set the path lora adapter')
