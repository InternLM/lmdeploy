# Copyright (c) OpenMMLab. All rights reserved.

import argparse


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


def get_engine_parser(add_pytorch: bool = False,
                      add_turbomind: bool = False) -> argparse.ArgumentParser:
    """Create the parser for engine config.

    Args:
        add_pytorch (bool): Whether to include pytorch engine arguments.
        add_turbomind (bool): Whether to include turbomind engine arguments.

    Returns:
        argparse.ArgumentParser
    """
    assert add_pytorch or add_turbomind, 'Should at least include one engine'
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument(
        '--tp',
        type=int,
        default=1,
        help='GPU number used in tensor parallelism. Should be 2^n')
    parser.add_argument('--max-batch-size',
                        type=int,
                        default=128,
                        help='Maximum batch size')

    if add_pytorch:
        name = 'engine arguments' if not add_turbomind \
            else 'pytorch engine arguments'
        group = parser.add_argument_group(name)
        group.add_argument('--block-size',
                           type=int,
                           default=64,
                           help='The block size for paging cache')
    if add_turbomind:
        name = 'engine arguments' if not add_pytorch \
            else 'turbomind engine arguments'
        group = parser.add_argument_group(name)
        group.add_argument('--model-format',
                           type=str,
                           default=None,
                           choices=['hf', 'llama', 'awq'],
                           help='The format of input model')
        group.add_argument('--quant-policy',
                           type=int,
                           default=0,
                           help='Whether to use kv int8')
        group.add_argument('--rope-scaling-factor',
                           type=float,
                           default=0.0,
                           help='Rope scaling factor')
        group.add_argument('--use-logn-attn',
                           action='store_true',
                           default=False,
                           help='Whether to use logn attention scaling')
    return parser
