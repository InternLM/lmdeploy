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
    parser.add_argument('--model-name',
                        type=str,
                        default='',
                        help='Model name ')
    parser.add_argument('--session-len', type=int, default=2048, help='sess')
    parser.add_argument('--max-batch-size',
                        type=int,
                        default=64,
                        help='batch-size')

    if add_pytorch:
        group = parser.add_argument_group('PyTorch arguments')
        group.add_argument('--num-cpu-blocks', type=int, default=0, help='x')
        group.add_argument('--num-gpu-blocks', type=int, default=0, help='x')

    if add_turbomind:
        group = parser.add_argument_group('TurboMind arguments')
        group.add_argument('--cache_max_entry_count',
                           type=float,
                           default=0.5,
                           help='x')
        group.add_argument('--cache_block_seq_len',
                           type=int,
                           default=128,
                           help='x')
    return parser
