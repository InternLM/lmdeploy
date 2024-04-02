# Copyright (c) OpenMMLab. All rights reserved.
import sys

from .chat import SubCliChat
from .cli import CLI
from .lite import SubCliLite
from .serve import SubCliServe


def run():
    """The entry point of running LMDeploy CLI."""
    args = sys.argv[1:]
    is_deprecated_chat_cli = args[0] == 'chat' and (args[1]
                                                    in ['torch', 'turbomind'])
    CLI.add_parsers()
    if is_deprecated_chat_cli:
        SubCliChat.add_parsers()
    else:
        CLI.add_parser_chat()
    SubCliServe.add_parsers()
    SubCliLite.add_parsers()
    parser = CLI.parser
    args = parser.parse_args()

    if 'run' in dir(args):
        args.run(args)
    else:
        try:
            args.print_help()
        except AttributeError:
            command = args.command
            if command == 'serve':
                SubCliServe.parser.print_help()
            elif command == 'lite':
                SubCliLite.parser.print_help()
            elif command == 'chat' and is_deprecated_chat_cli:
                SubCliChat.parser.print_help()
            else:
                parser.print_help()
