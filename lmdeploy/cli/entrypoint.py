# Copyright (c) OpenMMLab. All rights reserved.
from .chat import SubCliChat
from .cli import CLI
from .lite import SubCliLite
from .serve import SubCliServe


def run():
    """The entry point of running LMDeploy CLI."""
    CLI.add_parsers()
    SubCliChat.add_parsers()
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
            elif command == 'chat':
                SubCliChat.parser.print_help()
            else:
                parser.print_help()
