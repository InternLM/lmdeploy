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
        args.print_help()
