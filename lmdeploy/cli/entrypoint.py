# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys

from .cli import CLI
from .lite import SubCliLite
from .serve import SubCliServe


def run():
    """The entry point of running LMDeploy CLI."""
    args = sys.argv[1:]
    CLI.add_parsers()
    SubCliServe.add_parsers()
    SubCliLite.add_parsers()
    parser = CLI.parser
    args = parser.parse_args()

    if 'run' in dir(args):
        from lmdeploy.utils import get_model
        model_path = getattr(args, 'model_path', None)
        revision = getattr(args, 'revision', None)
        download_dir = getattr(args, 'download_dir', None)
        if model_path is not None and not os.path.exists(args.model_path):
            args.model_path = get_model(args.model_path,
                                        download_dir=download_dir,
                                        revision=revision)
        model_path_or_server = getattr(args, 'model_path_or_server', None)
        if model_path_or_server is not None and (
                ':' not in model_path_or_server
                and not os.path.exists(model_path_or_server)):
            args.model_path_or_server = get_model(args.model_path_or_server,
                                                  download_dir=download_dir,
                                                  revision=revision)

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
            else:
                parser.print_help()
