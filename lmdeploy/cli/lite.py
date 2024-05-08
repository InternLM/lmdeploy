# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import DictAction

from .cli import CLI
from .utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, convert_args


class SubCliLite(object):
    """CLI for compressing LLMs."""
    _help = 'Compressing and accelerating LLMs with lmdeploy.lite module'
    _desc = _help
    parser = CLI.subparsers.add_parser(
        'lite',
        help=_help,
        description=_desc,
    )
    subparsers = parser.add_subparsers(
        title='Commands', description='This group has the following commands:')

    @staticmethod
    def add_parser_auto_awq():
        """Add parser for auto_awq command."""
        parser = SubCliLite.subparsers.add_parser(
            'auto_awq',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliLite.auto_awq.__doc__,
            help=SubCliLite.auto_awq.__doc__)
        parser.set_defaults(run=SubCliLite.auto_awq)
        parser.add_argument('model',
                            type=str,
                            help='The path of model in hf format')
        ArgumentHelper.work_dir(parser)
        ArgumentHelper.calib_dataset(parser)
        ArgumentHelper.calib_samples(parser)
        ArgumentHelper.calib_seqlen(parser)
        ArgumentHelper.calib_batchsize(parser)
        ArgumentHelper.calib_search_scale(parser)
        ArgumentHelper.device(parser)
        parser.add_argument('--w-bits',
                            type=int,
                            default=4,
                            help='Bit number for weight quantization')
        parser.add_argument('--w-sym',
                            action='store_true',
                            help='Whether to do symmetric quantization')
        parser.add_argument(
            '--w-group-size',
            type=int,
            default=128,
            help='Group size for weight quantization statistics')

    @staticmethod
    def add_parser_calibrate():
        """Add parser for calibrate command."""
        parser = SubCliLite.subparsers.add_parser(
            'calibrate',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliLite.calibrate.__doc__,
            help=SubCliLite.calibrate.__doc__)
        parser.set_defaults(run=SubCliLite.calibrate)
        parser.add_argument('model',
                            type=str,
                            help='The name or path of the model to be loaded')
        ArgumentHelper.work_dir(parser)
        ArgumentHelper.calib_dataset(parser)
        ArgumentHelper.calib_samples(parser)
        ArgumentHelper.calib_seqlen(parser)
        ArgumentHelper.calib_batchsize(parser)
        ArgumentHelper.calib_search_scale(parser)
        ArgumentHelper.device(parser)

    @staticmethod
    def add_parser_smooth_quant():
        """Add parser for smooth_quant command."""
        parser = SubCliLite.subparsers.add_parser(
            'smooth_quant',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliLite.smooth_quant.__doc__,
            help=SubCliLite.smooth_quant.__doc__)
        parser.set_defaults(run=SubCliLite.smooth_quant)
        parser.add_argument('model',
                            type=str,
                            help='The name or path of the model to be loaded')
        parser.add_argument(
            '--work-dir',
            type=str,
            default='./work_dir',
            help='The working directory for outputs. defaults to "./work_dir"')
        ArgumentHelper.calib_dataset(parser)
        ArgumentHelper.calib_samples(parser)
        ArgumentHelper.calib_seqlen(parser)
        ArgumentHelper.calib_batchsize(parser)
        ArgumentHelper.calib_search_scale(parser)
        ArgumentHelper.device(parser)

    @staticmethod
    def add_parser_kv_qparams():
        """Add parser for kv_qparams command."""
        parser = SubCliLite.subparsers.add_parser(
            'kv_qparams',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliLite.kv_qparams.__doc__,
            help=SubCliLite.kv_qparams.__doc__)
        parser.set_defaults(run=SubCliLite.kv_qparams)

        parser.add_argument('work_dir',
                            type=str,
                            help='Directory path where the stats are saved')
        parser.add_argument('turbomind_dir',
                            type=str,
                            help='Directory path where to save the results')
        parser.add_argument('--kv-bits',
                            type=int,
                            default=8,
                            help='Number of bits for quantization')
        parser.add_argument('--kv-sym',
                            action='store_true',
                            help='Whether to use symmetric quantizaiton')
        parser.add_argument(
            '--num-tp',
            type=int,
            default=None,
            help='GPU number used in tensor parallelism. Should be 2^n')
        parser.add_argument('--tm-params',
                            nargs='*',
                            default=None,
                            action=DictAction,
                            help='Used key-values pairs in xxx=yyy format'
                            ' to update the turbomind model weights'
                            ' config')

    @staticmethod
    def auto_awq(args):
        """Perform weight quantization using AWQ algorithm."""
        from lmdeploy.lite.apis.auto_awq import auto_awq
        kwargs = convert_args(args)
        auto_awq(**kwargs)

    @staticmethod
    def calibrate(args):
        """Perform calibration on a given dataset."""
        from lmdeploy.lite.apis.calibrate import calibrate
        kwargs = convert_args(args)
        calibrate(**kwargs)

    @staticmethod
    def kv_qparams(args):
        """Export key and value stats."""
        from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams
        kwargs = convert_args(args)
        run_kv_qparams(**kwargs)

    @staticmethod
    def smooth_quant(args):
        """Perform w8a8 quantization using SmoothQuant."""
        from lmdeploy.lite.apis.smooth_quant import smooth_quant
        kwargs = convert_args(args)
        smooth_quant(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        SubCliLite.add_parser_auto_awq()
        SubCliLite.add_parser_calibrate()
        SubCliLite.add_parser_kv_qparams()
        SubCliLite.add_parser_smooth_quant()
