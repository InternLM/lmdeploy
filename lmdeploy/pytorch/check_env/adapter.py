# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseChecker


class AdapterChecker(BaseChecker):
    """Check adapter is available."""

    def __init__(self, adapter_path: str, logger=None):
        super().__init__(logger)
        self.adapter_path = adapter_path

    def check(self):
        """check."""
        path = self.adapter_path

        try:
            import peft  # noqa: F401
        except Exception as e:
            self.log_and_exit(e, 'Adapter', message='Failed to import peft.')

        try:
            from peft import PeftConfig
            PeftConfig.from_pretrained(path)
        except Exception as e:
            message = ('Please make sure the adapter can be loaded with '
                       '`peft.PeftConfig.from_pretrained`\n')
            err_msg = '' if len(e.args) == 0 else e.args[0]
            if 'got an unexpected keyword argument' in err_msg:
                message += ('Or try remove all unexpected keywords '
                            'in `adapter_config.json`.')
            self.log_and_exit(e, 'Adapter', message=message)
