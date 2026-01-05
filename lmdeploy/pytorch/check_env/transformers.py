# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version

from .base import BaseChecker

MIN_TRANSFORMERS_VERSION = '4.33.0'
MAX_TRANSFORMERS_VERSION = '4.57.3'


class TransformersChecker(BaseChecker):
    """Check transformers is available."""

    def check(self):
        """check."""
        import transformers
        logger = self.get_logger()
        try:
            trans_version = version.parse(transformers.__version__)
            min_version = version.parse(MIN_TRANSFORMERS_VERSION)
            max_version = version.parse(MAX_TRANSFORMERS_VERSION)
            if trans_version < min_version or trans_version > max_version:
                logger.warning('LMDeploy requires transformers version: '
                               f'[{MIN_TRANSFORMERS_VERSION} ~ '
                               f'{MAX_TRANSFORMERS_VERSION}], '
                               'but found version: '
                               f'{transformers.__version__}')
        except Exception as e:
            self.log_and_exit(e, 'transformers', 'transformers is not available.')
