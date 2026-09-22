# Copyright (c) OpenMMLab. All rights reserved.
from packaging.specifiers import SpecifierSet

from .base import BaseChecker

TRANSFORMERS_VERSION_SPEC = ('>=4.56.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0,!=5.7.*,!=5.8.*,!=5.9.*')


class TransformersChecker(BaseChecker):
    """Check transformers is available."""

    def check(self):
        """check."""
        import transformers
        logger = self.get_logger()
        try:
            if not SpecifierSet(TRANSFORMERS_VERSION_SPEC).contains(transformers.__version__):
                logger.warning(f'LMDeploy requires transformers{TRANSFORMERS_VERSION_SPEC}, '
                               f'but found transformers=={transformers.__version__}.')
        except Exception as e:
            self.log_and_exit(e, 'transformers', 'transformers is not available.')
