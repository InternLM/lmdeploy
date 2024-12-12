# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version

from .base import BaseChecker


class ModelChecker(BaseChecker):
    """check model is available."""

    def __init__(self,
                 model_path: str,
                 trust_remote_code: bool,
                 dtype: str,
                 device_type: str,
                 logger=None) -> None:
        super().__init__(logger=logger)
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.device_type = device_type
        self.dtype = dtype

    def check_config(self, trans_version):
        """check config."""
        model_path = self.model_path
        trust_remote_code = self.trust_remote_code
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            message = (
                f'Load model config with transformers=={trans_version}'
                ' failed. '
                'Please make sure model can be loaded with transformers API.')
            self.log_and_exit(e, 'transformers', message=message)
        return config

    def check_trans_version(self, config, trans_version):
        """check transformers version."""
        model_path = self.model_path
        try:
            model_trans_version = getattr(config, 'transformers_version', None)
            if model_trans_version is not None:
                model_trans_version = version.parse(model_trans_version)
                assert trans_version >= model_trans_version, (
                    'Version mismatch.')
        except Exception as e:
            message = (f'model `{model_path}` requires '
                       f'transformers version {model_trans_version} '
                       f'but transformers {trans_version} is installed.')
            self.log_and_exit(e, 'transformers', message=message)

    def check_dtype(self, config):
        """check dtype."""
        logger = self.get_logger()
        model_path = self.model_path
        device_type = self.device_type
        dtype = self.dtype
        try:
            import torch

            from lmdeploy.pytorch.config import ModelConfig
            from lmdeploy.utils import is_bf16_supported
            model_config = ModelConfig.from_hf_config(config,
                                                      model_path=model_path,
                                                      dtype=dtype)
            if model_config.dtype == torch.bfloat16:
                if not is_bf16_supported(device_type):
                    logger.warning('Device does not support bfloat16.')
        except Exception as e:
            message = (f'Checking failed with error {e}',
                       'Please send issue to LMDeploy with error logs.')
            self.log_and_exit(e, 'Model', message=message)

    def check_awq(self, config):
        """check awq."""
        logger = self.get_logger()
        device_type = self.device_type
        if device_type != 'cuda':
            return

        quantization_config = getattr(config, 'quantization_config', dict())
        quant_method = quantization_config.get('quant_method', None)
        if quant_method != 'awq':
            return
        try:
            import awq  # noqa
        except Exception as e:
            self.log_and_exit(e, 'autoawq', logger)

        try:
            import awq_ext  # noqa
        except Exception as e:
            logger.debug('Exception:', exc_info=1)
            self.log_and_exit(
                e,
                'awq_ext',
                message='Failed to import `awq_ext`. '
                'Try reinstall it from source: '
                'https://github.com/casper-hansen/AutoAWQ_kernels')

    def check(self):
        """check."""
        import transformers
        trans_version = version.parse(transformers.__version__)

        # config
        config = self.check_config(trans_version)

        # transformers version
        self.check_trans_version(config, trans_version)

        # dtype check
        self.check_dtype(config)

        # awq
        self.check_awq(config)
