# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version

from .base import BaseChecker


class ModelChecker(BaseChecker):
    """Check model is available."""

    def __init__(self, model_path: str, trust_remote_code: bool, dtype: str, device_type: str, logger=None) -> None:
        super().__init__(logger=logger)
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.device_type = device_type
        self.dtype = dtype

    def check_config(self, trans_version):
        """Check config."""
        model_path = self.model_path
        trust_remote_code = self.trust_remote_code
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            message = (f'Load model config with transformers=={trans_version}'
                       ' failed. '
                       'Please make sure model can be loaded with transformers API.')
            self.log_and_exit(e, 'transformers', message=message)
        return config

    def check_trans_version(self, config, trans_version):
        """Check transformers version."""
        model_path = self.model_path
        logger = self.get_logger()
        model_trans_version = getattr(config, 'transformers_version', None)
        if model_trans_version is not None:
            model_trans_version = version.parse(model_trans_version)
            if trans_version < model_trans_version:
                message = (f'model `{model_path}` requires '
                           f'transformers version {model_trans_version} '
                           f'but transformers {trans_version} is installed.')
                logger.warning(message)

    def check_dtype(self, config):
        """Check dtype."""
        logger = self.get_logger()
        model_path = self.model_path
        device_type = self.device_type
        dtype = self.dtype
        try:
            import torch

            from lmdeploy.pytorch.config import ModelConfig
            from lmdeploy.utils import is_bf16_supported
            model_config = ModelConfig.from_hf_config(config, model_path=model_path, dtype=dtype)
            if model_config.dtype == torch.bfloat16:
                if not is_bf16_supported(device_type):
                    logger.warning('Device does not support bfloat16.')
        except Exception as e:
            message = (f'Checking failed with error {e}. Please send issue to LMDeploy with error logs.')
            self.log_and_exit(e, 'Model', message=message)

        try:
            model_config.check_env_func(device_type)
        except Exception as e:
            message = (f'Checking failed with error {e}.')
            self.log_and_exit(e, 'Model', message=message)

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
