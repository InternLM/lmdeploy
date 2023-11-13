# Copyright (c) OpenMMLab. All rights reserved.
import copy

from transformers import PretrainedConfig

from lmdeploy.turbomind.deploy.target_model.base import TurbomindModelConfig
from lmdeploy.version import __version__ as lm_version


class LmdeployConfig(PretrainedConfig):

    def __init__(self, turbomind: dict = None, **kwargs):
        default_tm_cfg = copy.deepcopy(
            TurbomindModelConfig.from_dict({}, allow_none=True).__dict__)
        if turbomind is not None:
            default_tm_cfg.update(turbomind)
        self.turbomind = default_tm_cfg
        self.lmdeploy_version = lm_version
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        config, kwargs = super().from_pretrained(pretrained_model_name_or_path,
                                                 return_unused_kwargs=True,
                                                 **kwargs)
        for k, v in kwargs.items():
            if k in config.turbomind.keys():
                config.turbomind[k] = v
        if 'tp' in kwargs:
            config.turbomind['tensor_para_size'] = kwargs['tp']
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
