# Copyright (c) OpenMMLab. All rights reserved.

from mmengine import Registry

CONVERT_MOE_MODELS = Registry('moe_mlp_module', locations=['lmdeploy.lite.moe_mlp_modules'])
