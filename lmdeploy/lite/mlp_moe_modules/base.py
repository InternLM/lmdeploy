# Copyright (c) OpenMMLab. All rights reserved.

from mmengine import Registry

CONVERT_MOE_MODELS = Registry('mlp moe module', locations=['lmdeploy.lite.mlp_moe_modules.base'])
