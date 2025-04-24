# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

MIGRATION_BACKENDS = Registry('migration_backend', locations=['lmdeploy.disagg.backend.backend'])
