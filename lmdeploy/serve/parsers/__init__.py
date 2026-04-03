# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

from .response_parser import ResponseParser

ResponseParserManager = Registry('response_parser', locations=['lmdeploy.serve.parsers.response_parser'])

__all__ = ['ResponseParser', 'ResponseParserManager']
