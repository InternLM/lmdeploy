# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Registry

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.openai.reasoning_parser'])


@ReasoningParserManager.register_module(name=[
    'qwen-qwq', 'qwen3', 'intern-s1', 'deepseek-r1',
    'deepseek-v3'
])
class ReasoningParser:
    """Unified reasoning parser for all ``--reasoning-parser`` options."""

    def __init__(self, tokenizer: object, **kwargs):
        self.model_tokenizer = tokenizer

    def get_reasoning_open_tag(self) -> str | None:
        return '<think>'

    def get_reasoning_close_tag(self) -> str | None:
        return '</think>'

    def starts_in_reasoning_mode(self) -> bool:
        return True
