# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs

TypeModelMetas = dict[str, Any]

InputMultiModalType = list[dict[str, Any]]


@dataclass
class PreprocessInputResult:
    """Results of preprocess input."""
    input_ids: list[int]
    input_multimodals: MultiModalInputs | None = None
    model_metas: TypeModelMetas | None = None


class BaseModelInputProcessor(ABC):
    """Processor of model inputs."""

    @abstractmethod
    def preprocess_input(self,
                         input_ids: list[int],
                         input_mms: InputMultiModalType = None,
                         **kwargs) -> PreprocessInputResult:
        """Preprocess input."""
        raise NotImplementedError('Not implemented.')


class DefaultModelInputProcessor(BaseModelInputProcessor):
    """Default model input processor."""

    def preprocess_input(self,
                         input_ids: list[int],
                         input_mms: MultiModalInputs = None,
                         **kwargs) -> PreprocessInputResult:
        """Preprocess input."""
        return PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=input_mms,
        )
