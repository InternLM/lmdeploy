# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs

TypeModelMetas = Dict[str, Any]

InputMultiModalType = List[Dict[str, Any]]


@dataclass
class PreprocessInputResult:
    """Results of preprocess input."""
    input_ids: List[int]
    input_multimodals: Optional[MultiModalInputs] = None
    model_metas: Optional[TypeModelMetas] = None


class BaseModelInputProcessor(ABC):
    """Processor of model inputs."""

    @abstractmethod
    def preprocess_input(self,
                         input_ids: List[int],
                         input_mms: InputMultiModalType = None,
                         **kwargs) -> PreprocessInputResult:
        """Preprocess input."""
        raise NotImplementedError('Not implemented.')


class DefaultModelInputProcessor(BaseModelInputProcessor):
    """Default model input processor."""

    def preprocess_input(self,
                         input_ids: List[int],
                         input_mms: MultiModalInputs = None,
                         **kwargs) -> PreprocessInputResult:
        """Preprocess input."""
        return PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=input_mms,
        )
