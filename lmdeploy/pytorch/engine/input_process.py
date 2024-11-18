# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs

TypeModelMetas = Dict[str, Any]

InputMultiModalType = List[Dict[str, Any]]


@dataclass
class PreprocessInputResult:
    """results of preprocess input."""
    input_ids: List[int]
    input_multimodals: Optional[MultiModalInputs] = None
    model_metas: Optional[TypeModelMetas] = None


class BaseModelInputProcessor(ABC):
    """processor of model inputs."""

    @abstractmethod
    def preprocess_input(self,
                         input_ids: List[int],
                         input_mms: InputMultiModalType = None,
                         **kwargs) -> PreprocessInputResult:
        """preprocess input."""
        raise NotImplementedError('Not implemented.')


class DefaultModelInputProcessor(BaseModelInputProcessor):
    """default model input processor."""

    def preprocess_input(self,
                         input_ids: List[int],
                         input_mms: MultiModalInputs = None,
                         **kwargs) -> PreprocessInputResult:
        """preprocess input."""
        return PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=input_mms,
        )
