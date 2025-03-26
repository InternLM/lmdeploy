# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Iterator, Union

import torch
from mmengine import Registry

INPUT_MODELS = Registry('source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


class BaseReader(ABC):
    """Mapping between TM modules and source modules."""

    def __init__(self):
        pass

    def transform(self, x: Union[torch.Tensor, None], kind: str) -> Union[torch.Tensor, None]:
        return None if x is None else self._transform(x, kind)

    @abstractmethod
    def _transform(self, x: torch.Tensor, kind: str):
        """Transform x."""
        pass


class BaseInputModel(ABC):
    """Base class for input model."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        """Constructor for BaseInputModel.

        Args:
            model_path (str): the path of the model.
            tokenizer_path (str): the path of the tokenizer model.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    @abstractmethod
    def model_info(self) -> Dict:
        """Read model info."""
        pass

    @abstractmethod
    def readers(self) -> Iterator[BaseReader]:
        pass
