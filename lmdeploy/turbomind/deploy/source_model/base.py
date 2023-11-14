# Copyright (c) OpenMMLab. All rights reserved.
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple, Union

import torch
from mmengine import Registry

INPUT_MODELS = Registry(
    'source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


class BaseReader(ABC):
    """Base checkpoint manager."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def start_layer_id(self) -> int:
        """Get the start transformer layer number."""
        pass

    @property
    @abstractmethod
    def end_layer_id(self) -> int:
        """Get the end transformer layer number."""
        pass

    @abstractmethod
    def init_layer_id(self) -> None:
        """Get start and end transformer layer number."""
        self._start_layer_id = -1
        self._end_layer_id = -1
        layer_count = {}
        for key in self.params:
            layer_id = re.findall(self.attn_layer_patten, key)
            if len(layer_id) == 0:
                continue
            layer_id = int(layer_id[0])
            if layer_id not in layer_count:
                layer_count[layer_id] = 0
            layer_count[layer_id] += 1
        if len(layer_count) == 0:
            return
        if not (len(layer_count) > 1 or self.last_bin):
            return
        max_count = max([layer_count[layer_id] for layer_id in layer_count])
        valid_layer_id = [
            layer_id for layer_id in layer_count
            if layer_count[layer_id] == max_count
        ]
        self._start_layer_id = min(valid_layer_id)
        self._end_layer_id = max(valid_layer_id) + 1

    @abstractmethod
    def clean_up(self, last: bool) -> None:
        """Clean up unused params."""
        if last:
            self.params.clear()
        else:
            to_remove = []
            for key in self.params:
                layer_id = re.findall(self.attn_layer_patten, key)
                if len(layer_id) == 0:
                    # tok, norm, output
                    to_remove.append(key)
                else:
                    layer_id = int(layer_id[0])
                    if layer_id < self.end_layer_id:
                        to_remove.append(key)
            for key in to_remove:
                self.params.pop(key, None)
        torch.cuda.empty_cache()

    @abstractmethod
    def tok_embeddings(self) -> Union[torch.Tensor, None]:
        """Get embeddings."""
        pass

    @abstractmethod
    def norm_weight(self) -> Union[torch.Tensor, None]:
        """Get norm."""
        pass

    @abstractmethod
    def output_weight(self) -> Union[torch.Tensor, None]:
        """Get output."""
        pass

    @abstractmethod
    def attn(self, i: int) -> Tuple[torch.Tensor]:
        """Get q, k, v, o weight for layer i."""
        pass

    @abstractmethod
    def attn_bias(self, i: int) -> Tuple[torch.Tensor, None]:
        """Get q, k, v, o bias for layer i."""
        pass

    @abstractmethod
    def attn_zero(self, i: int) -> Tuple[torch.Tensor, None]:
        """Get q, k, v, o zero point for layer i."""
        pass

    @abstractmethod
    def attn_scale(self, i: int) -> Tuple[torch.Tensor, None]:
        """Get q, k, v, o scale for layer i."""
        pass

    @abstractmethod
    def attn_norm(self, i: int) -> torch.Tensor:
        """Get attn norm for layer i."""
        pass

    @abstractmethod
    def ffn(self, i: int) -> Tuple[torch.Tensor]:
        """Get ffn weight for layer i."""
        pass

    @abstractmethod
    def ffn_zero(self, i: int) -> Tuple[torch.Tensor, None]:
        """Get ffn zero point for layer i."""
        pass

    @abstractmethod
    def ffn_scale(self, i: int) -> Tuple[torch.Tensor, None]:
        """Get ffn scale for layer i."""
        pass

    @abstractmethod
    def ffn_norm(self, i: int) -> torch.Tensor:
        """Get ffn norm for layer i."""
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

    @property
    @abstractmethod
    def nmgrs(self) -> int:
        """Get number of checkpoint."""
        pass

    @abstractmethod
    def get_mgrs(self) -> Iterator[BaseReader]:
        """Conctruct all BaseReader."""
        pass

    @abstractmethod
    def tokenizer_info(self):
        """Read tokenizer info."""
        pass

    @abstractmethod
    def model_info(self) -> Dict:
        """Read model info."""
        pass

    def bins(self) -> Iterator[BaseReader]:
        """Get Reader."""
        for mgr in self.get_mgrs():
            yield mgr
