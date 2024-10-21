# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from typing import Iterator, Tuple

import torch
from safetensors import safe_open

# https://github.com/huggingface/transformers/blob/53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/modeling_utils.py#L372
WEIGHT_INDEX_NAME = 'pytorch_model.bin.index.json'
WEIGHT_PATTERN = 'pytorch_model*.bin'
SAFE_WEIGHT_INDEX_NAME = 'model.safetensors.index.json'
SAFE_WEIGHT_PATTERN = 'model*.safetensors'


class BaseLoader(ABC):

    def __init__(self, model_path: str, pattern):
        self.model_path = model_path
        self.pattern = pattern
        self.item_count = defaultdict(int)

    def get_index(self, index_path: str,
                  file_pattern: str) -> Tuple[dict, list]:
        """get shards and weight map (if possible) for the model."""
        index_path = osp.join(self.model_path, index_path)
        if osp.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
                index = index['weight_map']
                shards = set(index.values())
            shards = [osp.join(self.model_path, x) for x in shards]
        else:
            index = {}
            file_pattern = osp.join(self.model_path, file_pattern)
            shards = glob(file_pattern)
        return sorted(shards), index

    @abstractmethod
    def items(self) -> Iterator[Tuple[int, dict]]:
        pass


class SafetensorsLoader(BaseLoader):

    def __init__(self, model_path: str, pattern: str):
        super().__init__(model_path, pattern)
        self.pattern = pattern
        self.model_path = model_path
        self.shards, index = self.get_index(SAFE_WEIGHT_INDEX_NAME,
                                            SAFE_WEIGHT_PATTERN)
        if not index:
            for shard in self.shards:
                with safe_open(shard, 'pt') as f:
                    index.update({k: shard for k in f.keys()})
        # count layer-wise parameters
        for k in index.keys():
            match = re.findall(self.pattern, k)
            if match:
                self.item_count[int(match[0])] += 1

    def items(self):
        params = defaultdict(dict)
        for shard in self.shards:
            with safe_open(shard, 'pt') as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    match = re.findall(self.pattern, k)
                    if not match:
                        yield (-1, {k: tensor})
                    else:
                        idx = int(match[0])
                        param = params[idx]
                        param[k] = tensor
                        if len(param) == self.item_count[idx]:
                            yield (idx, params.pop(idx))
        assert not params


class PytorchLoader(BaseLoader):

    def __init__(self, model_path: str, pattern: str):
        super().__init__(model_path, pattern)
        self.shards, index = self.get_index(WEIGHT_INDEX_NAME, WEIGHT_PATTERN)
        for k in index.keys():
            match = re.findall(self.pattern, k)
            if match:
                self.item_count[int(match[0])] += 1

    def items(self):
        params = defaultdict(dict)
        for shard in self.shards:
            misc = {}
            tmp = torch.load(shard, map_location='cpu')
            for k, v in tmp.items():
                match = re.findall(self.pattern, k)
                if not match:
                    misc[k] = v
                else:
                    idx = int(match[0])
                    params[idx][k] = v
            del tmp
            if misc:
                yield (-1, misc)
                misc.clear()
            ready = []
            if self.item_count:
                for idx, param in params.items():
                    if len(param) == self.item_count[idx]:
                        ready.append(idx)
            else:
                ready = sorted(params.keys())[:-1]
            for idx in ready:
                yield (idx, params.pop(idx))
        idxs = sorted(params.keys())
        for idx in idxs:
            yield (idx, params.pop(idx))


def create_loader(model_path: str, pattern: str) -> BaseLoader:
    cls = None
    if osp.exists(osp.join(model_path, SAFE_WEIGHT_INDEX_NAME)):
        cls = SafetensorsLoader
    elif glob(osp.join(model_path, SAFE_WEIGHT_PATTERN)):
        cls = SafetensorsLoader
    elif osp.exists(osp.join(model_path, WEIGHT_INDEX_NAME)):
        cls = PytorchLoader
    elif glob(osp.join(model_path, WEIGHT_PATTERN)):
        cls = PytorchLoader
    assert cls is not None, f'Failed to find valid loader for {model_path}'
    return cls(model_path, pattern)
