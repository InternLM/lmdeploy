# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from glob import glob
from queue import Queue
from typing import Dict, Iterator, Tuple, Union

import torch
from safetensors import safe_open

# https://github.com/huggingface/transformers/blob/53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/modeling_utils.py#L372
WEIGHT_INDEX_NAME = 'pytorch_model.bin.index.json'
WEIGHT_PATTERN = 'pytorch_model*.bin'
SAFE_WEIGHT_INDEX_NAME = 'model.safetensors.index.json'
SAFE_WEIGHT_PATTERN = 'model*.safetensors'
EXTRA_WEIGHT_PATTERNS = ['*.pt', '*.bin']
EXTRA_SAFE_WEIGHT_PATTERN = '*.safetensors'


class BaseLoader(ABC):

    def __init__(self, model_path: str, pattern):
        self.model_path = model_path
        self.pattern = pattern
        self.item_count = defaultdict(int)

    def get_index(self, index_name: str, file_pattern: str) -> Tuple[dict, list]:
        """Get shards and weight map (if possible) for the model."""
        get_path = partial(osp.join, self.model_path)
        shards = []
        if index_name:
            with open(get_path(index_name), 'r') as f:
                index = json.load(f)
            index = index['weight_map']
            shards = list(map(get_path, set(index.values())))
        else:
            index = {}
            shards = glob(get_path(file_pattern))
        if not shards:
            raise RuntimeError(f'failed to locate weight files for {self.model_path}')
        return sorted(shards), index

    @abstractmethod
    def items(self) -> Iterator[Tuple[int, dict]]:
        pass


class SafetensorsLoader(BaseLoader):

    def __init__(self, model_path: str, pattern: str, index_name=None, file_pattern=None):
        super().__init__(model_path, pattern)
        self.shards, index = self.get_index(index_name, file_pattern)
        if not index:
            for shard in self.shards:
                with safe_open(shard, 'pt') as f:
                    index.update({k: shard for k in f.keys()})
        # self.index maps weight names to their containing safetensors file paths
        self.index = index
        # count layer-wise parameters
        for k in index.keys():
            match = re.findall(self.pattern, k)
            if match:
                self.item_count[int(match[0])] += 1

    def items(self):
        params = defaultdict(dict)
        for shard in self.shards:
            filename = osp.split(shard)[-1]
            with safe_open(shard, 'pt') as f:
                misc = []
                for k in f.keys():
                    # Filtering logic:
                    # - Exclude weights not found in the mapping
                    # - Exclude duplicated weights (present in multiple files)
                    if k not in self.index or self.index[k] != filename:
                        continue
                    match = re.findall(self.pattern, k)
                    if not match:
                        misc.append(k)
                    else:
                        idx = int(match[0])
                        param = params[idx]
                        param[k] = f.get_tensor(k)
                        if len(param) == self.item_count[idx]:
                            yield (idx, params.pop(idx))
                if misc:
                    yield (-1, {k: f.get_tensor(k) for k in misc})
        assert not params


class PytorchLoader(BaseLoader):

    def __init__(self, model_path: str, pattern: str, index_name=None, file_pattern=None):
        super().__init__(model_path, pattern)
        self.shards, index = self.get_index(index_name, file_pattern)
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


class StateDictLoader:
    """This loader is used for `update_params`.

    Currently, the item in the queue should be full state dict of a decoder layer or the meta of the model (embedding,
    lm_head, norm).
    """

    def __init__(self, queue: Queue[Dict[str, torch.Tensor]], pattern: str):
        self.que = queue
        self.pattern = pattern

    def items(self):
        for data in iter(self.que.get, None):
            # If data is state dict of a decoder layer, any key will match the pattern.
            # Otherwise, none of the keys will match the pattern.
            for k in data.keys():
                match = re.findall(self.pattern, k)
                break

            if not match:
                yield (-1, data)
            else:
                idx = int(match[0])
                yield (idx, data)

            torch.cuda.empty_cache()
            self.que.task_done()


def create_loader(model_path: Union[str, Queue], pattern: str) -> BaseLoader:
    args = (model_path, pattern)

    if isinstance(model_path, Queue):
        # used for `update_params`
        return StateDictLoader(*args)

    if osp.exists(osp.join(model_path, SAFE_WEIGHT_INDEX_NAME)):
        return SafetensorsLoader(*args, index_name=SAFE_WEIGHT_INDEX_NAME)

    if glob(osp.join(model_path, SAFE_WEIGHT_PATTERN)):
        return SafetensorsLoader(*args, file_pattern=SAFE_WEIGHT_PATTERN)

    if osp.exists(osp.join(model_path, WEIGHT_INDEX_NAME)):
        return PytorchLoader(*args, index_name=WEIGHT_INDEX_NAME)

    if glob(osp.join(model_path, WEIGHT_PATTERN)):
        return PytorchLoader(*args, file_pattern=WEIGHT_PATTERN)

    # non-standard safetensors model (*.safetensors)
    if glob(osp.join(model_path, EXTRA_SAFE_WEIGHT_PATTERN)):
        return SafetensorsLoader(*args, file_pattern=EXTRA_SAFE_WEIGHT_PATTERN)

    # non-standard pytorch model (*.bin, *.pt)
    for p in EXTRA_WEIGHT_PATTERNS:
        if glob(osp.join(model_path, p)):
            return PytorchLoader(*args, file_pattern=p)

    raise RuntimeError(f'Failed to find valid loader for {model_path}')
