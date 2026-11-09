# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('module_name', ['profile_throughput', 'profile_pipeline_api', 'benchmark_guided'])
@pytest.mark.parametrize(('output_len', 'range_ratio', 'minimum'), [
    (4, 0.0, 1),
    (4, 0.01, 1),
    (4, 0.5, 2),
    (4, 1.0, 4),
    (1, 0.0, 1),
])
def test_random_requests_have_positive_output_lengths(module_name, output_len, range_ratio, minimum, tmp_path,
                                                      monkeypatch):
    module = importlib.import_module(f'benchmark.{module_name}')
    monkeypatch.setattr(np.random, 'randint', np.random.RandomState(0).randint)
    dataset_path = tmp_path / 'sharegpt.json'
    dataset_path.write_text(json.dumps([{
        'conversations': [{'value': 'hello'}, {'value': 'world'}]
    }] * 64))
    tokenizer = SimpleNamespace(encode=lambda text: [ord(char) for char in text],
                                decode=lambda tokens: ''.join(chr(token) for token in tokens))

    requests = module.sample_random_requests(input_len=4,
                                             output_len=output_len,
                                             num_prompts=64,
                                             range_ratio=range_ratio,
                                             tokenizer=tokenizer,
                                             dataset_path=str(dataset_path))

    assert len(requests) == 64
    assert all(minimum <= request[2] <= output_len for request in requests)
