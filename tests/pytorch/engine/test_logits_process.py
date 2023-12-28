import pytest
import torch
from transformers.generation.logits_process import (LogitsProcessorList,
                                                    TemperatureLogitsWarper,
                                                    TopKLogitsWarper,
                                                    TopPLogitsWarper)

from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor
from lmdeploy.pytorch.messages import SamplingParam


class TestFusedLogitsProcessor:

    @pytest.fixture
    def scores(self):
        yield torch.rand(4, 100).cuda().half()

    @pytest.fixture
    def input_ids(self):
        yield None

    @pytest.fixture
    def sampling_param(self):
        yield SamplingParam(top_k=20, top_p=0.8, temperature=0.8)

    @pytest.fixture
    def gt(self, input_ids, scores, sampling_param):
        logits_processor = LogitsProcessorList([
            TopKLogitsWarper(sampling_param.top_k),
            TopPLogitsWarper(sampling_param.top_p),
            TemperatureLogitsWarper(sampling_param.temperature),
        ])
        yield logits_processor(input_ids, scores)

    def test_processor(self, input_ids, scores, sampling_param, gt):
        logits_processor = FusedLogitsProcessor(sampling_param)
        output = logits_processor(input_ids, scores)

        torch.testing.assert_close(output, gt)
