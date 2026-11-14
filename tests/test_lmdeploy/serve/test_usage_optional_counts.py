# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.openai.protocol import CompletionTokensDetails, UsageInfo


def test_unavailable_optional_counts_are_omitted_instead_of_null():
    usage = UsageInfo.build(10, 4, reasoning_tokens=2).model_dump()
    assert usage['completion_tokens_details'] == {'reasoning_tokens': 2}


def test_measured_zero_counts_are_retained():
    details = CompletionTokensDetails(accepted_prediction_tokens=0, rejected_prediction_tokens=0, audio_tokens=0)
    assert details.model_dump() == dict(reasoning_tokens=0, accepted_prediction_tokens=0,
                                        rejected_prediction_tokens=0, audio_tokens=0)
