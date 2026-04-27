# yapf: disable

# yapf: enable


def test_ffn_reader_kind_none():
    """FFN readers must handle kind=None (returns filter list, not tensors).

    This is the probe call from Ffn.apply() to discover parameter keys before loading actual tensor data. A missing
    guard causes KeyError with 'None' in the key string (regression test for InternLM2Reader._ffn bug).
    """
    import re

    from lmdeploy.turbomind.deploy.source_model.internlm2 import InternLM2Reader
    from lmdeploy.turbomind.deploy.source_model.llama import LlamaReader

    # Create minimal readers with fake params that match ffn patterns
    fake_params = {
        'model.layers.0.mlp.gate_proj.weight': None,
        'model.layers.0.mlp.down_proj.weight': None,
        'model.layers.0.mlp.up_proj.weight': None,
        'model.layers.0.feed_forward.w1.weight': None,
        'model.layers.0.feed_forward.w2.weight': None,
        'model.layers.0.feed_forward.w3.weight': None,
    }

    # LlamaReader with kind=None should return filtered key list
    reader = LlamaReader.__new__(LlamaReader)
    reader.params = dict(fake_params)
    reader.ffn_pattern = r'mlp'
    result = reader._ffn(0, None)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(k, str) for k in result)
    assert all(re.search(r'mlp', k) for k in result)

    # InternLM2Reader with kind=None should also return filtered key list
    reader2 = InternLM2Reader.__new__(InternLM2Reader)
    reader2.params = dict(fake_params)
    reader2.fp8_quant = None
    reader2.ffn_pattern = r'feed_forward'
    result2 = reader2._ffn(0, None)
    assert isinstance(result2, list)
    assert len(result2) > 0
    assert all(isinstance(k, str) for k in result2)
    assert all(re.search(r'feed_forward', k) for k in result2)
