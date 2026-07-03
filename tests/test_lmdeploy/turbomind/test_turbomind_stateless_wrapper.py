import inspect

from lmdeploy.turbomind.turbomind import TurboMindInstance


def test_turbomind_instance_stream_signature_is_stateless():
    params = inspect.signature(TurboMindInstance.async_stream_infer).parameters

    assert 'sequence_start' not in params
    assert 'sequence_end' not in params
    assert 'step' not in params
