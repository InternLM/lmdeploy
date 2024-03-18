import pytest
from utils.get_run_config import get_tp_num

from lmdeploy import TurbomindEngineConfig, pipeline


@pytest.mark.order(8)
@pytest.mark.pipeline_func
@pytest.mark.timeout(600)
class TestPipelineLongtextFunc:

    def test_long_test_chat_7b(self, config):
        model = 'internlm/internlm2-chat-7b'
        tp_config = get_tp_num(config, model)
        model_path = '/'.join([config.get('model_path'), model])

        backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                               session_len=210000,
                                               tp=tp_config)
        pipe = pipeline(model_path, backend_config=backend_config)
        prompt = '今 天 心 ' * int(200000 / 6)

        # batch infer
        pipe(prompt)

        # stream infer
        for outputs in pipe.stream_infer(prompt):
            continue

        prompts = ['今 天 心 ' * int(200000 / 6)] * 2
        # batch infer
        pipe(prompts)

        # stream infer
        for outputs in pipe.stream_infer(prompts):
            continue

    def test_long_test_chat_20b(self, config):
        model = 'internlm/internlm2-chat-20b'
        tp_config = get_tp_num(config, model)
        model_path = '/'.join([config.get('model_path'), model])

        backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                               session_len=210000,
                                               tp=tp_config)
        pipe = pipeline(model_path, backend_config=backend_config)
        prompt = '今 天 心 ' * int(200000 / 6)

        # batch infer
        pipe(prompt)

        # stream infer
        for outputs in pipe.stream_infer(prompt):
            continue

        prompts = ['今 天 心 ' * int(200000 / 6)] * 2
        # batch infer
        pipe(prompts)

        # stream infer
        for outputs in pipe.stream_infer(prompts):
            continue

    def test_long_test_20b(self, config):
        model = 'internlm/internlm2-20b'
        tp_config = get_tp_num(config, model)
        model_path = '/'.join([config.get('model_path'), model])

        backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                               session_len=210000,
                                               tp=tp_config)

        pipe = pipeline(model_path, backend_config=backend_config)
        prompt = '今 天 心 ' * int(200000 / 6)

        # batch infer
        pipe(prompt)

        # stream infer
        for outputs in pipe.stream_infer(prompt):
            continue

        prompts = ['今 天 心 ' * int(200000 / 6)] * 2
        # batch infer
        pipe(prompts)

        # stream infer
        for outputs in pipe.stream_infer(prompts):
            continue
