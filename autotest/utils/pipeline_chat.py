from lmdeploy import pipeline
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)


class PipelineChat:

    def __init__(self, hf_path: str, tp: int = 1):
        self.hf_path = hf_path
        if 'w4' in hf_path:
            backend_config = TurbomindEngineConfig(tp=tp, model_format='awq')
        else:
            backend_config = TurbomindEngineConfig(tp=tp)
        pipe = pipeline(hf_path, backend_config=backend_config)
        self.pipe = pipe

    def default_pipeline_chat(self, prompt):
        gen_config = GenerationConfig(temperature=0.01)
        return self.pipe([prompt], gen_config=gen_config)[0]


class PipelinePytorchChat:

    def __init__(self, hf_path: str, tp: int = 1):
        self.hf_path = hf_path
        backend_config = PytorchEngineConfig(tp=tp)
        pipe = pipeline(hf_path, backend_config=backend_config)
        self.pipe = pipe

    def default_pipeline_chat(self, prompt):
        gen_config = GenerationConfig(temperature=0.01)
        return self.pipe([prompt], gen_config=gen_config)[0]
