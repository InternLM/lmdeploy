import time

from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.vl import load_image

backend_config = PytorchEngineConfig(session_len=16384)
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
llm = pipeline('Qwen/Qwen2.5-VL-3B-Instruct', backend_config=backend_config)

start_time = time.time()
response = llm(('describe this image', image))
print(response)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
