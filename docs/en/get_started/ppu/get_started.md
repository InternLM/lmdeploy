# Get Started with PPU

The usage of lmdeploy on a ppu device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

## Installation

Please refer to [dlinfer installation guide](https://github.com/DeepLink-org/dlinfer#%E5%AE%89%E8%A3%85%E6%96%B9%E6%B3%95).

## Offline batch inference

> \[!TIP\]
> Graph mode is supported on ppu.
> Users can set `eager_mode=False` to enable graph mode, or set `eager_mode=True` to disable graph mode.

### LLM inference

Set `device_type="ppu"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    pipe = pipeline("internlm/internlm2_5-7b-chat",
                    backend_config=PytorchEngineConfig(tp=1, device_type="ppu", eager_mode=True))
    question = ['Hi, pls intro yourself', 'Shanghai is']
    response = pipe(question)
    print(response)
```

### VLM inference

Set `device_type="ppu"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
if __name__ == "__main__":
    pipe = pipeline('OpenGVLab/InternVL2-2B',
                    backend_config=PytorchEngineConfig(tp=1, device_type='ppu', eager_mode=True))
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```

## Online serving

> \[!TIP\]
> Graph mode is supported on ppu.
> Graph mode is default enabled in online serving. Users can add `--eager-mode` to disable graph mode.

### Serve an LLM model

Add `--device ppu` in the serve command.

```bash
lmdeploy serve api_server --backend pytorch --device ppu --eager-mode internlm/internlm2_5-7b-chat
```

### Serve a VLM model

Add `--device ppu` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device ppu --eager-mode OpenGVLab/InternVL2-2B
```

## Inference with Command line Interface

Add `--device ppu` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ppu --eager-mode
```
