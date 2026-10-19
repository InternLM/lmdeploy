# ReplaySSM

ReplaySSM caches recent recurrent inputs and periodically replays them into a
state checkpoint, reducing full-state writes during decoding. In MTP mode it
also replaces per-position state snapshots with a checkpoint and input rings.

## Supported configurations

ReplaySSM is experimental and **disabled by default**. It supports:

- The PyTorch CUDA backend on SM90 GPUs, with colocated `Hybrid` engines.
- Standard decoding for Qwen3-Next/Qwen3.5 Gated DeltaNet and GLM-5.3 Flash KDA.
- Qwen3.5 MTP with `qwen3_5_mtp`, or GLM-5.3 Flash MTP with `deepseek_mtp`.

TurboMind, non-CUDA devices, P/D disaggregation, external KV-cache connectors,
and other model/speculative-method combinations are not supported. Unsupported
ReplaySSM configurations are rejected at startup.

## Usage

### Command line

Qwen3.5 standard decoding:

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 16
```

For Qwen3.5 MTP, add `--speculative-algorithm qwen3_5_mtp` and
`--speculative-num-draft-tokens 3`. A buffer length of either 8 or 16 meets
this MTP configuration's minimum size.

GLM-5.3 Flash with five draft tokens (adjust TP for your deployment):

```bash
lmdeploy serve api_server /path/to/glm5.3-flash-model \
  --backend pytorch \
  --tp 4 \
  --enable-replayssm \
  --replayssm-buffer-len 16 \
  --speculative-algorithm deepseek_mtp \
  --speculative-num-draft-tokens 5
```

### Python API

The equivalent GLM-5.3 Flash MTP configuration is:

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':
    backend_config = PytorchEngineConfig(
        tp=4,
        enable_replayssm=True,
        replayssm_buffer_len=16,
    )
    speculative_config = SpeculativeConfig(
        method='deepseek_mtp',
        num_speculative_tokens=5,
    )
    pipe = pipeline(
        '/path/to/glm5.3-flash-model',
        backend_config=backend_config,
        speculative_config=speculative_config,
    )
    try:
        print(pipe(['Hello!']))
    finally:
        pipe.close()
```

Omit `speculative_config` for standard decoding. There is no separate MTP
ReplaySSM switch; the mode follows the speculative configuration.

## Buffer length and tradeoffs

- `replayssm_buffer_len` accepts **8, 16, or 32** and defaults to **16**.
- For MTP, the length must satisfy
  `L >= 2 * (num_speculative_tokens + 1)`: MTP3 needs at least 8, and MTP5
  needs at least 16.
- Larger buffers flush less often but consume more memory and read more
  history per step; smaller buffers make the opposite tradeoff.
- Standard ReplaySSM retains a full checkpoint and adds rings, increasing
  state-cache memory. MTP ReplaySSM can save state-cache memory by replacing
  per-position snapshots, allowing a larger batch under a fixed memory budget.
- Neither mode guarantees a speedup or bitwise-identical output. Validate
  accuracy, acceptance rate, latency, and throughput on your workload; report
  same-batch and capacity-expanded comparisons separately.

To disable ReplaySSM, remove `--enable-replayssm` or set
`enable_replayssm=False`.

## References

- [ReplaySSM: Cache SSM Inputs, Not State](https://dao-lab.ai/blog/2026/replayssm/)
- [ReplaySSM reference implementation](https://github.com/Johnny-Liou/ReplaySSM)
