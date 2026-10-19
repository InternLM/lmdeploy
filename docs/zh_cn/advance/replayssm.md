# ReplaySSM

ReplaySSM 缓存近期循环状态的输入，并周期性地将它们重放到 checkpoint，减少
解码时完整状态的写回。MTP 模式还使用 checkpoint 和输入 ring 替代逐位置状态快照。

## 支持范围

ReplaySSM 是实验性功能，**默认关闭**。当前支持：

- SM90 GPU 上的 PyTorch CUDA backend，以及同机 `Hybrid` engine。
- Qwen3-Next/Qwen3.5 Gated DeltaNet 和 GLM-5.3 Flash KDA 的标准解码。
- Qwen3.5 的 `qwen3_5_mtp`，以及 GLM-5.3 Flash 的 `deepseek_mtp`。

不支持 TurboMind、非 CUDA 设备、P/D 解耦、外部 KV-cache connector，以及其他
模型或投机算法组合。不支持的 ReplaySSM 配置会在启动时报错。

## 使用方式

### 命令行

Qwen3.5 标准解码：

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 16
```

Qwen3.5 MTP 额外添加 `--speculative-algorithm qwen3_5_mtp` 和
`--speculative-num-draft-tokens 3`；buffer 长度 8 或 16 均满足该配置的最小长度要求。

GLM-5.3 Flash，使用五个 draft token（按部署环境调整 TP）：

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

等价的 GLM-5.3 Flash MTP 配置：

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

标准解码省略 `speculative_config` 即可。无需单独设置 MTP ReplaySSM 开关，模式
由投机解码配置决定。

## Buffer 长度与取舍

- `replayssm_buffer_len` 支持 **8、16、32**，默认 **16**。
- MTP 要求 `L >= 2 * (num_speculative_tokens + 1)`：MTP3 至少需要 8，
  MTP5 至少需要 16。
- 较大的 buffer 降低 flush 频率，但占用更多显存、每步读取更多历史；较小的
  buffer 则相反。
- 标准 ReplaySSM 保留完整 checkpoint 并新增 ring，会增加状态缓存显存。
  MTP ReplaySSM 通过替代逐位置快照，可节省状态缓存显存，使固定显存预算下
  使用更大的 batch 成为可能。
- 两种模式都不保证加速或逐位一致。应针对实际负载验证精度、接受率、延迟和
  吞吐，并分别报告同 batch 对照与扩容对照。

移除 `--enable-replayssm` 或设置 `enable_replayssm=False` 即可关闭该功能。

## 参考资料

- [ReplaySSM: Cache SSM Inputs, Not State](https://dao-lab.ai/blog/2026/replayssm/)
- [ReplaySSM 参考实现](https://github.com/Johnny-Liou/ReplaySSM)
