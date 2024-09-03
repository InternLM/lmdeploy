# lmdeploy.pytorch 新模型支持

lmdeploy.pytorch 被设计用来简化新模型的支持以及原型的开发，用户可以根据自己的需求适配新的模型。

## 模型支持

### 配置加载（可选）

lmdeploy.pytorch 会根据模型的参数初始化引擎，如果需要接入的模型的参数命名与 transformers 中常见模型不同，可能存在解析错误的情况。可以添加自定义的 ConfigBuilder 来解析配置

```python
# lmdeploy/pytorch/configurations/gemma.py

from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class GemmaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        # 判断 hf_config 是否适配该 builder
        return hf_config.model_type in ['gemma', 'gemma2']

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        # 使用 transformers 加载的 hf_config
        # 构造 pytorch engine 的 ModelConfig
        return ModelConfig(hidden_size=hf_config.hidden_size,
                           num_layers=hf_config.num_hidden_layers,
                           num_attention_heads=hf_config.num_attention_heads,
                           num_key_value_heads=hf_config.num_key_value_heads,
                           bos_token_id=hf_config.bos_token_id,
                           eos_token_id=hf_config.eos_token_id,
                           head_dim=hf_config.head_dim,
                           vocab_size=hf_config.vocab_size)
```

可以使用 `lmdeploy.pytorch.check_env.check_model` 函数验证配置是否能够正确解析

### 实现模型

在确保能够正确解析配置后，就可以开始实现模型逻辑。以 llama 的实现为例，我们需要通过 transformers 的配置文件创建模型

```python
class LlamaForCausalLM(nn.Module):

    # 构造函数，通过传入的 config 搭建模型
    # ctx_mgr 是上下文管理器，可以通过它传入引擎配置或额外参数
    def __init__(self,
                 config: LlamaConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build LLamaModel
        self.model = LlamaModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    # 模型推理函数
    # 推荐尽可能使用与下面相同的参数
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits
```

除了这些以外，还有如下内容需要添加

```python
class LlamaForCausalLM(nn.Module):

    ...

    # 标注该模型是否支持 cudagraph
    # 可以是一个 callable 对象，接收 forward 输入
    # 动态判断是否支持 cudagraph
    support_cuda_graph = True

    # 构建模型输入
    # 返回词典，词典的 key 必须是 forward 的输入
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        ...

    # 加载权重
    # 模型的输入是 state dict 的 key value 对
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        ...
```

我们封装了许多融合算子以简化模型的搭建。这些算子能够更好的支持 tensor 并行、量化等各种功能，我们鼓励开发者尽可能使用这些 op 进行开发。

```python
# 使用预定义的 build_merged_colwise_linear, SiluAndMul, build_rowwise_linear
# 可以帮助我们更快搭建模型，并且不用关心 tensor 并发、量化等细节
class LlamaMLP(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=config.mlp_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=config.mlp_bias,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)
```

### 模型注册

为了能够让开发的模型实现可以正常使用，我们还需要在 `lmdeploy/pytorch/models/module_map.py` 中注册该模型

```python
MODULE_MAP.update({
    'LlamaForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
})
```

如果你不希望修改模型源码，也可以从外部传入自定义的 module map，方便整合进其他项目中

```
from lmdeploy import PytorchEngineConfig, pipeline

backend_config = PytorchEngineConfig(custom_module_map='/path/to/custom/module_map.py')
generator = pipeline(model_path, backend_config=backend_config)
```
