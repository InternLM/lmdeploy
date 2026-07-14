# lmdeploy.pytorch New Model Support

lmdeploy.pytorch is designed to simplify the support for new models and the development of prototypes. Users can adapt new models according to their own needs.

## Model Support

### Configuration Loading (Optional)

lmdeploy.pytorch initializes the engine based on the model's config file. If the parameter naming of the model to be integrated differs from common models in transformers, parsing errors may occur. A custom ConfigBuilder can be added to parse the configuration.

```python
# lmdeploy/pytorch/configurations/gemma.py

from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class GemmaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        # Check if hf_config is suitable for this builder
        return hf_config.model_type in ['gemma', 'gemma2']

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        # Use the hf_config loaded by transformers
        # Construct the ModelConfig for the pytorch engine
        return ModelConfig(hidden_size=hf_config.hidden_size,
                           num_layers=hf_config.num_hidden_layers,
                           num_attention_heads=hf_config.num_attention_heads,
                           num_key_value_heads=hf_config.num_key_value_heads,
                           bos_token_id=hf_config.bos_token_id,
                           eos_token_id=hf_config.eos_token_id,
                           head_dim=hf_config.head_dim,
                           vocab_size=hf_config.vocab_size)
```

The `lmdeploy.pytorch.check_env.check_model` function can be used to verify if the configuration can be parsed correctly.

### Implementing the Model

After ensuring that the configuration can be parsed correctly, you can start implementing the model logic. Taking the implementation of llama as an example, we need to create the model using the configuration file from transformers.

```python
class LlamaForCausalLM(nn.Module):

    # Constructor, builds the model with the given config
    # ctx_mgr is the context manager, which can be used to pass engine configurations or additional parameters
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

    # Model inference function
    # It is recommended to use the same parameters as below
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

In addition to these, the following content needs to be added:

```python
class LlamaForCausalLM(nn.Module):

    ...

    # Indicates whether the model supports cudagraph
    # Can be a callable object, receiving forward inputs
    # Dynamically determines if cudagraph is supported
    support_cuda_graph = True

    # Builds model inputs
    # Returns a dictionary, the keys of which must be inputs to forward
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        ...

    # Loads weights
    # The model's inputs are key-value pairs of the state dict
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        ...
```

We have encapsulated many fused operators to simplify the model construction. These operators better support various functions such as tensor parallelism and quantization. We encourage developers to use these ops as much as possible.

```python
# Using predefined build_merged_colwise_linear, SiluAndMul, build_rowwise_linear
# Helps us build the model faster and without worrying about tensor concurrency, quantization, etc.
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

### Model Registration

To ensure that the developed model implementation can be used normally, we also need to register the model in `lmdeploy/pytorch/models/module_map.py`

```python
MODULE_MAP.update({
    'LlamaForCausalLM':
    f'{LMDEPLOY_PYTORCH_MODEL_PATH}.llama.LlamaForCausalLM',
})
```

If you do not wish to modify the model source code, you can also pass a custom module map from the outside, making it easier to integrate into other projects.

```
from lmdeploy import PytorchEngineConfig, pipeline

backend_config = PytorchEngineConfig(custom_module_map='/path/to/custom/module_map.py')
generator = pipeline(model_path, backend_config=backend_config)
```
