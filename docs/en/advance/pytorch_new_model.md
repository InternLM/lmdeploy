# How to support new model in lmdeploy.pytorch

lmdeploy.pytorch is designed to ease new model deployment and prototype verification. If you are willing to use our engine, here is the tutorial.

## Support New Model

Let's begin with Llama.

Before delving into the details, it's essential to acquaint ourselves with the input specifications of the model. In order to accommodate new features within our engine, there are some deviations from the typical transformer inputs.

1. To circumvent the need for batch padding, continuous batching is employed. Consequently, the `input_ids` now represents the concatenation of all input sequences in the batch, followed by a `unsqueeze(0)` operation to align with the original `input_ids` dimension.

2. In an effort to optimize memory usage for the key/value cache, we implement paged attention. This transforms the `past_key_value` into a substantial tensor with dimensions `[num_blocks, block_size, num_heads, head_dim]`. Here, `num_blocks` denotes the number of page blocks, and `block_size` indicates the size of each block.

3. Accompanying these changes, additional inputs are imperative to support the modified inputs described above. These include the block table and history length. It's important to note that these supplementary inputs are not explicitly listed as arguments in the original forward method. Instead, a context object is utilized to furnish this essential information.

Due to the alterations in the input structure mentioned earlier, the forward methods for both `LlamaModel` and `LlamaAttention` modules need to be adjusted. Below are the modified implementations:

For `LlamaModel`:

```python
# lmdeploy/pytorch/models/llama.py

class LlamaModel(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
```

For LlamaAttention:

```python
# lmdeploy/pytorch/models/llama.py
from lmdeploy.pytorch.kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd

class LlamaAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        context = self.context.context
        history_lengths = context.history_lengths
        position_ids_1d = context.position_ids_1d
        block_offsets = context.block_offsets

        # qkv proj
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        # rotary embedding
        max_seq_len = position_ids.size(-1)
        kv_seq_len = max_seq_len + max(history_lengths)
        if kv_seq_len >= self.rotary_emb.max_seq_len_cached:
            cos, sin = self.rotary_emb(value_states,
                                        seq_len=kv_seq_len + 128)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            self.rotary_emb.cos_cached,
            self.rotary_emb.sin_cached,
            position_ids,
            position_ids_1d,
            q_embed=query_states,
            k_embed=key_states)

        # fill kv cache
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.seq_length
        q_start_loc = context.q_start_loc
        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)

        # attention
        attn_output = query_states
        block_size = past_key_value[0].size(1)
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            b_start_loc=q_start_loc,
            b_seq_len=q_seq_length,
            b_kv_seq_len=kv_seq_length,
            max_input_len=max_seq_len,
        )
        hidden_size = num_heads * head_dim
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], hidden_size)

        # o proj
        attn_output = o_proj(attn_output)
        return attn_output, None, past_key_value
```

Note: The additional arguments like `history_lengths` and `block_offsets` are accessed from the `context` object, which acts as a container for the necessary inputs required by continuous batching and paged attention. Refer to the [context info](#context-info) for more detail about `context` object.

We have replaced certain operations with our custom Triton kernel for two reasons:

1. The custom Triton kernel allows us to incorporate new features, such as `paged_attention_fwd`.
2. Fused kernels offer superior performance compared to the pure PyTorch implementation.

Now that we have the updated implementations for the two modules, let's register them in `lmdeploy/pytorch/models/module_map.py`.

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaAttention':
    'lmdeploy.pytorch.models.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    'lmdeploy.pytorch.models.llama.LlamaModel'
})
```

In this mapping, the revised modules are associated with their original counterparts. When creating an `Engine`, the `ModelAgent` will automatically patch the model. Subsequently, we can conduct inference using these updated implementations.

## Support Tensor Parallelism

If we aim to enable tensor parallelism (TP), it is necessary to partition the weights in the model. Let's build upon the previously mentioned modifications to accommodate TP in the Llama model:

In Llama (as well as in most Language Model models), the weight partition primarily affects the Linear layers. Specifically, for the following components:

- In `LlamaAttention`: `q_proj`, `k_proj`, `v_proj` require column-wise partitioning, while `o_proj` necessitates row-wise partitioning.
- In `LlamaMLP`: `gate_proj` and `up_proj` require column-wise partitioning, while `down_proj` requires row-wise partitioning.

We can implement the \_distribution_partition_fn in each of the rewritten modules:

```python
# lmdeploy/pytorch/models/llama.py
from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)

class LlamaAttention(nn.Module):
    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['o_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

class LlamaMLP(nn.Module):
    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['down_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

```

In the process of loading model weights, the `_distribute_partition_fn` is called to distribute the weights of specific modules across different devices. Following the weight partitioning, it becomes necessary to perform `all_reduce` on the output tensors of `o_proj` and `down_proj`. While one option is to include `all_reduce` directly in the forward method, an alternative approach is to introduce the `_distribute_output_fn` call:

```python
# lmdeploy/pytorch/models/llama.py
import torch.distributed as dist

class LlamaAttention(nn.Module):
    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

class LlamaMLP(nn.Module):
    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs
```

It is essential to remember to add `LlamaMLP` to the `module_map`:

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaMLP':
    'lmdeploy.pytorch.models.llama.LlamaMLP'
})
```

With these adjustments, the model is now capable of utilizing multiple GPUs for deploying Large Language Models (LLM). This enables efficient distribution of computations across different devices in a parallelized manner.

## Debug Module

When the output of the model does not meet expectations, we would like to debug a specific module to determine if the added rewrite is correct. `lmdeploy.pytorch` provides some tools to assist with accuracy alignment. Let’s take `LlamaAttention` module as an example.

First, create an instance of the module that we want to debug:

```python
import torch
from transformers import AutoModelForCausalLM

# get module
model_path = 'meta-llama/Llama-2-7b-chat-hf'
dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float16).cuda()
self_attn = model.model.layers[0].self_attn
```

Extract the inputs/outputs with `ModuleIOExtractor`.

```python
from lmdeploy.pytorch.tools.make_inputs import ModuleIOExtractor

# extract module input/output
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
extractor = ModuleIOExtractor(model, self_attn)
attn_args, attn_kwargs, attn_output = extractor.extract(input_ids)
```

The inputs of rewrite module are different from the inputs of origin module:

1. Module requires some special inputs, which are passed through `StepContext`. We can create one with `make_step_context`.
2. `input_ids`, `hidden_states` should be continuous. We can use `continuous_tensor` to do the process.
3. `past_key_value` should be paged to meet the demond of paged attention.

Based on the reason above, the input should be updated:

```python
from lmdeploy.pytorch.tools.make_inputs import make_step_context
from lmdeploy.pytorch.tools.layout_convert import continuous_tensor

# create patched input/output
context = make_step_context(input_ids,
                            kv_cache_dtype=dtype)
seq_length = context.seq_length
attn_kwargs['hidden_states'] = continuous_tensor(
    attn_kwargs['hidden_states'],
    seq_length)
attn_kwargs['past_key_value'] = context.kv_caches[0]
```

Then you can start the rewrite and compare the correctness of the results.

```python
from lmdeploy.pytorch.models import patch

# patch and test
patched_self_attn = patch(self_attn, extra_args=['context'])
patched_output = patched_self_attn.patched_forward(*attn_args,
                                                    **attn_kwargs,
                                                    context=context)
torch.testing.assert_close(patched_output[0],
                            continuous_tensor(attn_output[0], seq_length))
```

Adjust the rewrite module until the output can be aligned.

## Appendix

### context info

```python
@dataclass
class StepContext:
    """context of Model.
    """
    inputs: ModelInputs
    block_offsets: torch.LongTensor
    position_ids: torch.LongTensor
    position_ids_1d: torch.LongTensor
    q_start_loc: torch.LongTensor
    history_lengths: torch.LongTensor
    seq_length: torch.LongTensor
    max_seq_length: int
    kv_seq_length: torch.LongTensor
    kv_caches: List
    is_decoding: bool
    world_size: int = 1
    json_config: Dict = None
    local_adapter_ids: torch.LongTensor = None
    global_adapter_ids: torch.LongTensor = None
    adapter_offsets: torch.LongTensor = None
    max_rank: int = 0
```

### FAQ

- **How to invoke the original forward method?**

A common approach is to add hooks to a method rather than performing a complete rewrite. To access the unpatched module, you can utilize self.origin_mod within the rewritten method.

- **How to register modules in remote code?**

For modules located in remote code, pinpointing them via `qualname` might be challenging. `lmdeploy.pytorch` facilitates registration using abbreviations for such modules:n:

```python
MODULE_MAP.update({
    'modeling_internlm.InternLMAttention':
    'lmdeploy.pytorch.models.internlm.PatchedInternLMAttention',
})
```

> \[!NOTE\]
>
> Although abbreviations are supported, they tend to have lower priority. It is advisable to register modules using their complete `qualname` for more robust and accurate mapping.

- **How to support different modules with the same name?**

You can accommodate multiple modules with the same name within a single rewrite module by providing distinct implementations based on their attributes. For instance, consider `baichuan2` 7b/13b:

```python
class BaichuanModel(nn.Module):
    def forward(self, ...):
        if self.config.num_hidden_layers == 32:
            return forward_7b(...)
        else:
            return forward_default(...)
```

- **How to perform post-initialization for a rewrite module?**

To execute tasks after model weight loading, introduce a `_update_model_fn` method in your rewrite module. This method will be automatically called post-initialization:

```python
class LlamaAttention:
    def _update_model_fn(self):
        # ADD YOUR CODE HERE
```

Here, you can include any additional post-initialization steps or configurations needed for your specific use case.
