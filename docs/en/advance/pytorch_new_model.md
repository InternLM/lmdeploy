# How to support new model in lmdeploy.pytorch

lmdeploy.pytorch is designed to ease new model deployment and prototype verification. If you are willing to use our engine, here is the tutorial.

## Support New Model

Let's start with Llama.

before we start, let's take a look at the inputs of the model. To support new features in our engine, the inputs are a little bit different from the inputs in transformers.

1. Continuous batching is used to avoid batch padding, so the `input_ids` would be the concatenation of all input sequence in batch, than `unsqueeze(0)` to match the dimension of origin input_ids.
2. Paged attention is used to reduce the memory usage of key/value cache, `past_key_value` become a big Tensor with shape `[num_blocks, block_size, num_heads, head_dim]`, where num_blocks is the number of page block, block_size is the the size of each block.
3. Extra inputs are necessary to support the inputs above, such as block table, history length. These extra inputs are not listed in arguments of origin forward method. A context object is used to provide these info.

Because of the change of the inputs above, we need to rewrite forward of `LlamaModel` and `LlamaAttention` to fit the new inputs. First, let's rewrite the `LlamaModel`, we only keep the minimal codes to support deployment:

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

For LlamaAttention module, we need to perform following steps:

1. kqv proj
2. rotary embedding
3. filling kv cache
4. MHA
5. o proj

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

Notice that some arguments such as `history_lengths` and `block_offsets` comes from `self.context.context`. As we have mentioned above, continuous batching and paged attention require extra arguments to support them, `context` is the container to store these inputs. If you need more detail about context object, please read [context info](#context-info).

We replace some operation to our custom triton kernel for two reason.

1. Custom triton kernel can be used to support new features such as `paged_attention_fwd`.
2. Fuse kernels have better performance than the pure PyTorch implementation.

Now we have new implementations of two modules, let's register them into `lmdeploy/pytorch/models/module_map.py`.

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaAttention':
    'lmdeploy.pytorch.models.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    'lmdeploy.pytorch.models.llama.LlamaModel'
})
```

The rewritten module has been mapped to the origin module. When we create an Engine, ModelAgent would patch the model automatically, then we can perform inference with these new implementation.

## Support Tensor Parallelism

If we want to support tensor parallelism(tp), we have partition the weights in the model. Let's try extend the rewrite above.

In Llama (and most LLM), most Linear layers are involved in the weight partition. Among them:

- `LlamaAttention`: `q_proj`, `k_proj`, `v_proj` need column wise partition; `o_proj` needs row wise partition.
- `LlamaMLP`: `gate_proj`, `up_proj` need column wise partition; `down_proj` needs row wise partition.

We can implement `_distribution_partition_fn` in each rewrite modules:

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

`_distribute_partition_fn` would be called when loading model weights, the weights of special module would be distributed to different devices.

After partition, we need to perform `all_reduce` on the output of `o_proj` and `down_proj`. Of cause you can just put `all_reduce` in the forward method, another option is add an `_distribute_output_fn` call:

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

Don't forget to add `LlamaMLP` in `module_map`.

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaMLP':
    'lmdeploy.pytorch.models.llama.LlamaMLP'
})
```

That's all. Now it is possible to utilize multiple GPUs to deploy LLM.

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

- How to call origin forward?

It is a common practice to add hooks to a method instead a full rewrite. You can use `self.origin_mod` to visit the unpatched module.

- How to register modules in remote code?

Some modules are contained in remote code, it is hard to locate the module with `qualname`. `lmdeploy.pytorch` support register them with abbreviation:

```python
MODULE_MAP.update({
    'modeling_internlm.InternLMAttention':
    'lmdeploy.pytorch.models.internlm.PatchedInternLMAttention',
})
```

> \[!NOTE\]
>
> Abbreviation tends to have a low priority. It is recommend to register modules with `qualname`.

- How to support different modules with same name?

You can support them in the same rewrite module, and give them different implement by their attribute, take `baichuan2` 7b/13b as example:

```python
class BaichuanModel(nn.Module):
    def forward(self, ...):
        if self.config.num_hidden_layers == 32:
            return forward_7b(...)
        else:
            return forward_default(...)
```

- How to do post-initialization for rewrite module?

Add a `_update_model_fn` method, it will be called after weight loading.

```python
class LlamaAttention:
    def _update_model_fn(self):
        # ADD YOUR CODE HERE
```
