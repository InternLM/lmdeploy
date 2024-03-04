# lmdeploy.pytorch 新模型支持

lmdeploy.pytorch 被设计用来简化新模型的支持以及原型的开发，新模型的支持依赖于 patch 机制，对原模型做修改以及功能添加，以期可以最大程度上复用模型的原始实现，减少工作量。

## 模型支持

我们以 transformers 中的 llama 实现来介绍模型支持的流程

在开始之前，我们首先要了解一下模型的输入。lmdeploy.pytorch 的输入与标准 transformers 模型的输入略有不同，差异主要体现在如下方面：

1. 由于支持了 continuous batching，一个 batch 的输入 `input_ids` 会被拼接成一维的长序列，然后 `unsqueeze(0)` 来保证输入维度与 transformers 中相同。这样的输入不会影响 MLP 以及 RMSNorm 等模块的计算。
2. 由于添加了对 paged attention 的支持，`past_key_value` 不再是原来的大小，而是一组形状为 `[num_blocks, block_size, num_heads, head_dim]` 的 cache 块，num_blocks 为总 block 数量，由可用显存大小决定，block_size 为预设的块大小。这样的输入改变会影响到 LlamaModel 和 LlamaAttention 的计算，因此要对这两个模块的实现进行修改。
3. 由于上述输入的改变，模型中需要一些额外的输入来支持推理，比如 batch 中的序列起始位置和长度，kv cache 的 block table 等。这些输入并不在模块的 forward 参数列表中，我们需要维护一个上下文以获得这些输入。

上面的输入改动会影响 LlamaModel 和 LlamaAttention，首先我们来实现新的 LlamaModel，这是对原始实现的简化，我们删除了很多检查代码，以避免由于输入改变造成的断言失败，仅保留了最小程度的代码：

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

然后是对 LlamaAttention 模块的改写。按顺序实现如下操作：

1. kqv proj
2. rotary embedding
3. 填充 kv cache
4. MHA 计算
5. o proj

continuous batching 和 kv cache 的改动对该模块的影响比较大

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
        q_seq_length = context.q_seq_length
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
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_seq_len,
        )
        hidden_size = num_heads * head_dim
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], hidden_size)

        # o proj
        attn_output = o_proj(attn_output)
        return attn_output, None, past_key_value
```

上面的代码有几处值得注意的地方，首先是 context 对象。我们需要 history_lengths、block_offsets 等参数辅助运算，这些参数无法通过模型的 forward 函数传递进来。因此我们维护了一个 context 对象，把几乎所有可能用到的输入参数都保存在其中，方便在各个模块间共享。context 对象可以通过 `self.context.context` 来访问，结构可以参考 [context-结构](#context-结构)。

另一个值得注意的地方就是自定义 kernel，由于输入形式的改变，原来的 LlamaAttention 实现变得不再适用，为了保证推理的速度和正确性，我们在 lmdeploy.pytorch.kernels 中实现了许多自定义的 triton kernel，上面的模块中就用到了 `apply_rotary_pos_emb`，`fill_kv_cache` 和 `paged_attention_fwd` ，分别负责实现 rotary embedding，填充 kv cache 还有 attention 的计算。

有了上述的两个模块后，还需要将他们注册到 `lmdeploy/pytorch/models/module_map.py` 中，进行原模块与 patch 模块的映射

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaAttention':
    'lmdeploy.pytorch.models.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    'lmdeploy.pytorch.models.llama.LlamaModel'
})
```

完成注册后，Engine 在启动时就会将这两个模块 patch 成新的实现，完成后续的部署任务。

## Tensor 并发支持

为了支持 Tensor 并发，需要对模型的权重做切分。让我们试着为上面接入的 Llama 模型添加 TP 的支持。

Llama 中涉及到 Tensor 并发的模块是 LlamaAttention 中的 qkvo proj 和 LlamaMLP 中的 gate,up 和 down proj。其中 o_proj 和 down_proj 需要按行切分，剩下的按列切分。我们可以在对应的模块中实现 `_distribution_partition_fn` 函数：

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

`_distribute_partition_fn` 会在加载模型权重时被调用，对应的权重会被按照特定的形式分配到对应的设备中。

按照目前的方案切分后的权重，需要对 o_proj 和 down_proj 的结果进行 all_reduce 操作才能得到正确的结果。可以选择将 all_reduce 放在模型的 forward 函数中，也可以选择另一种方案，添加 `_distribute_output_fn` 函数：

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

最后别忘了将 LlamaMLP 也注册进 module_map 中

```python
# lmdeploy/pytorch/models/module_map.py
MODEL_MAP.update({
    'transformers.models.llama.modeling_llama.LlamaMLP':
    'lmdeploy.pytorch.models.llama.LlamaMLP'
})
```

这样就可以利用多卡的优势，让更大的模型部署成为可能

## 模块调试

当模型的输出不符合预期时，我们会希望调试某个特定模块以确定添加的重写是否正确。`lmdeploy.pytorch` 提供了一些工具以帮助进行精度对齐。还是以上面提到的 `LlamaAttention` 模块为例。

首先，我们通过 transformers 的 API 得到想要调试的子模块的一个实例：

```python
import torch
from transformers import AutoModelForCausalLM

# get module
model_path = 'meta-llama/Llama-2-7b-chat-hf'
dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float16).cuda()
self_attn = model.model.layers[0].self_attn
```

然后，使用 `ModuleIOExtractor` 工具可以生成该模块的一组输入输出

```python
from lmdeploy.pytorch.tools.make_inputs import ModuleIOExtractor

# extract module input/output
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
extractor = ModuleIOExtractor(model, self_attn)
attn_args, attn_kwargs, attn_output = extractor.extract(input_ids)
```

重写模块的输入与原模块略有不同，主要体现在三方面：

1. 模型需要一些特殊输入输出，他们以 `StepContext` 的形式传入，可以使用 `make_step_context` 生成。
2. `input_ids`，`hidden_states` 等数据都被 continuous 化，可以使用 `continuous_tensor` 进行处理。
3. 由于 paged caching 的需要， `past_key_value` 需要被 page 化处理。

基于以上原因，我们要对提取的输入进行加工：

```python
from lmdeploy.pytorch.tools.make_inputs import make_step_context
from lmdeploy.pytorch.tools.layout_convert import continuous_tensor

# create patched input/output
context = make_step_context(input_ids,
                            kv_cache_dtype=dtype,
                            num_key_value_heads=32)
seq_length = context.q_seq_length
attn_kwargs['hidden_states'] = continuous_tensor(
    attn_kwargs['hidden_states'],
    seq_length)
attn_kwargs['past_key_value'] = context.kv_caches[0]
```

然后就可以启动重写，并比较结果正确性了。（注意输出也要 continuous 化后进行比较）

```python
from lmdeploy.pytorch.models import patch

# patch and test
patched_self_attn = patch(self_attn, extra_args=['context'])
with torch.inference_mode():
    patched_output = patched_self_attn.patched_forward(*attn_args,
                                                       **attn_kwargs,
                                                       context=context)
torch.testing.assert_close(patched_output[0],
                            continuous_tensor(attn_output[0], seq_length))
```

可以通过上述方法调试重写模块，直到精度满足预期。

## 附录

### context 结构

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

- **如何访问 patch 前的模块？**

有时我们只希望在函数前后加一个 hook 代码，不希望大段的拷贝函数，可以通过 `self.origin_mod` 访问 patch 前的模块。

- **非 transformers 官方的模型该如何注册？**

一些模型的实现代码可能是以 remote code 的形式添加的，这样的模块无法通过完整的 qualname 来定位。lmdeploy.pytorch 支持使用缩写的模块名进行注册：

```python
MODULE_MAP.update({
    'modeling_internlm.InternLMAttention':
    'lmdeploy.pytorch.models.internlm.PatchedInternLMAttention',
})
```

> \[!NOTE\]
>
> 缩写的优先级会更低，有条件的话还是鼓励使用完整的 qualname 进行注册。

- **模块出现同名但不同实现怎么处理？**

目前推荐的做法是同名就映射到同一个实现中，然后在实现内部根据模块的固有参数来判断模型该使用的类型，以 baichuan2 7b/13b 为例：

```python
class BaichuanModel(nn.Module):
    def forward(self, ...):
        if self.config.num_hidden_layers == 32:
            return forward_7b(...)
        else:
            return forward_default(...)
```

- **如果希望在推理前对模块进行初始化?**

可以实现模块的 `_update_model_fn` 函数，它会在模块的权重都加载完，完成 TP 权重切分后被调用

```python
class LlamaAttention:
    def _update_model_fn(self):
        # ADD YOUR CODE HERE
```
