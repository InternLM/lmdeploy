# TurboMind

TurboMind 是一个基于英伟达公司的 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 构建的支持对话式 LLMs 的高吞吐量的推理引擎。 TurboMind 的主要功能包括一个高效简洁的 LLaMa 实现、推理的 persistent batch 技术以及一个可扩展的 KV 缓存管理器。

## TurboMind 工作结构

```
  +--------------------+
  |        API         |
  +--------------------+
          |    ^
    请 求  |    | 流式回调
          v    |
  +--------------------+    获取   +-------------------+
  |  Persistent Batch  | <-------> |  KV Cache 管理器 |
  +--------------------+    更新   +-------------------+
             ^
             |
             v
+------------------------+
| TurboMind 的 LLaMA 实现  |
+------------------------+
| FT kernels & utilities |
+------------------------+
```

## Persistent Batch

你也许在别的项目中看到这项机制的另一个名字： `continuous batching` 。在开发这个功能时，我们将对话式 LLM 的推理建模为一个持续运行的 batch ，其生命周期跨越整个服务过程，故将其命名为 `persistent batch` 。简单来说是这样实现的：

- 该功能会预先准备好 N 个 batch slots。
- 当有新的请求来并且有空闲的 slots 可用时该请求就会加入到 batch 中。当请求对应的 tokens 都生成完毕后对应的 batch slot 就会立刻被释放并且可以直接复用。
- **当判断为 `cache-hits`（本文后续会说明）时，历史 tokens 就不需要在一段对话的每一轮中都被解码，对应的回复的 tokens 就自动开始生成**。
- 整个 batch 会自动扩缩容来避免不必要的计算。

## KV 缓存管理器

TurboMind 的 [KV 缓存管理器](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/models/llama/LlamaCacheManager.h) 是一个内存池类型的对象，并且在其中加入了 LRU 的实现，这样整个管理器可以被看作是一个 **KV 缓存的缓存**。大致工作方式如下：

- KV 缓存所需的所有设备内存都由管理器分配，管理器会先在系统内存空间上配置好一个固定数值对应的所有 slots，每个 slot 对应一个单独的 sequence 的 KV 缓存所需的内存。分配块大小可通过配置来实现预分配或者按需分配（或介于两者之间）。
- 当缓存池中已经没有空闲的 slot 并且新 sequence 的 KV 缓存需要使用对应的存储空间时，根据 LRU 机制，管理器会踢除最近使用最少的 sequence 对应的 slot 同时新的 sequence 会直接使用这个 slot。但不仅仅如此。
- 当获取到已经在 slots 中的 sequnce 时即标记为 _cache-hit_ ，此时 KV 缓存的历史就会被直接返回并且不需要进行文本解码。
- 被踢除的 sequences 不会被完全的删除，而是会被转换成最简洁的形式，例如 token IDs 。当之后获取到相同的 sequence id 时 (即 _cache-miss_ 状态)，这些 token IDs 将被 FMHA 的文本解码器解码并被转回 KV 缓存。
- 踢除和转换均由 TurboMind 内部自动管理所以对用户来说是透明的。__从用户的使用角度来看，使用了 TurboMind 的系统就像是可以访问无限的设备内存__。

## TurboMind 的 LLaMa 实现

我们对 LLaMa 系列模型的实现是从 FasterTransformer 中的 Gpt-NeX 模型修改而来的。除了对 LLaMa 系列进行基本重构和修改外，我们还做了一些改进以实现会话模型的高性能推理，其中最重要的是：

- 支持多轮对话中的快速文本解码。我们用基于 [cutlass](https://github.com/NVIDIA/cutlass) 的 FMHA 实现替代了文本解码器中的注意力机制实现，从而支持了 Q/K 长度不匹配的情况。
- 我们在文本 FMHA 和生成式 FMHA 中都加入了间接缓冲指针来支持 batch 中 KV 缓存不连续的问题。
- 为了支持 persistent batch 的并发推理，我们设计了新的同步机制来协调在张量并型模式下的 worker 线程。
- 为了最大限度提高吞吐量，我们完成了 INT8 KV 缓存的实现从而提高了最大批处理的规模。这在实际生产使用场景中是很有效的，因为相比于权重或其他的激活函数， KV 缓存会消耗更多的内存和内存带宽。
- 我很还解决了在单个进程中以 TP 模式（张量并行）运行多个模型实例时 NCCL 挂起的问题。NCCL APIs 现在会被 host 端的同步 barriers 监控。

## API

TurboMind 的 Python API 支持流式结果返回和张量并行模式。

同时 TurboMind 也继承了 FasterTransformer 的能够使用 [tritonserver](https://github.com/triton-inference-server/server) 推理的能力。但为了支持对 persistent batch 的模型进行并发请求，我们没有像 FasterTransformer 一样使用序列化 batching 或者动态 batching ，而是用 TurboMind 管理所有请求的 bookkeeping 以及序列状态。

## TurboMind 和 FasterTransformer 的区别

除了上文中提到的功能外，TurboMind 相较于 FasterTransformer 还有不少小的差别。譬如不少 FasterTransformer 的功能在 TurboMind 中都被去掉了，这其中包括前缀提示词、 beam search 、上下文 embedding、稀疏化 GEMM 操作和对应 GPT 或 T5 等结构的模型的支持等等。

## FAQ

### 对 Huggingface 模型的支持

因为历史因素， TurboMind 的权重设计是基于 [LLaMa 的官方实现](https://github.com/facebookresearch/llama) 完成的，两者只相差一个转置操作。但是 Huggingface 版本的实现却是[另一种形式](https://github.com/huggingface/transformers/blob/45025d92f815675e483f32812caa28cce3a960e7/src/transformers/models/llama/convert_llama_weights_to_hf.py#L123C76-L123C76)，两种权重实现方式在 `W_q` 和 `W_k` 上的区别我们在 [deploy.py](https://github.com/InternLM/lmdeploy/blob/ff4648a1d09e5aec74cf70efef35bfaeeac552e0/lmdeploy/serve/turbomind/deploy.py#L398) 进行了适配处理，用户可前往查看。
