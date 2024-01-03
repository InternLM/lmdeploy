# TurboMind 框架

TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。

## TurboMind 结构

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
|      LLaMa推理实现      |
+------------------------+
| FT kernels & utilities |
+------------------------+
```

## Persistent Batch

你也许在别的项目中看到这项机制的另一个名字： `continuous batching` 。在开发这个功能时，我们将对话式 LLM 的推理建模为一个持续运行的 batch ，其生命周期跨越整个服务过程，故将其命名为 `persistent batch` 。简单来说是这样实现的：

- 该功能会预先准备好 N 个 batch slots。
- 当有空闲 slots 时， 请求就会加入到 batch 中。当请求对应的 tokens 都生成完毕后，对应的 batch slot 会立刻被释放，接收新的请求。
- **当一个 sequence 命中缓存时（见下文），它的历史 token 不必在每轮中都进行解码，所以它的 token 生成过程会即刻开始**。
- 整个 batch 会自动扩缩容来避免不必要的计算。

## KV 缓存管理器

TurboMind 的 [KV 缓存管理器](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/models/llama/SequenceManager.h) 是一个内存池类型的对象，并且在其中加入了 LRU 的实现，这样整个管理器可以被看作是一个 **KV 缓存的缓存**。大致工作方式如下：

- KV 缓存由管理器分配。管理器会根据预先配置好的 slot 数量开辟空间。每个 slot 对应于一个 sequence 所需的 KV 缓存。分配的内存块大小可通过配置来实现预分配或者按需分配（或介于两者之间）。
- 当有新的请求，但是缓存池中没有空闲 slot时，根据 LRU 机制，管理器会踢除最近使用最少的 sequence，把它占据的 slot 分给新的请求。不仅仅如此，
- sequence获取到了slot，类似缓存命中。它在缓存中的历史KV会被直接返回，而不用再进行context decoding 。
- 被踢除的 sequences 不会被完全的删除，而是会被转换成最简洁的形式，例如 token IDs 。当之后获取到相同的 sequence id 时 (即 _cache-miss_ 状态)，这些 token IDs 将被 FMHA 的 context decoder 解码并被转回 KV 缓存。
- 踢除和转换均由 TurboMind 内部自动管理所以对用户来说是透明的。__从用户的使用角度来看，使用了 TurboMind 的系统就像是可以访问无限的设备内存__。

## TurboMind 的 LLaMa 实现

我们对 LLaMa 系列模型的实现是从 FasterTransformer 中的 Gpt-NeX 模型修改而来的。除了对 LLaMa 系列进行基本重构和修改外，我们还做了一些改进以实现会话模型的高性能推理，其中最重要的是：

- 支持多轮对话中的快速文本解码。我们用基于 [cutlass](https://github.com/NVIDIA/cutlass) 的 FMHA 实现替代了 context decoder 中的注意力机制实现，从而支持了 Q/K 长度不匹配的情况。
- 我们在 context FMHA 和 generation FMHA 中都加入了间接缓冲指针，支持 batch 中不连续的 KV 缓存。
- 为了支持 persistent batch 的并发推理，我们设计了新的同步机制来协调在张量并型模式下的工作线程。
- 我们实现了 INT8 KV cache，降低了内存开销，提高了批处理大小和系统吞吐量。这在实际场景中非常有用，因为相比权重和其他激活，KV cache 会消耗更多的内存和内存带宽。
- 我们解决了单个进程内多个模型实例在 TP 模式下运行时 NCCL 卡住的问题。NCCL APIs 现由 host 端的同步 barriers 保护。

## API

TurboMind 的 Python API 支持流式结果返回和张量并行模式。

同时 TurboMind 也继承了 FasterTransformer 能够注册为 [Triton Inference Server](https://github.com/triton-inference-server/server) 推理后端的能力。但是为了支持 persistent batch 中的并发请求，我们不再像 FasterTransformer 那样使用 sequence batching 或者 dynamic batching 。相反，TurboMind 负责记录和管理请求序列的状态。

## TurboMind 和 FasterTransformer 的区别

除了上文中提到的功能外，TurboMind 相较于 FasterTransformer 还有不少差别。譬如不少 FasterTransformer 的功能在 TurboMind 中都被去掉了，这其中包括前缀提示词、 beam search 、上下文 embedding、稀疏化 GEMM 操作和对应 GPT 或 T5 等结构的模型的支持等等。

## FAQ

### 对 Huggingface 模型的支持

因为历史因素， TurboMind 的权重设计是基于 [LLaMa 的官方实现](https://github.com/facebookresearch/llama) 完成的，两者只相差一个转置操作。但是 Huggingface 版本的实现却是[另一种形式](https://github.com/huggingface/transformers/blob/45025d92f815675e483f32812caa28cce3a960e7/src/transformers/models/llama/convert_llama_weights_to_hf.py#L123C76-L123C76)，两种权重实现方式在 `W_q` 和 `W_k` 上的区别我们在 [deploy.py](https://github.com/InternLM/lmdeploy/blob/ff4648a1d09e5aec74cf70efef35bfaeeac552e0/lmdeploy/serve/turbomind/deploy.py#L398) 进行了适配处理，用户可前往查看。
