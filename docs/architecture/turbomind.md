# Architecture of TurboMind

TurboMind is an inference engine that supports high throughput inference for conversational LLMs. It's based on NVIDIA's [FasterTransformer](https://github.com/NVIDIA/FasterTransformer). Major features of TurboMind include an efficient LLaMa implementation, the persistent batch inference model and an extendable KV cache manager.  

## High level overview of TurboMind 

```
  +--------------------+
  |        API         |
  +--------------------+
          |    ^
  request |    | stream callback
          v    |
  +--------------------+   fetch   +-------------------+
  |  Persistent Batch  | <-------> |  KV Cache Manager |
  +--------------------+   update  +-------------------+
             ^ 
             | 
             v
+------------------------+
|  LLaMA implementation  |
+------------------------+
| FT kernels & utilities |
+------------------------+
```

## Persistent Batch 

You may recognize this feature as "continuous batching" in other repos. But during the concurrent development of the feature, we modeled the inference of a conversational LLM as a persistently running batch whose lifetime spans the entire serving process, hence the name "persistent batch". To put it simply

- The persistent batch as N pre-configured batch slots.
- Requests join the batch when there are free slots available. A batch slot is released and can be reused once the generation of the requested tokens is finished.
- __On cache-hits (see below), history tokens don't need to be decoded in every round of a conversation; generation of response tokens will start instantly.__
- The batch grows or shrinks automatically to minimize unnecessary computations.


## KV Cache Manager

The [KV cache manager](/src/turbomind/models/llama/LlamaCacheManager.h) of TurboMind is a memory-pool-liked object that also implements LRU policy so that it can be viewed as a form of __cache of KV caches__. It works in the following way

- All device memory required for KV cache is allocated by the manager. A fixed number of slots is pre-configured to match the memory size of the system. Each slot corresponds to the memory required by the KV cache of a single sequence. Allocation chunk-size can be configure to implement pre-allocate/on-demand style allocation policy (or something in-between).
- When space for the KV cache of a new sequence is requested but no free slots left in the pool, the least recently used sequence is evicted from the cache and its device memory is directly reused by the new sequence. However, this is not the end of the story.
- Fetching sequence currently resides in one of the slots resembles a _cache-hit_, the history KV cache is returned directly and no context decoding is needed.
- Victim (evicted) sequences are not erased entirely but converted to its most compact form, i.e. token IDs. When the same sequence id is fetched later (_cache-miss_) the token IDs will be decoded by FMHA backed context decoder and converted back to KV cache.
- The eviction and conversion are handled automatically inside TurboMind and thus transparent to the users. __From the user's aspect, system that use TurboMind has access to infinite device memory.__

## LLaMa implementation

Our implementation of the LLaMa family models is modified from Gpt-NeoX model in FasterTransformer. In addition to basic refactoring and modifications to support the LLaMa family, we made some improvements to enable high performance inference of conversational models, most importantly:

- To support fast context decoding in multi-round conversations. We replaced the attention implementation in context decoder with a [cutlass](https://github.com/NVIDIA/cutlass)-based FMHA implementation that supports mismatched Q/K lengths.
- We introduced indirect buffer pointers in both context FMHA and generation FMHA to support the discontinuity in KV cache within the batch.
- To support concurrent inference with persistent batch, new synchronization mechanism was designed to orchestrate the worker threads running in tensor parallel mode.
- To maximize the throughput, we implement INT8 KV cache support to increase the max batch size. It's effective because in real-world serving scenarios, KV cache costs more memory and consumes more memory bandwidth than weights or other activations.
- We resolved an NCCL hang issue when running multiple model instances in TP mode within a single process, NCCL APIs are now guarded by host-side synchronization barriers.

## API

TurboMind supports a Python API that enables streaming output and tensor parallel mode. 

The ability to use [tritonserver](https://github.com/triton-inference-server/server) for serving is also inherited from FasterTransformer. However, to support submitting concurrent requests into our persistent batch model, we no longer use sequence batching or dynamic batching as FasterTransformer does. The bookkeeping of request and sequence states are managed by TurboMind instead.

## Difference between FasterTransformer and TurboMind

Apart of the features described above, there are still many minor differences that we don't cover in this document. Notably, many capabilities of FT are dropped in TurboMind because of the difference in objectives (e.g. prefix prompt, beam search, context embedding, sparse GEMM, GPT/T5/other model families, etc)

## FAQ

### Supporting Huggingface models

For historical reasons, TurboMind's weight layout is based on [the original LLaMa implementation](https://github.com/facebookresearch/llama) (differ only by a transpose). The implementation in huggingface transformers uses a [different layout](https://github.com/huggingface/transformers/blob/45025d92f815675e483f32812caa28cce3a960e7/src/transformers/models/llama/convert_llama_weights_to_hf.py#L123C76-L123C76) for `W_q` and `W_k` which is handled in [deploy.py](/lmdeploy/serve/turbomind/deploy.py#L362).
