# Architecture of lmdeploy.pytorch

`lmdeploy.pytorch` is an inference engine in LMDeploy. It provides a developer friendly framework to users who want to deploy their own model and develop new features.

## Design

![pytorch arch](https://github.com/grimoire/lmdeploy/blob/media/lmdeploy_pytorch_arch.png?raw=true)

## API

`lmdeploy.pytorch` share service interfaces with `Turbomind`, these interfaces perform inference through `Engine` and `EngineInstance` in lmdeploy.pytorch.

EngineInstance is the inference request sender, it will pack the inference request and send the packed request to Engine. EngineInstance is thread-safe, multiple threads can send request through their own EngineInstance simultaneously. Engine will perform batching automatically according to resources usage.

Engine is the request receiver and executor. It contain modules that support the task as follow:

- `ModelAgent` is a wrapper of the model. It is responsible for loading model/adapters, cache management and tensor parallelism.
- `Scheduler` is the sequence manager. It will decide which sequences and adapters would participated in current step, then allocate resources for them.
- `RequestManager` is responsible for request sending and receiving. It is the bridge between Engine and EngineInstance.

## Engine

Engine would response the requests in a sub-thread, looping as following:

1. Get new requests through RequestManager. These requests would be cached.
2. Scheduler perform scheduling, decide which cached requests should be processed and allocate resources for them.
3. ModelAgent would swap the caches according to the information provided by Scheduler, then performing inference with the patched model.
4. Scheduler update the status of requests according to the inference result of ModelAgent.
5. RequestManager response to the sender (EngineInstance), back to step 1.

Let's dive deeper into these modules.

### Scheduler

It is a common practice to cache history key and value states in LLM inference to prevent redundant computation. Since history lengths are different in batch of sequences, we have to padding the caches so we can perform the batching inference. The padding would waste a lot of memory and limit the performance of the transformer.

[vLLM](https://docs.vllm.ai) provide a paging based strategy, allocating caches in page blocks to prevent extra memory usage. The Scheduler module in our Engine share the same design, allocating resources according to the sequence length in blocks and evicting unused blocks to support larger batching and longer session length.

We also support [S-LoRA](https://github.com/S-LoRA/S-LoRA). S-LoRA can be used to support multiple LoRA adapters on limited memory.

### ModelAgent

lmdeploy.pytorch support Tensor Parallelism, which would leads to complex model initialization, cache allocation and weight partition. ModelAgent is designed to hide these details so Engine just need to focus on maintaining the pipeline.

ModelAgent is composed of two component:

1. `patched_model` is the transformer model after patch. Compared to the origin model, patched model has more features, such as TP, quantization and high performance kernels.
2. `cache_engine` is the maintainer of caches. It receive command from Scheduler, perform host-device page swap. Only gpu blocks can be used to cache key/value and adapters.

## Patching

In order to ease the deployment of new model, we have develop a tool to patch the modules.

Let's say, if we want to reimplement the forward of `LlamaAttention.forward`:

```python
class CustomLlamaAttention(nn.Module):
    def forward(self, ...):
        # custom forward
```

Just register the implementation above into `lmdeploy.pytorch.models.module_map`.

```python
MODULE_MAP.update({
'transformers.models.llama.modeling_llama.LlamaAttention':
'qualname.to.CustomLlamaAttention'})
```

ModelAgent would load and patch `LlamaAttention` with `CustomLlamaAttention` and leave anything other unchanged. Than you can perform inference with the new implementation.

## Features

lmdeploy.pytorch support new features include:

- Continuous Batching: Since the sequence length in a batch might be different, padding is required to support batching inference. Large padding leads to extra memory usage and useless computation. We use continuous batching, concatenate all sequence into a single long sequence to avoid padding.

- Tensor Parallelism: The GPU memory usage of LLM might be larger than the memory of a single GPU. Tensor parallelism can be used to fit such model on multiple devices. Each device has parts of the model and can be computed simultaneous, the result would be gathered to ensure the correctness.

- S-LoRA: LoRA adapter can be used to support training LLM on device with limited memory. It is a common practice to merge adapter into weights of the model before deployment, load multiple adapter in such way would consume a lot of memory. We have support S-LoRA, adapters would be paged and swapped in when necessary, special kernels are developed to support inference with unmerged adapters. Which made it possible to load a lot of different adapters.

- Quantization: Model quantization perform computation with low precision. lmdeploy.pytorch has support w8a8 quantization. Read [w8a8](../quantization/w8a8.md) for more details.
