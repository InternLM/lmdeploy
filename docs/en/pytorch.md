# Pytorch

## Chat in command line

LMDeploy support chatting with PyTorch models with submodule `lmdeploy.pytorch.chat`.

This submodule allow user to chat with language model through command line, and optionally accelerate model using backends like deepspeed.

**Example 1**: Chat with default setting

```shell
python -m lmdeploy.pytorch.chat $PATH_TO_HF_MODEL
```

**Example 2**: Disable sampling and chat history

```shell
python -m lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --temperature 0 --max-history 0
```

**Example 3**: Accelerate with deepspeed inference

```shell
python -m lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --accel deepspeed
```

Note: to use deepspeed, you need to install deepspeed, and if hope to accelerate InternLM, you need a customized version <https://github.com/wangruohui/DeepSpeed/tree/support_internlm_0.10.0>

**Example 4**: Tensor parallel the model on 2 GPUs

```shell
deepspeed --module --num_gpus 2 lmdeploy.pytorch.chat \
    $PATH_TO_LLAMA_MODEL_IN_HF_FORMAT \
    --accel deepspeed \
```

This module also allow the following control commands to change generation behaviors during chat.

- `exit`: terminate and exit chat
- `config set key=value`: change generation config `key` to `value`, e.g. config temperature=0 disable sampling for following chats
- `clear`: clear chat history

### Simple diagram of components

```mermaid
graph LR;
    subgraph model specific adapter
        p((user_input))-->tokenize-->id((input_ids))-->decorate
        tmpl_ids((template_ids))-->decorate;
    end
    subgraph generate
        model[CausalLM_model.generate]-->gen_result(("gen_result"))
        gen_result-->hid
        gen_result-->attn((attention))
    end
    subgraph streamer
        model-->s[streamer]--value-->decode_single--token-->output
    end
    subgraph session_manager
        prepend_history-->fullid((complete_ids));
        trim-->prepend_history
    end
    decorate-->prepend_history
    hid((history_ids))-->trim;
    attn-->trim;
    fullid-->model
    tokenizer((tokenizer))-->decode_single
    tokenizer-->tokenize
    p-->genconfig(GenConfig)-->model
```
