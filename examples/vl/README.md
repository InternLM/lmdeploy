# Vision-Language Web Demo

A chatbot demo with image input.

## Supported Models

- [InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer/tree/main)
- [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)

## Quick Start

### internlm/internlm-xcomposer-7b

- extract llm model from huggingface model
  ```python
  python extract_xcomposer_llm.py
  # the llm part will saved to internlm_model folder.
  ```
- lanuch the demo
  ```python
  python app.py --model-name internlm-xcomposer-7b --llm-ckpt internlm_model
  ```

### Qwen-VL-Chat

- lanuch the dmeo
  ```python
  python app.py --model-name qwen-vl-chat --hf-ckpt Qwen/Qwen-VL-Chat
  ```

## Limitations

- this demo the code in their repo to extract image features that might not very efficiency.
- this demo only contains the chat function. If you want to use localization ability in Qwen-VL-Chat or article generation function in InternLM-XComposer, you need implement these pre/post process. The difference compared to chat is how to build prompts and use the output of model.
