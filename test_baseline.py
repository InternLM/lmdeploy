import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LlamaForCausalLM, LlamaTokenizer)

torch.set_default_device('cuda')

tokenizer = AutoTokenizer.from_pretrained('/nvme/wangruohui/llama-7b-hf/')
model = AutoModelForCausalLM.from_pretrained('/nvme/wangruohui/llama-7b-hf/',
                                             torch_dtype=torch.float16)

# tokenizer = LlamaTokenizer.from_pretrained("/nvme/wangruohui/llama-7b-hf/")
# model = LlamaForCausalLM.from_pretrained("/nvme/wangruohui/llama-7b-hf/")
# correct result

prompt = 'Hey, are you conscious? Can you talk to me?'
inputs = tokenizer(prompt, return_tensors='pt')

gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

# Generate
generate_ids = model.generate(inputs.input_ids, gen_config)
out = tokenizer.batch_decode(generate_ids,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]

print(out)
