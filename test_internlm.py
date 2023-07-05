import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

torch.set_default_device('cuda')

tokenizer = AutoTokenizer.from_pretrained('/share_140/InternLM/7B/0705/hf/',
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/share_140/InternLM/7B/0705/hf/',
                                             torch_dtype=torch.float16,
                                             trust_remote_code=True)

prompt = 'Hey, are you conscious? Can you talk to me?'
inputs = tokenizer(prompt, return_tensors='pt')

gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

# Generate
generate_ids = model.generate(inputs.input_ids, gen_config)
out = tokenizer.batch_decode(generate_ids,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]

print(out)
