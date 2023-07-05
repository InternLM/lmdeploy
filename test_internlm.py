import re

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, StoppingCriteria,
                          StoppingCriteriaList)

torch.set_default_device('cuda')


class Decorator:

    def decorate(self, prompt):
        return prompt

    def extract(self, gen_out):
        return gen_out


class InternLMDecorator(Decorator):
    regex = re.compile(r'<\|Bot\|>:(.*)')

    def decorate(self, prompt):
        return f'<|User|>:{prompt}<TOKENS_UNUSED_0>'

    def extract(self, gen_out):
        return self.regex.search(gen_out).group(1)


class InternLMStoppingCriteria(StoppingCriteria):

    def __call__(self, input_ids: torch.LongTensor, *args, **kwargs) -> bool:
        print(input_ids.shape)
        return input_ids[0, -1] in [2, 103028]


stop_criteria = InternLMStoppingCriteria()
stopping_criteria = StoppingCriteriaList([stop_criteria])

tokenizer = AutoTokenizer.from_pretrained('/share_140/InternLM/7B/0705/5299/',
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    '/share_140/InternLM/7B/0705/5299/',
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

d = InternLMDecorator()

prompt = 'Hey, are you conscious? Can you talk to me?'
prompt = d.decorate(prompt)
inputs = tokenizer(prompt, return_tensors='pt')
print(inputs)

gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)

for _ in range(2):
    # Generate
    generate_ids = model.generate(inputs.input_ids, gen_config)

    out = tokenizer.batch_decode(generate_ids,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]

    print(generate_ids)
    print(out)
    print('+' * 100)
    print(d.extract(out))
