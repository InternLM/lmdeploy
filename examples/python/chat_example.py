import fire
from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from transformers import AutoTokenizer
import random


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(model_name, model_path, tokenizer_model_path, session_id: int = 1):
    tm_model = tm.TurboMind(model_path)
    generator = tm_model.create_instance()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    model = MODELS.get(model_name)()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)

    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            pass
        else:
            prompt = model.get_prompt(prompt, nth_round == 1)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    request_output_len=512,
                    sequence_start=(nth_round == 1),
                    sequence_end=False,
                    step=step,
                    stop=False,
                    top_k=40,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.05,
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
                res, tokens = outputs[0]
                # decode res
                response = tokenizer.decode(
                    res[step:], skip_special_tokens=True)
                print(f'session {session_id}, {tokens}, {response}')
                # update step
                step = tokens - 1

        nth_round += 1


if __name__ == '__main__':
    fire.Fire(main)
