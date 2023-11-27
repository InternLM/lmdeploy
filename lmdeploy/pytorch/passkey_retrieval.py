# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import random

from lmdeploy.model import MODELS
from lmdeploy.tokenizer import Tokenizer

from . import engine as tm
from .config import SchedulerConfig
from .messages import SamplingParam

os.environ['TM_LOG_LEVEL'] = 'ERROR'


class LLM(object):
    """LLM."""

    def __init__(self,
                 model_path: str,
                 model_name: str,
                 tp: int = 1,
                 max_session_len=40000) -> None:
        self.tokenizer = Tokenizer(model_path, trust_remote_code=True)

        self.scheduler_config = SchedulerConfig(
            max_batches=64,
            max_session_len=max_session_len,
            max_request_output_len=512)
        self.tm_model = tm.Engine(model_path,
                                  tp=tp,
                                  trust_remote_code=True,
                                  scheduler_config=self.scheduler_config)
        self.generator = self.tm_model.create_instance()
        self.model = MODELS.get(model_name)()
        seed = random.getrandbits(64)
        self.sampling_param = SamplingParam(
            top_k=40,
            top_p=0.8,
            temperature=0.8,
            repetition_penalty=1.0,
            ignore_eos=False,
            random_seed=seed,
        )
        self.session_id = 1

    def say(self, question: str):
        """say."""
        prompt = self.model.get_prompt(question, True)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = self.model.update_input_ids(input_ids)
        _, token_ids, __ = self.generator.infer(
            session_id=self.session_id,
            prompt_token_ids=input_ids,
            request_output_len=32,
            sampling_param=self.sampling_param)
        response = self.tokenizer.decode(token_ids)
        self.generator.end(self.session_id)
        self.session_id += 1
        return response

    def tokenize(self, question: str):
        """tokenize."""
        prompt = self.model.get_prompt(question, True)
        return self.tokenizer.encode(prompt)


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


def parse_config():
    """parse arguments."""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/models/openbuddy-llama2-13b-v8.1-fp16',
        help='LLM path, use /models/openbuddy-llama2-13b-v8.1-fp16 by default')
    parser.add_argument('--model_name',
                        type=str,
                        default='llama2',
                        help='LLM type name, use llama2 by default')
    parser.add_argument('--max_tokens',
                        type=int,
                        default=50000,
                        help='maximum token length for evaluation')
    parser.add_argument('--interval',
                        type=int,
                        default=1024,
                        help='interval for evaluation')
    parser.add_argument('--num_tests',
                        type=int,
                        default=1,
                        help='number of repeat testing for each length')
    args = parser.parse_args()
    return args


# copy from https://github.com/dvlab-research/LongLoRA/blob/main/passkey_retrivial.py  # noqa: E501
def generate_prompt_landmark(n_garbage=60000, seed=666):
    """Generates a text file and inserts an passkey at a random position."""
    from numpy import random as nprandom
    rnd_state = nprandom.get_state()
    nprandom.seed(seed)
    n_garbage_prefix = nprandom.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = 'There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.'  # noqa: E501
    garbage = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'  # noqa: E501
    garbage_inf = ' '.join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = nprandom.randint(1, 50000)
    information_line = f'The pass key is {pass_key}. Remember it. {pass_key} is the pass key.'  # noqa: E501
    final_question = 'What is the pass key? The pass key is'
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    nprandom.set_state(rnd_state)

    return '\n'.join(lines), str(pass_key)


def main(args):
    """main."""
    # Load model and tokenizer
    llm = LLM(model_path=args.model_path, model_name=args.model_name)

    all_accuries = {}
    # This is a rough ratio to control the number of texts and tokens
    for val in range(4096, args.max_tokens, args.interval):
        n_garbage = int(3.75 * val // 1024 * 1024)
        assert n_garbage > 0
        passed_tests = 0
        total_tokens = 0

        for j in range(args.num_tests):
            question, pass_key = generate_prompt_landmark(n_garbage=n_garbage,
                                                          seed=(val + j))
            response = llm.say(question)

            print(response)
            if pass_key in response:
                passed_tests += 1
            total_tokens += len(llm.tokenize(question=question))
        avg_tokens = total_tokens // args.num_tests
        accuracy = passed_tests / args.num_tests
        print('accuracy on the token length %d is %f' % (avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy

    print('accuries over tokens', all_accuries)


if __name__ == '__main__':
    args = parse_config()
    main(args)
