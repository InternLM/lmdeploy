from typing import List
import fire
import sys


class Tokenizer:
    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            from sentencepiece import SentencePieceProcessor
            self.model = SentencePieceProcessor(model_file=model_file)
            self.vocab_size = self.model.vocab_size()
            self.start_id = self.model.bos_id()
            self.end_id = self.model.eos_id()
            self.pad_id = self.model.pad_id()
        else:
            from transformers import AutoTokenizer
            self.model = AutoTokenizer.from_pretrained(model_file)
            self.vocab_size = self.model.vocab_size
            self.start_id = self.model.bos_token_id
            self.end_id = self.model.eos_token_id
            self.pad_id = self.model.pad_token_id

        print(f'vocab_size = {self.vocab_size}')
        print(f'start_id = {self.start_id}')
        print(f'end_id = {self.end_id}')
        print(f'pad_id = {self.pad_id}')

    def encode(self, s: str):
        if hasattr(self.model, 'Encode'):
            return self.model.Encode(s, add_bos=True)
        else:
            return self.model.encode(s, add_special_tokens=True)

    def decode(self, t: List[int]):
        if hasattr(self.model, 'Decode'):
            return self.model.Decode(t)
        else:
            return self.model.decode(t)


def main(model_file: str = '/data/llama/model/tokenizer.model',
         encode_file: str = None, decode_file: str = None):
    tokenizer = Tokenizer(model_file)
    if encode_file:
        with open(encode_file, 'r') as f:
            xs = tokenizer.encode(f.read())
            print(','.join(map(str, xs)))
    elif decode_file:
        with open(decode_file, 'r') as f:
            ys = tokenizer.decode(f.read())
            print(ys)
    else:
        first = True
        while True:
            try:
                s = input()
            except EOFError:
                break
            if not first:
                print('---------------------------------------------')
            first = False
            try:
                xs = map(int, s.strip().split(' '))
                s = tokenizer.decode(list(xs))
                print(s)
            except ValueError:
                xs = tokenizer.encode(s)
                print(' '.join(map(str, xs)))


if __name__ == '__main__':
    fire.Fire(main)