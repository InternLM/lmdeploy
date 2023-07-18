from lmdeploy.turbomind.tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer('huggyllama/llama-7b')

    prompts = ['cest la vie', '上帝已死']
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        output = tokenizer.decode(tokens)
        print(output)


if __name__ == '__main__':
    main()
