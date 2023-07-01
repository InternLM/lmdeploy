from lmdeploy.turbomind.tokenizer import Tokenizer, Preprocessor, Postprocessor

def main():
    tokenizer = Tokenizer('huggyllama/llama-7b')
    preprocessor = Preprocessor(tokenizer)
    postprocessor = Postprocessor(tokenizer)

    prompts = ['cest la vie', '上帝已死']
    tokens = preprocessor(prompts)
    print(tokens)

    decode_prompts = postprocessor(*tokens)
    print(decode_prompts)

if __name__ == '__main__':
    main()
