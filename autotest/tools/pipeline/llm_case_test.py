import fire

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)


def test():
    model_path = 'Qwen/Qwen3-8B-Base'
    backend_config = TurbomindEngineConfig(tp=1, communicator='nccl', quant_policy=0)

    pipe = pipeline(model_path, backend_config=backend_config)

    prompts = []
    prompts.append({'role': 'user', 'content': '你好，你是谁'})

    response = pipe([prompts], gen_config=gen_config, log_level='INFO', max_log_len=10)[0].text
    print(response)
    pipe.close()


if __name__ == '__main__':
    fire.Fire(test)
