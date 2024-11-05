import argparse

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of the model in localhost or '
                        'the repo_id of the model in huggingface.co',
                        default='llava-hf/llava-interleave-qwen-7b-hf')
    parser.add_argument('--model-format',
                        type=str,
                        help='model format',
                        default='hf',
                        choices=['hf', 'awq'])
    parser.add_argument('--max-new-tokens',
                        type=int,
                        help='output max tokens number',
                        default=128)
    args = parser.parse_args()
    pipe = pipeline(
        args.model_path,
        backend_config=TurbomindEngineConfig(cache_max_entry_count=0.5,
                                             model_format=args.model_format),
        gen_config=GenerationConfig(max_new_tokens=args.max_new_tokens))

    image = load_image('https://qianwen-res.oss-cn-beijing.aliyuncs.com/' +
                       'Qwen-VL/assets/demo.jpeg')
    for prompt in ['Describe the image.', 'How many people in the image?']:
        print(f'prompt:{prompt}')
        response = pipe((prompt, image))
        print(response)
