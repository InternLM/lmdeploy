from lmdeploy.pytorch.model import accel_model, init_model


def test_init_model():
    cprint = lambda x: print(f'\033[92m{x}\033[0m')  # noqa: E731

    # Test llama2-7b
    for model_path in ['llama2/huggingface/llama-2-7b', 'internlm-7b']:
        model, tokenizer = init_model(model_path)
        assert tokenizer.is_fast
        cprint('llama2 on CPU')
        print(model)
        model1 = accel_model(model)
        cprint('llama2 on GPU')
        print(model1)
        cprint('llama2 with kernel injection')
        model2 = accel_model(model, accel='deepspeed')
        assert 'DeepSpeedSelfAttention' in repr(model2)
        assert 'DeepSpeedMLP' in repr(model2)
