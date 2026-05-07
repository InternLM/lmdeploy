import argparse

from compressed_tensors.offload import dispatch_model
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Run FP8 quantization for Qwen3 model')

    parser.add_argument('--work-dir',
                        type=str,
                        default='./qwen3_30b_a3b_fp8',
                        required=True,
                        help='The directory to save the quantized model')

    parser.add_argument('--model-id',
                        type=str,
                        default='Qwen/Qwen3-30B-A3B',
                        help='The Hugging Face model ID to quantize')
    return parser.parse_args()

def main():
    # 1. Achieve command args
    args = parse_args()
    MODEL_ID = args.model_id
    SAVE_DIR = args.work_dir

    print(f'Loading model: {MODEL_ID}')
    print(f'Saving to: {SAVE_DIR}')

    # 2. Load_dataset and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype='auto', device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 3. Configure quant args
    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per channel via ptq
    #   * quantize the activations to fp8 with dynamic per token
    recipe = QuantizationModifier(
        targets='Linear',
        scheme='FP8_BLOCK',
        ignore=['lm_head', 're:.*mlp.gate$'],
    )

    # 4. Run quantization
    print('Starting quantization...')
    oneshot(model=model, recipe=recipe)

    # 5. Confirm generations of the quantized model look sane
    print('========== SAMPLE GENERATION ==============')
    dispatch_model(model)
    input_ids = tokenizer('Hello my name is', return_tensors='pt').input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print('==========================================')

    # 6. Save quantized model
    print('Saving model...')
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == '__main__':
    main()
