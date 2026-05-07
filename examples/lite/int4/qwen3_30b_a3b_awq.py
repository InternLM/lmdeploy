import argparse

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Run AWQ quantization for Qwen3 model')

    parser.add_argument('--work-dir',
                        type=str,
                        default='./qwen3_30b_a3b_awq',
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

    # 3. Prepare calibration dataset
    DATASET_ID = 'neuralmagic/calibration'
    DATASET_SPLIT = 'train'
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 512

    def get_calib_dataset(tokenizer):
        ds = load_dataset(
            DATASET_ID,
            'LLM',
            split=f'{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]',
        )

        def preprocess(example):
            messages = []
            for message in example['messages']:
                if message['role'] == 'user':
                    messages.append({'role': 'user', 'content': message['content']})
                elif message['role'] == 'assistant':
                    messages.append({'role': 'assistant', 'content': message['content']})

            return tokenizer(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                ),
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=False,
            )

        ds = (ds.shuffle(seed=42).map(preprocess,
                                      remove_columns=ds.column_names).select(range(NUM_CALIBRATION_SAMPLES)))
        return ds

    # 4. Configure quant args (W4A16_ASYM AWQ)
    recipe = [
        AWQModifier(
            ignore=['lm_head', 're:.*mlp.gate$'],
            scheme='W4A16_ASYM',
            targets=['Linear'],
            duo_scaling='both',
        ),
    ]

    # 5. Run quantization
    print('Starting quantization...')
    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        log_dir=None,
    )

    # 6. Save quantized model
    print('Saving model...')
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f'Successfully saved to {SAVE_DIR}')


if __name__ == '__main__':
    main()
