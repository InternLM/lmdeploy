import argparse
import base64
import itertools
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from io import BytesIO

import torch
from datasets import concatenate_datasets, load_dataset
from openai import OpenAI
from tqdm import tqdm

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}


def encode_image(image):
    # with open(image, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode('utf-8')
    if isinstance(image, str):
        if image.startswith('/'):
            with open(image, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            image_data = tcs_loader(image)
            buffered = BytesIO()
            image_data.save(buffered, format='PNG')
            # image_data.save(os.path.basename(image),format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        image_data = image
        buffered = BytesIO()
        image_data.save(buffered, format='PNG')
        # image_data.save(os.path.basename(image),format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)
    images = [
        data['image_1'], data['image_2'], data['image_3'], data['image_4'],
        data['image_5'], data['image_6'], data['image_7']
    ]
    return {
        'id': data['id'],
        'question': question,
        'options': data['options'],
        'answer': data['answer'],
        'image': images,
        'question_type': data['question_type']
    }


def openai_chat(client,
                question,
                model_name,
                generate_config,
                image=None,
                sys_prompt=None):

    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': question,
            },
        ],
    }]
    if sys_prompt:
        messages = [{'role': 'system', 'content': sys_prompt}] + messages
    # start_time = time.time()
    if image:
        base64_images = [encode_image(image=img) for img in image]
        # print("读取图片时间：", time.time()-start_time)
        # start_time = time.time()
        # execution_time = end_time - start_time
        for base64_image in base64_images:
            messages[-1]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{base64_image}',
                    'max_dynamic_patch': 12
                }
            })
    count = 0
    while count < 5:
        try:
            response = client.chat.completions.create(model=model_name,
                                                      messages=messages,
                                                      **generate_config)
            return response.choices[0].message.content

        except Exception as e:
            count += 1
            print(e)
            time.sleep(1)


ds_collections = {
    'MMMU_validation': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'validation'
    },
    'MMMU_test': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'test'
    },
    'MMMU_dev': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'dev'
    },
}

IMAGE_TOKEN = '<IMAGE_TOKEN>'


def collate_fn(batches, tokenizer=None):
    images = [_['image'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    question_types = [_['question_type'] for _ in batches]
    return images, questions, answers, data_ids, options, question_types


vars_to_remove = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']


def unset_env_vars(vars_to_unset):
    """Temporarily unset specific environment variables."""
    removed_vars = {}
    for var in vars_to_unset:
        if var in os.environ:
            # 保存当前环境变量的值，以便稍后恢复
            removed_vars[var] = os.environ.pop(var)
    return removed_vars


def restore_env_vars(removed_vars):
    """Restore the previously unset environment variables."""
    os.environ.update(removed_vars)


class MMMUDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        split,
    ):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values()):

            sub_dataset = load_dataset(root,
                                       subject,
                                       split=split,
                                       cache_dir=os.path.join(os.getcwd()))
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = process_single_sample(self.data[idx])
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type']

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'
        ]
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)

        images = []
        for idx, pil_image in enumerate(pil_images):
            if pil_image is not None:

                images.append(pil_image)

        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)

        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        # question += '\n' + self.prompt[question_type]
        question = question.strip()

        return {
            'question': question,
            'image': images,
            'answer': answer,
            'option': options,
            'data_id': data_id,
            'question_type': question_type
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    prompt = {
        'multiple-choice':
        "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }

    cot_prompt = {
        'multiple-choice': (
            r"Answer the preceding multiple choice question. The last line of your response should follow this format: 'Answer: \boxed{$LETTER}' (without quotes), "
            'where LETTER is one of the options. If you are uncertain or the problem is too complex, '
            'make a reasoned guess based on the information provided. '
            'Avoid repeating steps indefinitely—provide your best guess even if unsure. '
            'Think step by step logically, considering all relevant information before answering.'
            '\n\n'
            'Question:'
            '\n\n'
            # "{question}"
        ),
        'open': (
            r"Answer the preceding question. The last line of your response should follow this format: 'Answer: \boxed{$FINAL_ANSWER}' (without quotes),' "
            "where 'FINAL_ANSWER' is your conclusion based on the reasoning provided. "
            'If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. '
            'Avoid repeating steps indefinitely—provide your best guess even if unsure. '
            'Think step by step logically, considering all relevant information before answering.'
            '\n\n'
            'Question:'
            '\n\n'
            # "{question}"
        )
    }

    random.seed(args.seed)
    # if args.cot:
    #     prompt['multiple-choice']=''
    #     prompt['open']=''
    for ds_name in args.datasets:
        dataset = MMMUDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )

        outputs = []
        # count = 0
        for _, (images, questions, answers, data_ids, options,
                question_types) in tqdm(enumerate(dataloader)):

            generation_config = dict(
                max_tokens=ds_collections[ds_name]['max_new_tokens']
                if not args.cot else 20000,
                temperature=args.temperature,
            )
            if args.cot:
                question = cot_prompt[question_types[0]] + questions[0]
            else:
                question = questions[0] + prompt[question_types[0]]

            if len(images) > 1:
                question = '\n'.join(
                    [f'Image-{i}: {IMAGE_TOKEN}'
                     for i in range(len(images))]) + '\n' + question
            # print(f'[evaluate_chat_model] {len(images)} images')
            # print(f'[evaluate_chat_model] {question}')

            with ThreadPoolExecutor(max_workers=args.infer_times +
                                    1) as executor:
                now = time.perf_counter()
                futures = [
                    executor.submit(openai_chat, clients[0], question,
                                    model_name, generation_config, images[0])
                    for i in range(args.infer_times)
                ]
                print(
                    f'[evaluate_chat_model] submitting {len(futures)} tasks cost {time.perf_counter() -now } s'
                )
                now = time.perf_counter()
                k_preds = [future.result() for future in futures]
                print(
                    f'[evaluate_chat_model] waiting for {len(k_preds)} predictions cost {time.perf_counter() -now } s'
                )
                # print(f'[evaluate_chat_model] predictions: {k_preds}')

            print(
                f'[evaluate_chat_model] random choose predition {random.choice(k_preds)}'
            )
            k_preds = [k_preds]

            # if len(options[0]) == 0:
            #     preds = [pred]
            # else:
            #     preds = [post_process(pred, options[0])]

            for question, pred, answer, data_id, option in zip(
                    questions, k_preds, answers, data_ids, options):
                outputs.append({
                    'question': question,
                    'output': pred,
                    'gt_answers': answer,
                    'data_id': data_id,
                    'option': option
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs,
                                            json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [
            _ for _ in itertools.chain.from_iterable(merged_outputs)
        ]

        if torch.distributed.get_rank() == 0:

            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{args.infer_times}_cot_{time_prefix}.json' if args.cot else f'{ds_name}_{args.infer_times}_{time_prefix}.json'
            output_path = os.path.join(args.out_dir, results_file)
            outputs = {}
            for item in merged_outputs:
                outputs[item['data_id']] = item
            with open(output_path, 'w') as f:
                json.dump(outputs, f, indent=4)
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MMMU_dev')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--infer-times', type=int, default=32)
    parser.add_argument('--url', type=str, default='lmdeploy_url')
    parser.add_argument('--model-name', type=str, default='InternVL2_5-78B')
    args, unknown_args = parser.parse_known_args()

    if args.infer_times > 1:
        args.num_beams = 1
        args.temperature = 1.0
    else:
        args.temperature = 0.0

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(backend='gloo',
                                         world_size=int(
                                             os.getenv('WORLD_SIZE', '1')),
                                         rank=int(os.getenv('RANK', '0')),
                                         timeout=timedelta(days=1))

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    urls = [
        # "http://10.140.66.137:10000/v1",
        args.url
    ]
    model_name = args.model_name
    clients = [
        OpenAI(api_key='YOUR_API_KEY', base_url=url, timeout=None)
        for url in urls
    ]

    evaluate_chat_model()
