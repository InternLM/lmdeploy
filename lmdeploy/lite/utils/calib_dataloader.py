# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import torch
from datasets import VerificationMode, load_dataset

NUM_LOADED_SAMPLES = 30000


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def find_first_files(root_dir: str, filename: str):
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path):
            result = find_first_files(full_path, filename)
            if result:
                return result
        else:
            if item == filename:
                return full_path
    return None


def find_full_path_name(root_dir: str, target_subdir: str):
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path) and target_subdir in item:
            return full_path, item
    return None, None


def bool_load_from_disk(dataset_name: str):
    from pathlib import Path
    cache_root = Path(os.environ.get('HF_DATASETS_CACHE', Path.home() / '.cache/huggingface/datasets'))
    dataset_fullpath, _ = find_full_path_name(str(cache_root), dataset_name)
    if dataset_fullpath is None:
        return False, None
    dataset_info_path = find_first_files(dataset_fullpath, 'dataset_info.json')
    if dataset_info_path is None:
        return False, None
    return True, dataset_info_path


def load_dataset_from_disk(dataset_info_path: str):
    import glob
    import json
    from pathlib import Path

    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    num_examples = 0
    for split in dataset_info.get('splits').values():
        if num_examples < split.get('num_examples', 0):
            num_examples = split.get('num_examples', 0)
            split_name = split['name']
            dataset_name = split.get('dataset_name', '')

    if num_examples == 0:
        return None, None

    if dataset_name:
        dataset_fullname = f'*{dataset_name}-{split_name}*.arrow'
    else:
        dataset_fullname = f'*{split_name}*.arrow'

    dataset_info_subpath = Path(dataset_info_path).parent
    dataset_fullpath = os.path.join(dataset_info_subpath, dataset_fullname)
    files = sorted(glob.glob(dataset_fullpath))

    if not files:
        return None, None

    print('Loading files matching:' + '\n'.join(files))

    dataset = load_dataset('arrow',
                           data_files={'train': files},
                           split=f'train[:{NUM_LOADED_SAMPLES}]',
                           verification_mode=VerificationMode.NO_CHECKS)

    return dataset, dataset_name.lower()


def load_dataset_(is_load_from_disk: str,
                  dataset_info_path: str,
                  path: str,
                  name=None,
                  data_files=None,
                  split=None,
                  verification_mode=None):
    if is_load_from_disk:
        train_data, dataset_name = load_dataset_from_disk(dataset_info_path)
        # Dataset_info.json is exists, but can't find dataset file locally，Forcing redownload
        if train_data is None:
            print('Dataset cache miss. Forcing redownload...')
            train_data = load_dataset(path,
                                      name,
                                      data_files=data_files,
                                      split=split,
                                      download_mode='force_redownload',
                                      verification_mode=verification_mode)
            dataset_name = train_data.info.dataset_name.lower()
    else:
        train_data = load_dataset(path, name, data_files=data_files, split=split, verification_mode=verification_mode)
        dataset_name = train_data.info.dataset_name.lower()

    return train_data, dataset_name


# adapted from https://github.com/vllm-project/llm-compressor/blob/main/tests/testing_utils.py
def process_dataset(ds, tokenizer, max_seq_length, ds_name):
    """Helper function to preprocess and tokenize a dataset according to
    presets.

    Args:
        ds: Language dataset to preprocess and tokenize.
        tokenizer: Tokenizer to encode text.
        max_seq_length: Maximum sequence length of samples.

    Returns:
        ds: Tokenized dataset.
    """
    if ds_name == 'gsm8k':

        def tokenize(sample):
            return tokenizer(
                sample['question'],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == 'ultrachat_200k':

        def tokenize(sample):

            return tokenizer(
                tokenizer.apply_chat_template(
                    sample['messages'],
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == 'open-platypus':
        # use the output rather than the instruction
        def tokenize(sample):
            messages = [{
                'role': 'user',
                'content': sample['instruction'] + ' ' + sample['input']
            }, {
                'role': 'assistant',
                'content': sample['output']
            }]
            return tokenizer(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    # "neuralmagic/calibration"
    elif ds_name == 'calibration':

        def tokenize(sample):
            messages = []
            for message in sample['messages']:
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
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == 'openwebtext':

        def tokenize(sample):
            return tokenizer(
                sample['text'],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    else:
        raise NotImplementedError(f'Cannot preprocess dataset {ds.info.dataset_name} '
                                  f'Only `gsm8k`, `ultrachat_200k`, `open-platypus` '
                                  f'`calibration`, `openwebtext` are supported by preprocess. ')

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds


def get_wikitext2(tokenizer, nsamples, seed, seqlen):
    """Load Wikitext-2 train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized Wikitext-2 test set.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('wikitext')
    traindata, _ = load_dataset_(is_load_from_disk,
                                 dataset_info_path,
                                 path='wikitext',
                                 name='wikitext-2-raw-v1',
                                 split=f'train[:{NUM_LOADED_SAMPLES}]')

    trainenc = tokenizer('\n\n'.join(traindata['text']), return_tensors='pt')
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    return trainloader


def get_c4(tokenizer, nsamples, seed, seqlen):
    """Load C4 train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('c4')
    traindata, _ = load_dataset_(is_load_from_disk,
                                 dataset_info_path,
                                 path='allenai/c4',
                                 name='en',
                                 data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                                 split=f'train[:{NUM_LOADED_SAMPLES}]',
                                 verification_mode=VerificationMode.NO_CHECKS)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader


def get_pileval(tokenizer, nsamples, seed, seqlen=512):
    """Load pileval train dataset and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('pile-val-backup')
    dataset, _ = load_dataset_(is_load_from_disk,
                               dataset_info_path,
                               path='mit-han-lab/pile-val-backup',
                               split=f'validation[:{NUM_LOADED_SAMPLES}]')

    # pileval samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    samples_encode = []
    lengths = []
    for data in dataset:
        ids = tokenizer.encode(data['text'].strip())
        if not ids or len(ids) > 512:
            continue
        samples_encode.append(torch.tensor([ids]))
        lengths.append(len(ids))
        if len(samples_encode) >= len(dataset):
            break

    avg_tokens = sum(lengths) / len(lengths)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == needed_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)]


def get_ultrachat_200k(tokenizer, nsamples, seed, seqlen):
    """Load ultrachat_200k train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('ultrachat_200k')
    train_data, dataset_name = load_dataset_(
        is_load_from_disk,
        dataset_info_path,
        path='HuggingFaceH4/ultrachat_200k',
        data_files={'train_sft': 'data/train_sft-00000-of-00003-a3ecf92756993583.parquet'},
        split=f'train_sft[:{NUM_LOADED_SAMPLES}]',
        verification_mode=VerificationMode.NO_CHECKS)

    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, tokenizer, seqlen, dataset_name)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(train_data) - 1)
            trainenc = train_data[i]
            if len(trainenc['input_ids']) >= seqlen:
                break
        i = random.randint(0, len(trainenc['input_ids']) - seqlen)
        j = i + seqlen
        inp = trainenc['input_ids'][i:j]
        inp = torch.tensor([inp])
        trainloader.append(inp)

    return trainloader


def get_gsm8k(tokenizer, nsamples, seed, seqlen):
    """Load GSM8K train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('gsm8k')
    train_data, dataset_name = load_dataset_(is_load_from_disk,
                                             dataset_info_path,
                                             path='openai/gsm8k',
                                             name='main',
                                             split=f'train[:{NUM_LOADED_SAMPLES}]')

    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, tokenizer, seqlen, dataset_name)

    # GSM8K samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() // len(train_data)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(train_data)):
        line = train_data[i]['input_ids']
        sample = torch.tensor([line])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == needed_samples:
            break
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)]


def get_neuralmagic_calibration(tokenizer, nsamples, seed, seqlen):
    """Load neuralmagic_calibration train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('calibration')
    train_data, dataset_name = load_dataset_(is_load_from_disk,
                                             dataset_info_path,
                                             path='neuralmagic/calibration',
                                             name='LLM',
                                             split=f'train[:{NUM_LOADED_SAMPLES}]')

    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, tokenizer, seqlen, dataset_name)

    # neuralmagic_calibration samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(train_data)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(train_data)):
        line = train_data[i]['input_ids']
        sample = torch.tensor([line])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == needed_samples:
            break
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)]


def get_open_platypus(tokenizer, nsamples, seed, seqlen):
    """Load open-platypus train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('open-platypus')
    train_data, dataset_name = load_dataset_(is_load_from_disk,
                                             dataset_info_path,
                                             path='garage-bAInd/Open-Platypus',
                                             split=f'train[:{NUM_LOADED_SAMPLES}]')

    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, tokenizer, seqlen, dataset_name)

    # open-platypus samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(train_data)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(train_data)):
        line = train_data[i]['input_ids']
        sample = torch.tensor([line])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == needed_samples:
            break
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)]


def get_openwebtext(tokenizer, nsamples, seed, seqlen):
    """Load openwebtext train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    is_load_from_disk, dataset_info_path = bool_load_from_disk('openwebtext')
    train_data, dataset_name = load_dataset_(is_load_from_disk,
                                             dataset_info_path,
                                             path='Skylion007/openwebtext',
                                             data_files={'train': 'plain_text/train-00000-of-00080.parquet'},
                                             split=f'train[:{NUM_LOADED_SAMPLES}]',
                                             verification_mode=VerificationMode.NO_CHECKS)

    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, tokenizer, seqlen, dataset_name)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(train_data) - 1)
            trainenc = train_data[i]
            if len(trainenc['input_ids']) >= seqlen:
                break
        i = random.randint(0, len(trainenc['input_ids']) - seqlen)
        j = i + seqlen
        inp = trainenc['input_ids'][i:j]
        inp = torch.tensor([inp])
        trainloader.append(inp)

    return trainloader


def get_calib_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    """Get calibration data loaders for a dataset.

    Args:
      name: Dataset name ('wikitext2', 'c4', 'pileval', 'ultrachat_200k', 'gsm8k',
            'neuralmagic_calibration', 'open-platypus', 'openwebtext').
      tokenizer: Tokenizer to encode text.
      nsamples: Number of samples to take from train set.
      seed: Random seed for sampling.
      seqlen: Maximum sequence length.

    Returns:
      train_loader: List of sampled and tokenized training examples.
      test_data: Full tokenized validation set.
    """
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, nsamples, seed, seqlen)

    if 'c4' in name:
        return get_c4(tokenizer, nsamples, seed, seqlen)

    if 'pileval' in name:
        return get_pileval(tokenizer, nsamples, seed, seqlen)

    if 'ultrachat_200k' in name:
        return get_ultrachat_200k(tokenizer, nsamples, seed, seqlen)

    if 'gsm8k' in name:
        return get_gsm8k(tokenizer, nsamples, seed, seqlen)

    if 'neuralmagic_calibration' in name:
        return get_neuralmagic_calibration(tokenizer, nsamples, seed, seqlen)

    if 'open-platypus' in name:
        return get_open_platypus(tokenizer, nsamples, seed, seqlen)

    if 'openwebtext' in name:
        return get_openwebtext(tokenizer, nsamples, seed, seqlen)
