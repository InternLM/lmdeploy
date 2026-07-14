# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

NUM_LOADED_SAMPLES = 30000


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# adapted from https://github.com/vllm-project/llm-compressor/blob/main/tests/testing_utils.py
def process_dataset(ds, tokenizer, max_seq_length):
    """Helper function to preprocess and tokenize a dataset according to
    presets.

    Args:
        ds: Language dataset to preprocess and tokenize.
        tokenizer: Tokenizer to encode text.
        max_seq_length: Maximum sequence length of samples.

    Returns:
        ds: Tokenized dataset.
    """
    ds_name = ds.info.dataset_name.lower()
    if ds_name == 'gsm8k':

        def tokenize(sample):
            return tokenizer(
                sample['question'],
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
                                  f'Only `gsm8k`, `open-platypus`, `calibration`, `openwebtext` '
                                  f'are supported by preprocess. ')

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds


def get_wikitext2(dataset, tokenizer, nsamples, seed, seqlen):
    """Load Wikitext-2 train and test datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    trainenc = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    return trainloader


def get_c4(dataset, tokenizer, nsamples, seed, seqlen):
    """Load C4 train and validation datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader


def get_pileval(dataset, tokenizer, nsamples, seed, seqlen=512):
    """Load pileval train dataset and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """

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


def get_gsm8k(dataset, tokenizer, nsamples, seed, seqlen):
    """Load GSM8K train and test datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    dataset = dataset.shuffle(seed=seed)
    dataset = process_dataset(dataset, tokenizer, seqlen)

    # GSM8K samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in dataset], dtype=torch.long)
    avg_tokens = lengths.sum().item() // len(dataset)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(dataset)):
        line = dataset[i]['input_ids']
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


def get_neuralmagic_calibration(dataset, tokenizer, nsamples, seed, seqlen):
    """Load neuralmagic_calibration train and test datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    dataset = dataset.shuffle(seed=seed)
    dataset = process_dataset(dataset, tokenizer, seqlen)

    # neuralmagic_calibration samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in dataset], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(dataset)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(dataset)):
        line = dataset[i]['input_ids']
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


def get_open_platypus(dataset, tokenizer, nsamples, seed, seqlen):
    """Load open-platypus train and test datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    dataset = dataset.shuffle(seed=seed)
    dataset = process_dataset(dataset, tokenizer, seqlen)

    # open-platypus samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in dataset], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(dataset)
    needed_samples = max(1, int((seqlen * nsamples) // avg_tokens))

    samples = []
    n_run = 0
    for i in range(len(dataset)):
        line = dataset[i]['input_ids']
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


def get_openwebtext(dataset, tokenizer, nsamples, seed, seqlen):
    """Load openwebtext train and test datasets and tokenize.

    Args:
        dataset: calib dataset
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    dataset = dataset.shuffle(seed=seed)
    dataset = process_dataset(dataset, tokenizer, seqlen)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = dataset[i]
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
        name: Dataset name ('wikitext2', 'c4', 'pileval', 'gsm8k',
                'neuralmagic_calibration', 'open-platypus', 'openwebtext').
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        List of sampled and tokenized training examples.
    """
    from datasets import VerificationMode, load_dataset
    if 'wikitext2' in name:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        return get_wikitext2(dataset, tokenizer, nsamples, seed, seqlen)

    if 'c4' in name:
        dataset = load_dataset('allenai/c4',
                               'en',
                               data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                               split='train',
                               verification_mode=VerificationMode.NO_CHECKS)
        return get_c4(dataset, tokenizer, nsamples, seed, seqlen)

    if 'pileval' in name:
        from datasets.builder import DatasetGenerationError
        try:
            dataset = load_dataset('mit-han-lab/pile-val-backup', split=f'validation[:{NUM_LOADED_SAMPLES}]')
        except DatasetGenerationError:
            raise InterruptedError('There have been some issues when generating '
                                   'the dataset, you could try to download it '
                                   'locally first, and replace the `data_files`'
                                   'with local addresses or use other datasets '
                                   '(c4, wiki, ptb).')
        return get_pileval(dataset, tokenizer, nsamples, seed, seqlen)

    if 'gsm8k' in name:
        dataset = load_dataset('openai/gsm8k', 'main', split='train')
        return get_gsm8k(dataset, tokenizer, nsamples, seed, seqlen)

    if 'neuralmagic_calibration' in name:
        dataset = load_dataset('neuralmagic/calibration', 'LLM', split='train')
        return get_neuralmagic_calibration(dataset, tokenizer, nsamples, seed, seqlen)

    if 'open-platypus' in name:
        dataset = load_dataset('garage-bAInd/Open-Platypus', split='train')
        return get_open_platypus(dataset, tokenizer, nsamples, seed, seqlen)

    if 'openwebtext' in name:
        dataset = load_dataset('Skylion007/openwebtext',
                               data_files={'train': 'plain_text/train-00000-of-00080.parquet'},
                               split=f'train[:{NUM_LOADED_SAMPLES}]',
                               verification_mode=VerificationMode.NO_CHECKS)
        return get_openwebtext(dataset, tokenizer, nsamples, seed, seqlen)
