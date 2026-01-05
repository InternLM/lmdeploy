# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import ProcessorMixin


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def process_dataset(ds: Dataset, processor: ProcessorMixin, max_seq_length: int) -> Dataset:
    """Helper function to preprocess and tokenize a dataset according to
    presets.

    :param ds: language dataset to preprocess and tokenize
    :param tokenizer: tokenizer to be used for tokenization
    :param max_seq_length: maximum sequence length of samples
    """
    ds_name = ds.info.dataset_name.lower()
    if ds_name == 'gsm8k':

        def process(sample):
            return processor(
                sample['question'],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == 'ultrachat_200k':

        def process(sample):
            return processor(
                processor.apply_chat_template(
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
        def process(sample):
            messages = [{
                'role': 'user',
                'content': sample['instruction'] + ' ' + sample['input']
            }, {
                'role': 'assistant',
                'content': sample['output']
            }]
            return processor(
                processor.apply_chat_template(
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

        def process(example):
            messages = []
            for message in example['messages']:
                if message['role'] == 'user':
                    messages.append({'role': 'user', 'content': message['content']})
                elif message['role'] == 'assistant':
                    messages.append({'role': 'assistant', 'content': message['content']})

            return processor(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                ),
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    elif ds_name == 'openwebtext':

        def process(sample):
            return processor(
                sample['text'],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

    else:
        raise NotImplementedError(f'Cannot preprocess dataset {ds.info.dataset_name}')

    ds = ds.map(process, remove_columns=ds.column_names)

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
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)

    trainenc = tokenizer('\n\n'.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


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
    from datasets import VerificationMode
    traindata = load_dataset('allenai/c4',
                             'en',
                             data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                             split='train',
                             verification_mode=VerificationMode.NO_CHECKS)
    valdata = load_dataset('allenai/c4',
                           'en',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation',
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
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


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
    from datasets.builder import DatasetGenerationError
    try:
        dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation', trust_remote_code=True)
    except DatasetGenerationError:
        raise InterruptedError('There have been some issues when generating '
                               'the dataset, you could try to download it '
                               'locally first, and replace the `data_files`'
                               'with local addresses or use other datasets '
                               '(c4, wiki, ptb).')

    # pileval samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    max_keep = 20000
    samples_encode = []
    lengths = []
    for data in dataset:
        ids = tokenizer.encode(data['text'].strip())
        if not ids or len(ids) > 512:
            continue
        samples_encode.append(torch.tensor([ids]))
        lengths.append(len(ids))
        if len(samples_encode) >= max_keep:
            break

    avg_tokens = sum(lengths) / len(lengths)
    needed_samples = (seqlen * nsamples) // avg_tokens

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
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)], None


def get_ultrachat_200k(processor, nsamples, seed, seqlen):
    """Load ultrachat_200k train and test datasets and tokenize.

    Args:
        processor: Processor to apply chatplate encoding and encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    # from datasets import load_dataset, VerificationMode
    from datasets import VerificationMode
    train_data = load_dataset('HuggingFaceH4/ultrachat_200k',
                              data_files={'train_sft': 'data/train_sft-00000-of-00003-a3ecf92756993583.parquet'},
                              split='train_sft',
                              verification_mode=VerificationMode.NO_CHECKS)
    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, processor, seqlen)

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

    return trainloader, None


def get_gsm8k(processor, nsamples, seed, seqlen):
    """Load GSM8K train and test datasets and tokenize.

    Args:
        processor: Processor to apply chatplate encoding and encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    train_data = load_dataset('openai/gsm8k', 'main', split='train')
    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, processor, seqlen)

    # GSM8K samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() // len(train_data)
    needed_samples = (seqlen * nsamples) // avg_tokens

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
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)], None


def get_neuralmagic_calibration(processor, nsamples, seed, seqlen):
    """Load neuralmagic_calibration train and test datasets and tokenize.

    Args:
        processor: Processor to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    # from datasets import load_dataset
    train_data = load_dataset('neuralmagic/calibration', 'LLM', split='train')
    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, processor, seqlen)

    # neuralmagic_calibration samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(train_data)
    needed_samples = (seqlen * nsamples) // avg_tokens

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
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)], None


def get_open_platypus(processor, nsamples, seed, seqlen):
    """Load open-platypus train and test datasets and tokenize.

    Args:
        processor: Processor to apply chatplate encoding and encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    train_data = load_dataset('garage-bAInd/Open-Platypus', split='train')
    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, processor, seqlen)

    # open-platypus samples have far fewer tokens than seqlen; recompute how many
    # train items to select so it can still yield enough samples after concatenation.
    lengths = torch.tensor([len(sample['input_ids']) for sample in train_data], dtype=torch.long)
    avg_tokens = lengths.sum().item() / len(train_data)
    needed_samples = (seqlen * nsamples) // avg_tokens

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
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)], None


def get_openwebtext(processor, nsamples, seed, seqlen):
    """Load openwebtext train and test datasets and tokenize.

    Args:
        processor: Processor to apply chatplate encoding and encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: None.
    """
    from datasets import VerificationMode
    train_data = load_dataset('Skylion007/openwebtext',
                              data_files={'train': 'plain_text/train-00000-of-00080.parquet'},
                              split='train',
                              verification_mode=VerificationMode.NO_CHECKS)
    train_data = train_data.shuffle(seed=seed)
    train_data = process_dataset(train_data, processor, seqlen)

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

    return trainloader, None


def get_calib_loaders(name, tokenizer, processor, nsamples=128, seed=0, seqlen=2048):
    """Get calibration data loaders for a dataset.

    Args:
      name: Dataset name ('wikitext2', 'c4', 'pileval', 'ultrachat_200k', 'gsm8k',
            'neuralmagic_calibration', 'open-platypus', 'openwebtext').
      tokenizer: Tokenizer to encode text.
      processor: Processor to apply chatplate encoding and encode text.
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
        return get_ultrachat_200k(processor, nsamples, seed, seqlen)

    if 'gsm8k' in name:
        return get_gsm8k(processor, nsamples, seed, seqlen)

    if 'neuralmagic_calibration' in name:
        return get_neuralmagic_calibration(processor, nsamples, seed, seqlen)

    if 'open-platypus' in name:
        return get_open_platypus(processor, nsamples, seed, seqlen)

    if 'openwebtext' in name:
        return get_openwebtext(processor, nsamples, seed, seqlen)
