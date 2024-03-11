import re
import string
import unicodedata
from collections import Counter

import numpy

TRANSLATION_TABLE_PUNCTUATION = str.maketrans('', '', string.punctuation)


def normalize(text: str,
              remove_punct: bool = True,
              lowercase: bool = True,
              nfd_unicode: bool = True,
              white_space: bool = True) -> str:
    """Normalize the text by lowercasing and removing punctuation."""
    # remove punctuation
    if remove_punct:
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION)

    # lowercase
    if lowercase:
        text = text.lower()

    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)

    return text


def form_ngrams(sequence, n):
    history = []
    # build the first ngram, yielding only when we have a full ngram
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1

    # yield each ngram we have, then add the next item and repeat
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def base_rps_frac_chars_in_dupe_ngrams(NGRAM_SIZE: int = 5, content: str = ''):
    """Base class for calculating the fraction of characters in duplicate word
    N-grams.

    This operates on the lower-cased, punctuation removed content. The function
    also ensures that characters in overlapping ngrams are only counted once.
    """
    normalized_content = normalize(content)
    normalized_words = tuple(normalized_content.split())

    if len(normalized_words) < NGRAM_SIZE:
        return 0

    # fetch the ngrams from the document if they exist, otherwise
    # compute them
    doc_n_grams = tuple(form_ngrams(iter(normalized_words), NGRAM_SIZE))

    # keep only ngrams which occur at least twice
    ngram_dupes = {
        ngram
        for ngram, count in Counter(doc_n_grams).items() if count > 1
    }

    duplicated_grams = numpy.zeros(len(normalized_words), dtype=int)
    i = 0
    for ngram in doc_n_grams:
        if ngram in ngram_dupes:
            duplicated_grams[i:i + NGRAM_SIZE] = 1

        i += 1

    word_lengths = numpy.array(list(map(len, normalized_words)))
    chars_duped = numpy.sum(word_lengths * duplicated_grams)
    total_chars = numpy.sum(word_lengths)

    if total_chars == 0:
        return 0

    score = float(chars_duped / total_chars) * 100
    return score
