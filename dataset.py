from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter
from itertools import chain
import random

### You may import any Python standard libray here, but don't import external libraries such as torchtext, numpy, or nltk.
from collections import defaultdict
import numpy as np
import datetime
### END YOUR LIBRARIES

import torch

def preprocess(
    raw_src_sentence: List[str],
    raw_trg_sentence: List[str],
    src_word2idx: Dict[str, int],
    trg_word2idx: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int]]:
    """ Sentence preprocessor for Seq2Seq with Attention
    Before training the model, you should preprocess the data to feed Seq2Seq.
    Implement preprocessor with following rules.

    Preprocess Rules:
    1. All words should be converted into thier own index number by word2idx.
    1-1. If there is no matched word in word2idx, you should replace that word by <UNK> token.
    1-2. You have to use matched word2idx for each source/target language.

    2. You have to insert <SOS>, <EOS> tokens properly into the target sentence.
    2-1. You don't need them in source sentence.

    3. The length of preprocessed sentences should not exceed max_len.
    3-1. If the lenght of the sentence exceed max_len, you must truncate the back of the sentence.

    Note: as same with assign1, you can use only Python standard library and pyTorch.
    torchtext or torchvision is not a sub library of pyTorch. They are just another library by pyTorch team which needs to be installed separately.

    Arguments:
    raw_src_sentence -- raw source sentence without any modification
    raw_trg_sentence -- raw target sentence without any modification 
    src_word2idx -- dictionary for source language which maps words to their unique numbers
    trg_word2idx -- dictionary for target language which maps words to their unique numbers
    max_len -- maximum length of sentences

    Return:
    src_sentence -- preprocessed source sentence
    trg_sentence -- preprocessed target sentence

    """
    # Special tokens, use these notations if you want
    UNK = Language.UNK_TOKEN_IDX
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX

    ### YOUR CODE HERE (~2 lines, this is not a mandatory requirement, but try to make efficent codes)
    src_sentence: List[int] = None
    trg_sentence: List[int] = None

    # Truncate
    truncated_src_sentence = raw_src_sentence[:max_len] if len(raw_src_sentence) > max_len else raw_src_sentence
    truncated_tgt_sentence = raw_trg_sentence[:max_len-2] if len(raw_trg_sentence) > max_len-2 else raw_trg_sentence

    # word to idx
    src_sentence = [src_word2idx[word] if word in src_word2idx.keys() else UNK for word in truncated_src_sentence]
    trg_sentence = [trg_word2idx[word] if word in trg_word2idx.keys() else UNK for word in truncated_tgt_sentence]

    # insert <SOS> and append <EOS>
    trg_sentence.insert(0, SOS)
    trg_sentence.append(EOS)

    ### END YOUR CODE
    return src_sentence, trg_sentence

def bucketed_batch_indices(
        sentence_length: List[Tuple[int, int]],
        batch_size: int,
        max_pad_len: int
) -> List[List[int]]:
    """ Function for bucketed batch indices
    Although the loss calculation does not consider PAD tokens, it actually takes up GPU resources and degrades performance.
    Therefore, the number of <PAD> tokens in a batch should be minimized in order to maximize GPU utilization.
    Implement a function which groups samples into batches that satisfy the number of needed <PAD> tokens in each sentence is less than or equals to max_pad_len.

    Note 1: several small batches which have less samples than batch_size are okay but should not be many. If you pass the test, it means "okay".

    Note 2: you can directly apply this function to torch.utils.data.dataloader.DataLoader with batch_sampler argument.
    Read the test codes if you are interested in.

    Hint 1: The most easiest way for bucketing is sort-and-batch manner, but soon you will realize this doesn't work for this time.
    The second easiest way is binning, however one-dimensional binning is not enough because there are two sentences per a sample.

    Hint 2: defaultdict in collections library might be useful.

    Arguments:
    sentence_length -- list of (length of source_sentence, length of target_sentence) pairs.
    batch_size -- batch size
    max_pad_len -- maximum padding length. The number of needed <PAD> tokens in each sentence should not exceed this number.

    return:
    batch_indices_list -- list of indices to be a batch. Each element should contain indices of sentence_length list.

    Example:
    If sentence_length = [7, 4, 9, 2, 5, 10], batch_size = 3, and max_pad_len = 3,
    then one of the possible batch_indices_list is [[0, 2, 5], [1, 3, 4]]
    because [0, 2, 5] indices has simialr length as sentence_length[0] = 7, sentence_length[2] = 9, and sentence_length[5] = 10.
    """

    ### YOUR CODE HERE (~7 lines)
    batch_indices_list: List[List[int]] = None
    sentence_length = torch.LongTensor(sentence_length)
    min_dim0 = min(sentence_length[:,0])
    max_dim0 = max(sentence_length[:,0])
    min_dim1 = min(sentence_length[:,1])
    max_dim1 = max(sentence_length[:,1])
    dim0_bin = list(range(min_dim0, max_dim0, max_pad_len))
    dim1_bin = list(range(min_dim1, max_dim1, max_pad_len))

    bin =  []
    for i in dim0_bin:
        for j in dim1_bin:
            elem = (i,j)
            bin.append(torch.LongTensor(elem))
    bin = torch.stack(bin)
    diff = []
    for i in range(len(bin)):
        diff.append(sentence_length-bin[i])
    diff = torch.stack(diff)
    mask_dim1 = (diff[:, :, 0] <= 5) * (0 <= diff[:, :, 0])
    mask_dim2 = (diff[:, :, 1] <= 5) * (0 <= diff[:, :, 1])
    mask = mask_dim1 * mask_dim2
    mask = mask.transpose(0,1)
    bin_id_per_sentence = [int(mask[i].nonzero()[0]) for i in range(len(mask))]
    sentence_idx = range(len(sentence_length))
    dict = defaultdict(list)
    for d0, d1 in zip(bin_id_per_sentence,sentence_idx):
        dict[d0].append(d1)

    list1 = dict.values()
    list2 = []
    for bin in list1:
        n_batches = len(bin)//batch_size
        namuji = len(bin)%batch_size
        tmp_list = [bin[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        list2 += tmp_list
        if namuji != 0:
            last_batch = bin[n_batches*batch_size:]
            list2.append(last_batch)

    batch_indices_list = list2

    ### END YOUR CODE
    # Don't forget shuffling batches because length of each batch could be biased
    random.shuffle(batch_indices_list)

    return batch_indices_list


def collate_fn(
    batched_samples: List[Tuple[List[int], List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Collate function
    Because each sentence has variable length, you should collate them into one batch with <PAD> tokens.
    Implement collate_fn function which collates source/target sentence into source/target batchs appending <PAD> tokens behind
    Meanwhile, for the convenience of latter implementations, you should sort the sentences within a batch by its source sentence length in descending manner.

    Note 1: if you are an expert on time-series data, you may know a tensor of [sequence_length, batch_size, ...] is much faster than [batch_size, sequence_length, ...].
    However, for simple intuitive understanding, let's just use batch_first this time.

    Note 2: you can directly apply this function to torch.utils.data.dataloader.DataLoader with collate_fn argument.
    Read the test codes if you are interested in.

    Hint: torch.nn.utils.rnn.pad_sequence would be useful

    Arguments:
    batched_samples -- list of (source_sentence, target_sentence) pairs. This list should be converted to a batch

    Return:
    src_sentences -- batched source sentence
                        in shape (batch_size, max_src_sentence_length)
    trg_sentences -- batched target sentence
                        in shape (batch_size, max_trg_sentence_length)

    """
    PAD = Language.PAD_TOKEN_IDX
    batch_size = len(batched_samples)

    ### YOUR CODE HERE (~4 lines)
    src_sentences: torch.Tensor = None
    trg_sentences: torch.Tensor = None
    src = [sample[0] for sample in batched_samples]
    tgt = [sample[1] for sample in batched_samples]
    src_lengths = [len(sample) for sample in src]
    tgt_lengths = [len(sample) for sample in tgt]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    for i in range(len(batched_samples)):
        src[i] = np.asarray(src[i], dtype=np.int64)
        src[i] = np.pad(src[i], (0, max_src_len-len(src[i])), mode="constant", constant_values=PAD)
        tgt[i] = np.asarray(tgt[i], dtype=np.int64)
        tgt[i] = np.pad(tgt[i], (0, max_tgt_len-len(tgt[i])), mode="constant", constant_values=PAD)

    src_sentences = torch.LongTensor(src)
    trg_sentences = torch.LongTensor(tgt)

    _, sorted_indices = torch.sort(torch.LongTensor(src_lengths), dim=0, descending=True)
    src_sentences = src_sentences.index_select(0, sorted_indices)
    trg_sentences = trg_sentences.index_select(0, sorted_indices)

    ### END YOUR CODE
    assert src_sentences.shape[0] == batch_size and trg_sentences.shape[0] == batch_size
    assert src_sentences.dtype == torch.long and trg_sentences.dtype == torch.long
    return src_sentences, trg_sentences

#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class Language(Sequence[List[str]]):
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3

    def __init__(self, path: str) -> None:
        with open(path, mode='r', encoding='utf-8') as f:
            self._sentences: List[List[str]] = [line.split() for line in f]

        self.word2idx: Dict[str, int] = None
        self.idx2word: List[str] = None
    
    def build_vocab(self, min_freq: int=2) -> None:
        SPECIAL_TOKENS: List[str] = [Language.PAD_TOKEN, Language.UNK_TOKEN, Language.SOS_TOKEN, Language.EOS_TOKEN]
        self.idx2word = SPECIAL_TOKENS + [word for word, count in Counter(chain(*self._sentences)).items() if count >= min_freq]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
    
    def set_vocab(self, word2idx: Dict[str, int], idx2word: List[str]) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __getitem__(self, index: int) -> List[str]:
        return self._sentences[index]
    
    def __len__(self) -> int:
        return len(self._sentences)

class NmtDataset(Sequence[Tuple[List[int], List[int]]]):
    def __init__(self, src: Language, trg: Language, max_len: int=30) -> None:
        assert len(src) == len(trg)
        assert src.word2idx is not None and trg.word2idx is not None

        self._src = src
        self._trg = trg
        self._max_len = max_len

    def __getitem__(self, index: int) -> Tuple[List[str], List[str]]:
        return preprocess(self._src[index], self._trg[index], self._src.word2idx, self._trg.word2idx, self._max_len)

    def __len__(self) -> int:
        return len(self._src)

#############################################
# Testing functions below.                  #
#############################################

def preprocess_test():
    print ("======Preprocessing Test Case======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()

    # First test
    src_sentence, trg_sentence = preprocess(french[0], english[0], french.word2idx, english.word2idx, max_len=100)
    assert src_sentence == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13] and \
           trg_sentence == [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3], \
           "Your preprocessed result does not math expected result"
    print("The first test passed!")

    # Second test
    src_sentence, trg_sentence = preprocess(french[1], english[1], french.word2idx, english.word2idx, max_len=5)
    assert src_sentence == [14, 6, 15, 16, 17] and \
           trg_sentence == [2, 15, 16, 17, 3], \
           "Result of max_len do not match expected result" 
    print("The second test passed!")
    
    # Third test
    raw_src_sentence = 'parfois , un mot peut être hors du vocabulaire . like THIS .'.split()
    raw_trg_sentence = 'sometimes , a word could be out of vocabulary . comme ÇA .'.split()
    src_sentence, trg_sentence = preprocess(raw_src_sentence, raw_trg_sentence, french.word2idx, english.word2idx, max_len=12)
    assert src_sentence == [1, 79, 19, 2259, 1058, 774, 2279, 90, 1, 13, 1, 1] and \
           trg_sentence == [2, 1, 6, 21, 2140, 5517, 1015, 458, 74, 1, 14, 3], \
           "Words which are out of vocabulary (OOV) should be replaced by <UNK> tokens" 
    print("The third test passed!")

    print("All 3 tests passed!")

def buckting_test():
    print ("======Bucketing Test Case======")
    dataset_length = 50000
    min_len = 10
    max_len = 30
    batch_size = 64
    max_pad_len = 5

    sentence_length = [(random.randint(min_len, max_len), random.randint(min_len, max_len)) for _ in range(dataset_length)]
    batch_indices = bucketed_batch_indices(sentence_length, batch_size=batch_size, max_pad_len=max_pad_len)
    
    # the first test
    assert sorted(chain(*batch_indices)) == list(range(0, dataset_length)), \
        "Some of the samples are duplicated or missing."
    print("The first test passed!")
    
    # the second test
    assert sum(1 for batch in batch_indices if len(batch) < batch_size) < 30, \
        "You returned too many batches smaller than batch_size."
    print("The second test passed!")

    # the third test
    for batch in batch_indices:
        src_length, trg_length = zip(*list(sentence_length[idx] for idx in batch))
        assert max(src_length) - min(src_length) <= max_pad_len and max(trg_length) - min(trg_length) <= max_pad_len, \
            "There is a sentence which needs more <PAD> tokens than max_pad_len."
    print("The third test passed!")

    print("All 3 tests passed!")

def collate_fn_test():
    print ("======Collate Function Test Case======")
    
    # the first test
    batched_samples = [([1, 2, 3, 4], [1]), ([1, 2, 3], [1, 2]), ([1, 2], [1, 2, 3]), ([1], [1, 2, 3, 4])]
    src_sentences, trg_sentences = collate_fn(batched_samples)
    assert (src_sentences == torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 0], [1, 2, 0, 0], [1, 0, 0, 0]])).all() and \
           (trg_sentences == torch.Tensor([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]])).all(), \
           "Your collated batch does not math expected result."
    print("The first test passed!")

    # the second test
    batched_samples = [([1], [1, 2, 3, 4]), ([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2]), ([1, 2, 3, 4], [1])]
    src_sentences, trg_sentences = collate_fn(batched_samples)
    assert (src_sentences == torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 0], [1, 2, 0, 0], [1, 0, 0, 0]])).all() and \
           (trg_sentences == torch.Tensor([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]])).all(), \
           "Your collated batch should be sorted in descending manner by its source sentence length."
    print("The second test passed!")

    print("All 2 tests passed!")


def dataloader_test():
    print ("======Dataloader Test======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()
    dataset = NmtDataset(src=french, trg=english)

    batch_size = 4
    max_pad_len = 5
    sentence_length = list(map(lambda pair: (len(pair[0]), len(pair[1])), dataset))

    bucketed_batch_indices(sentence_length, batch_size=batch_size, max_pad_len=max_pad_len)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, collate_fn=collate_fn, num_workers=2, \
                                                        batch_sampler=bucketed_batch_indices(sentence_length, batch_size=batch_size, max_pad_len=max_pad_len))

    src_sentences, trg_sentences = next(iter(dataloader))
    print("Tensor for Source Sentences: \n", src_sentences)
    print("Tensor for Target Sentences: \n", trg_sentences)

    print("Dataloader test passed!")

if __name__ == "__main__":
    random.seed(1234)
    preprocess_test()
    buckting_test()
    collate_fn_test()
    dataloader_test()
