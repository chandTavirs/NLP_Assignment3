import os
from io import open
from collections import Counter

from torch.autograd import Variable

from transformer_model import subsequent_mask


class Dictionary(object):
    def __init__(self):
        self.all_words = []
        self.word2idx = {}
        self.idx2word = []
        self.exclude = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def gather_words(self, word):
        self.all_words.append(word)

    def exclude_rare_words(self, min_count):
        word_count = Counter(self.all_words)
        for w, c in word_count.items():
            if c < min_count:
                self.exclude.append(w)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, min_count=0):
        self.dictionary = Dictionary()
        self.min_count = min_count
        # self.train, self.train_sent = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        # self.valid, self.valid_sent = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        # self.test, self.test_sent = self.tokenize(os.path.join(path, 'wiki.test.tokens'))
        self.train_sent = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid_sent = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test_sent = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        if self.min_count > 0:
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.gather_words(word)

            self.dictionary.exclude_rare_words(self.min_count)

        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.dictionary.exclude:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    if word not in self.dictionary.exclude:
                        ids.append(self.dictionary.word2idx[word])
                #tokenized_line = torch.tensor(ids).type(torch.int64)

                idss.append(ids)

            #ids = torch.stack(idss)

        return idss

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, 0:]
            self.trg_y = trg[:, 0:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask