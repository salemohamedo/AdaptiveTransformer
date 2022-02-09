import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.counts = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.counts:
            self.counts[word] = 1
        else:
            self.counts[word] += 1

    def gen_idx(self):
        counts_sorted = sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)
        for idx, kv in enumerate(counts_sorted):
            self.idx2word.append(kv[0])
            self.word2idx[kv[0]] = idx

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.compute_counts(os.path.join(path, 'train.txt'))
        self.compute_counts(os.path.join(path, 'valid.txt'))
        self.compute_counts(os.path.join(path, 'test.txt'))
        self.dictionary.gen_idx()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def compute_counts(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids