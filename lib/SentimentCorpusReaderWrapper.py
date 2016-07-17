import codecs
import os
import random
import re

from collections import (
    Counter
)

from random import shuffle as randomize

from Tools import (
    decode,
    encode,
    strip_punct
#     splitter,
#     tokenizer
)


PATH_SENTIMENT = 'data/Microsoft.tsv'
PATH_SENTIMENT = '../data/Microsoft.tsv'

# SEPARATOR = re.compile('.{1,30}:')



class SentimentCorpusReaderWrapper:

    def __init__(self):
        self.root = PATH_SENTIMENT
        self.documents = []
        self.paths = []
        self.tags = []
        self.tagdist = Counter()
        self.__load()

    
    def __load(self):
        with open(self.root) as rd:
            for l in rd:
                try:
                    parts = l.decode('utf-8').strip().split('\t')
                except Exception:
                    pass
                tag = parts[0]
                text = '\t'.join(parts[1:])
                i = len(self.documents)
                self.paths.append(i)
                self.documents.append(text)
                self.tags.append(tag)
        self.__subsample(25000)
    
    
    def __subsample(self, n):
        all = zip(self.documents, self.tags)
        randomize(all)
        pos, neg = [], []
        for doc, tag in all:
            if tag == 'positive' and len(pos) < n:
                pos.append((doc, tag))
            elif tag == 'negative' and len(neg) < n:
                neg.append((doc, tag))
        sample = pos + neg
        self.documents, self.tags = zip(*sample)
        self.paths = range(n * 2)


    def fileids(self):
        for i in range(len(self.documents)):
            yield i
    

    def words(self, path=None):

        if not path:
            space = self.paths
        else:
            space = [path]

        _words = []
        for i in space:
            text = self.documents[i]
            for w in text.split():
                token = strip_punct(w).lower()
                if token:
                    _words.append(token)

        return _words


    def categories(self, path=None):

        if not path:
            space = self.paths
        else:
            space = [path]

        categories = []
        for i in space:
            categories.append(self.tags[i])

        return categories
