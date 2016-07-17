import nltk

from collections import defaultdict as deft

from nltk import (
    ngrams,
    Text
)

from nltk.collocations import *

from Tools import tokenizer


class Collocations:

    def __init__(
        self,
        data,
        min_bigram_freq=None,
        min_trigram_freq=None,
    ):
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        self.trigram_measures = nltk.collocations.TrigramAssocMeasures()
        self.min_bigram_freq = min_bigram_freq
        self.min_trigram_freq = min_trigram_freq
        if isinstance(data, basestring):
            self.corpus = Text([w.lower() for w in tokenizer(data)])
        elif isinstance(data, list):
            self.corpus = Text(data)
        self.colls = dict([])


    def extract(self):
        self.bigram_finder = BigramCollocationFinder.from_words(self.corpus)
        self.trigram_finder = TrigramCollocationFinder.from_words(self.corpus)
        if self.min_bigram_freq:
            self.bigram_finder.apply_freq_filter(self.min_bigram_freq)
        if self.min_trigram_freq:
            self.trigram_finder.apply_freq_filter(self.min_trigram_freq)


    def __iter__(self, bigram_n=5000, trigram_n=5000):
        for coll in self.bigram_finder.nbest(self.bigram_measures.pmi, bigram_n) + \
                    self.trigram_finder.nbest(self.bigram_measures.pmi, trigram_n):
            yield coll


    def compile(self):
        for coll in self:
            n = len(coll)
            if n not in self.colls.keys():
                self.colls[n] = deft(bool)
            self.colls[n][coll] = True


    def __getitem__(self, coll):
        n = len(coll)
        if n not in self.colls.keys():
            return False
        if self.colls[n][coll]:
            return True
        return False
    
    
    def __call__(self, words):
        grams = list(ngrams(words, 2)) + list(ngrams(words, 3))
        positives = [
            (i, len(gram), gram) for i, gram in enumerate(grams)
            if self.colls[len(gram)][gram]
        ]
        if not positives:
            return words
        positives.sort(key=lambda x: (x[1], len(words) - x[0]), reverse=True)
        matches, covered = self.__non_overlapping(positives)
        unigrams = [(i, w) for i, w in enumerate(words) if i not in covered]
        catted = sorted(matches + unigrams)
        return zip(*catted)[1]


    def __non_overlapping(self, positives):
        covered = set([])
        matches = []
        while positives:
            match = positives.pop(0)
            i, n, gram = match
            area = set(range(i, i + n))
            if covered.intersection(area):
                continue
            matches.append((i, gram))
        return matches, covered
            
