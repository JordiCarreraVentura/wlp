
import math
import nltk
import numpy as np

from collections import (
    Counter,
    defaultdict as deft
)

from nltk.corpus import stopwords

from TermIndex import TermIndex

from Tools import (
    ngrams
)

#    Advanced features
from Collocations import Collocations

from Lemmatizer import PatternLemmatizer

from TextStreamer import TextStreamer

from WordNet import WordNet




STOPWORDS = deft(bool)
for w in stopwords.words('english'):
    STOPWORDS[w] = True


class FeatureExtractor:
    
    def __init__(
        self,
        name,
        lowercase=True,
        lemmatize=False,
        rm_stopwords=False,
        rm_numbers=False,
        rm_punct=False,
        weight='tfidf',
        ngrams=[],
        collocations=False,
        pos=False,
        sentiment=None,
        synsets=None,
        embeddings='',
        min_df=3,
        max_df=None
    ):
        self.name = name
        self.lowercase = lowercase
        self.lemmatizer = lemmatize
        self.rm_stopwords = rm_stopwords
        self.rm_numbers = rm_numbers
        self.rm_punct = rm_punct
        self._weight = weight
        self.ngrams = ngrams
        self.collocations = collocations
        self.pos = pos
        self.sentiment = sentiment
        self._embeddings = embeddings
        self.min_df = min_df
        self.max_df = max_df
        self.synsets = synsets
        #
        self.index = TermIndex(u'')
        self.generalized = deft(str)
        self.tf = deft(Counter)
        self.df = Counter()
        self.mass = 0.0
        self.n = 0
        self.documents = []
        self.embeddings = None
        self.colls = None
        self.__configure()
    
    def __str__(self):
        return self.name
    
    def __configure(self):
        if self.lemmatizer:
            self.lemmatizer = PatternLemmatizer()
        if self.synsets:
            self.wordnet = WordNet()
    
    def clear(self):
        self.index = TermIndex(u'')
        self.tf = deft(Counter)
        self.df = Counter()
        self.mass = 0.0
        self.n = 0
        self.documents = []
        self.index = TermIndex(u'')
        self.generalized = deft(str)
        self.tf = deft(Counter)
        self.df = Counter()
        self.mass = 0.0
        self.n = 0
        self.documents = []
        self.embeddings = None
        self.colls = None
    
    def update(self, tokens):
        preproced = []
        tokens = self.lemmatize(tokens)
        for token in tokens:
            token = self.__preproc(token)
            if not token:
                continue
            i = self.index(token)
            self.tf[self.n][i] += 1
            self.df[i] += 1
            self.mass += 1
            preproced.append(i)
        self.n += 1
        self.documents.append(preproced)
    
    def __preproc(self, token):
        if self.rm_punct and not (token.isalpha() or token.isdigit()):
            return None
        if self.rm_numbers and not token.isalpha():
            return None
        if self.rm_stopwords and STOPWORDS[token.lower()]:
            return None
        if self.lowercase:
            token = token.lower()
        token = self.generalize(token)
        return token
    
    def compute(self):
        self.__fit_generalize()
        self.__extract_collocations()
    
    def dump(self):
        _dump = []
        for doc in self.documents:
            _dump += doc
        return _dump
    
    def __extract_collocations(self):
        self.colls = Collocations(
            self.dump(),
            min_bigram_freq=5,
            min_trigram_freq=3
        )
        self.colls.extract()
        self.colls.compile()
    
    def __fit_generalize(self):
        if not self.synsets:
            return
        for w in self.tf.keys():
            generalized = self.wordnet.generalize(self.index[w])
            i = self.index(generalized)
            self.generalized[w] = i
    
    def generalize(self, token):
        if not self.synsets:
            return token
        generalized = self.generalized[token]
        if not generalized:
            return token
        return generalized
    
    def weight(self, i, w):
        if self._weight == 'tfidf':
            return self.tfidf(i, w)
        else:
            exit('Metric not supported')
            
    def tfidf(self, i, w):
        idf = math.log(self.n / float(self.df[w]), 10)
        tf = self.tf[i][w]
        _tfidf = tf * idf
        if _tfidf < 0:
            return 0.0
        else:
            return _tfidf
    
    def __getitem__(self, i):
        doc = self.documents[i]
#         n_dim = self.index.n
#         vector = np.zeros(n_dim, dtype=float)
#         for w in doc:
#             w = self.to_embedding(w)
#             if self.__out_of_bounds(w):
#                 continue
#             weight = self.weight(i, w)
#             vector[w] = weight
        vector = []
        coll = self.collocate(doc)
        for w in doc:
            oob = self.__out_of_bounds(w)
            if oob > 0:
                continue
            elif oob < 0:
                vector.append(self.coll2word(w))
            else:
                vector.append(self.index[w])
        vector = self.__to_grams(vector)
        return ' '.join(vector)
    
    def collocate(self, doc):
        if self.collocations:
            return colls(doc)
        else:
            return doc
    
    def coll2word(self, w):
        return '_'.join([self.index[token] for token in w])
    
    def __to_grams(self, vector):
        if not self.ngrams:
            return vector
        else:
            grams = []
            for n, skip in self.ngrams:
                for g in ngrams(vector, n):
                    if not skip:
                        grams.append('_'.join(g))
                    else:
                        grams.append('_'.join((g[0], '*', g[-1])))
            return grams
    
    def __out_of_bounds(self, w):
        if isinstance(w, tuple):
            return -1
        if self.min_df and self.df[w] < self.min_df:
            return 1
        elif self.max_df and self.df[w] > self.max_df:
            return 1
        else:
            return 0
    
    def lemmatize(self, tokens):
        if not self.lemmatizer:
            return tokens
        try:
            return  self.lemmatizer(' '.join(tokens))
        except Exception:
            return tokens
