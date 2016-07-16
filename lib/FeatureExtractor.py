
import math
import nltk
import numpy as np

from collections import (
    Counter,
    defaultdict as deft
)

from nltk.corpus import stopwords

from TermIndex import TermIndex

#    Advanced features
from Lemmatizer import PatternLemmatizer

from WordNet import WordNet

#from Collocations import Collocations

#from WordEmbeddings import WordEmbeddings




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
        self.synsets = synsets
        self.generalized = deft(str)
        self.embeddings = embeddings
        self.min_df = min_df
        self.max_df = max_df
        #
        self.index = TermIndex(u'')
        self.tf = deft(Counter)
        self.df = Counter()
        self.mass = 0.0
        self.n = 0
        self.documents = []
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
        n_dim = self.index.n
        doc = self.documents[i]
        vector = np.zeros(n_dim, dtype=float)
        for w in doc:
            if self.__out_of_bounds(w):
                continue
            weight = self.weight(i, w)
            vector[w] = weight
        return vector
    
    def __out_of_bounds(self, w):
        if self.min_df and self.df[w] < self.min_df:
            return True
        elif self.max_df and self.df[w] > self.max_df:
            return True
        else:
            return False
    
    def lemmatize(self, tokens):
        if not self.lemmatizer:
            return tokens
        lemmatized = self.lemmatizer(' '.join(tokens))
        return lemmatized
