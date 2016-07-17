import nltk
import random
import time

from collections import (
    Counter,
    defaultdict as deft
)

from nltk.corpus import brown, reuters

from random import shuffle as randomize

from Tools import (
    average,
    median
)

from tqdm import tqdm


class CrossValidator:
    
    def __init__(
        self,
        clf,
        dataset,
        extractor=None,
        train_r=0.8,
        max_n=None
    ):
        self.classifier = clf
        self.dataset = dataset
        self.extractor = extractor
        self.subsamples = deft(list)
        self.tags = deft(list)
        self.documents = []
        self.nfolds = 0
        self.folds = deft(list)
        self.r = train_r
        self.max_n = max_n
        self.vectorized = False
        self.__configure_iterations()
        self.__build()
        self.shuffle()
        self.__folds()
        if extractor:
            self.vectorized = True
    
    def __configure_iterations(self):
        diff = 1 - self.r
        self.nfolds = int(1 / diff)
    
    def shuffle(self):
        for tag, ii in self.subsamples.items():
            randomize(ii)
    
    def __folds(self):
        for tag, ii in self.subsamples.items():
            n = int(round(len(ii) * (1 - self.r)))
#             print tag, len(ii), n
            curr = []
            for i in ii:
                curr.append(i)
                if len(curr) == n:
                    self.folds[tag].append(curr)
                    curr = []
            if curr and len(curr) < n * 0.5:
                self.folds[tag][-1] += curr
            elif curr:
                self.folds[tag].append(curr)
    
    def __build(self):
        d = self.dataset
        if self.extractor:
            self.extractor.clear()
        for i, _id in tqdm(enumerate(d.fileids())):
            tags = d.categories(_id)
            self.documents.append(self.__preproc(d.words(_id)))
            if not isinstance(tags, list):
                tags = [tags]
            for tag in [tags[0]]:                   #   If a document has several tags,
                self.subsamples[tag].append(i)      #   we only add it for one.
        self.__fit()
    
    def __preproc(self, words):
        if not self.extractor:
            return ' '.join(words)
        else:
            self.extractor.update(words)
            return None
    
    def __fit(self):
        if not self.extractor:
            return
        self.extractor.compute()
    
    def run(self):
        globals = deft(list)
        for fold in range(self.nfolds):
            start = time.time()
            train, test = self.__collect(fold)
#             self.classifier.train(train, vectorized=self.vectorized)
#             guesses = self.classifier.test(test, vectorized=self.vectorized)
            self.classifier.train(train, vectorized=False)
            guesses = self.classifier.test(test, vectorized=False)
            runtime = round(time.time() - start, 2)
            acc = self.accuracy(test, guesses, topn=1)
            globals['accuracy'].append(acc)
            globals['runtime'].append(runtime)
            globals['train'].append(len(train))
            globals['test'].append(len(test))
            print self.dataset, \
                  self.classifier, \
                  self.extractor, \
                  fold, len(train), \
                  len(test), \
                  acc, \
                  runtime
        print self.dataset, self.classifier, self.extractor, \
              'run', sum(globals['train']), sum(globals['test']), \
              average(globals['accuracy']), average(globals['runtime'])
        print '--------------'
              
                
    
    def accuracy(self, test, guesses, topn=1):
        hits = 0
        for i, guess in enumerate(guesses):
            best = guess[0][1]
            tag = test[i][2]
            #print i, best, '/%s' % tag, test[i]
            if best == tag:
                hits += 1
        return round(hits / float(len(test)), 2)
    

    def __collect(self, fold):
        train, test = [], []
        for tag, _folds in self.folds.items():
            tag_train, tag_test = [], []
            if fold >= len(_folds):
                continue
            tag_test = _folds[fold]
            for __folds in _folds[:fold] + _folds[fold + 1:]:
                tag_train += __folds

            if self.max_n:
                tag_train = tag_train[:self.max_n]
                tag_test = tag_test[:int(self.max_n * self.r)]
            train += [(i, self[i], tag) for i in tag_train]
            test += [(i, self[i], tag) for i in tag_test]

        return train, test
    
    def __getitem__(self, i):
        if not self.extractor:
            return self.documents[i]
        else:
            return self.extractor[i]
        


if __name__ == '__main__':
    import nltk
    from nltk.corpus import reuters, brown

    from Classifier import Classifier
    from CrossValidator import CrossValidator
    from FeatureExtractor import FeatureExtractor

    from Sentiment import Sentiment
    from SentimentCorpusReaderWrapper import SentimentCorpusReaderWrapper as sentiment_corpus

    NGRAMS = [
        (1, False),
        (2, False),
        (3, True),
    ]

    datasets = [sentiment_corpus()]
    
    sentiment = Sentiment('sentiwordnet')

    lr = Classifier(
        classifier='lr',
#         min_df=5,
    )

    nb = Classifier(
        classifier='mnb',
#         min_df=5,
    )

    clfs = [nb, lr]
#     clfs = [nb]
#     clfs = [lr]

    xtor_none = None

    xtor_off = FeatureExtractor(
        'off'
    )

    xtor_clean = FeatureExtractor(
        'clean',
        rm_numbers=True,
        rm_punct=True,
    )

    xtor_grams = FeatureExtractor(
        'ngrams',
        rm_numbers=True,
        rm_punct=True,
        ngrams=NGRAMS,
    )

    xtor_sent = FeatureExtractor(
        'sentiment',
        rm_numbers=True,
        rm_punct=True,
        sentiment=sentiment
    )

    xtor_batt = FeatureExtractor(
        'battery',
        rm_numbers=True,
        sentiment=sentiment,
        ngrams=NGRAMS,
    )


    xtors = [xtor_none, xtor_off, xtor_clean, xtor_sent, xtor_grams, xtor_batt]

    #    Experimental workflow:
    for clf in clfs:
        for dataset in datasets:
            for xtor in xtors:
                c = CrossValidator(clf, dataset, train_r=0.8, extractor=xtor)
                c.run()
