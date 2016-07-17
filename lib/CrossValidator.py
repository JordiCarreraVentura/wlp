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
                  self.extractor, \
                  self.classifier, \
                  fold, len(train), \
                  len(test), \
                  acc, \
                  runtime
        print self.dataset, self.extractor, self.classifier, \
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

    NGRAMS = [
        (1, False),
        (2, False),
        (3, True),
        (4, True)
    ]

    #    We will re-use the SimpleCorpusReader class we saw when introducing
    #    our datasets, and extend it with .categories(), .words(), etc. meth-
    #    ods like those in NLTK's CorpusReaders so that we can use them with-
    #    in the same training and testing workflow:
    from TwentyNewsgroupsCorpusWrapper import TwentyNewsgroupsCorpusWrapper as twenty_newsgroups
    from SentimentCorpusReaderWrapper import SentimentCorpusReaderWrapper as sentiment_corpus

    datasets = [brown, reuters, twenty_newsgroups()]
#     datasets = [brown, reuters]
#     datasets = [sentiment_corpus()]
#     datasets = [brown]
#     datasets = [reuters]
#     datasets = [twenty_newsgroups()]


    #    We have also implemented a wrapper class 'Classifier' that gives us
    #    easy and consistent access to scikit-learn's classification algori-
    #    thms:
    from Classifier import Classifier

    #    Although NLTK's datasets usually come with pre-defined train and test
    #    splits of the data, in our experiments we will ignore that distinct-
    #    ion and we will be performing cross-validation. When cross-validating,
    #    the ratio between training and testing data is observed (for instance,
    #    8 training instances for every 2 test instances) but combining diffe-
    #    rent parts of the corpus: in the 1st cross-validation fold, the first
    #    20% of the dataset is used as training and the remaining 80% for tes-
    #    ting; in the 2nd cross-validation fold, testing is performed on the
    #    21-40% of the dataset, and training on the remaining 1-20% + 41-100%,
    #    and so on. Cross-validation is preferable as an evaluation methodolo-
    #    gy because it is far more robust. Results that generalize well to all
    #    subsets of our dataset will probably perform well on new data.
    #
    #    For convenience, we have also implemented a cross-validation wrapper
    #    to take care of the experimental design for us:
    from CrossValidator import CrossValidator

    from FeatureExtractor import FeatureExtractor

    lr = Classifier(
        classifier='lr'
    )

    nb = Classifier(
        classifier='mnb'
    )

    clfs = [nb, lr]
#     clfs = [nb]
#     clfs = [lr]

    xtor1 = FeatureExtractor(
        'off'
    )
    
    xtor2 = FeatureExtractor(
        'coll',
        collocations=True
    )

    xtor3 = FeatureExtractor(
        'rm',
        rm_numbers=True,
        rm_punct=True,
        rm_stopwords=True,
    )

    xtor4 = FeatureExtractor(
        'lemma',
        lemmatize=True
    )
    
    xtor5 = FeatureExtractor(
        'battery',
        rm_numbers=True,
        rm_punct=True,
        rm_stopwords=True,
        collocations=True,
        lemmatize=True
    )

    xtor0 = FeatureExtractor(
        'ngrams',
        ngrams=NGRAMS,
        rm_punct=True,
    )
    
#     xtor0 = None
    xtor01 = FeatureExtractor(
        'ngrams-punct',
        ngrams=[(1, False), (2, False)],
        rm_punct=True,
    )
    xtor02 = FeatureExtractor(
        'off'
#         'ngrams-punct',
#         ngrams=[(1, False), (2, False)],
#         rm_punct=True,
    )
    
    xtors = [xtor1, xtor2, xtor3, xtor4, xtor5]
    xtors = [xtor02, xtor01]
    xtors = [xtor0]

    #    Experimental workflow:
    for clf in clfs:
        for dataset in datasets:
            for xtor in xtors:
                c = CrossValidator(clf, dataset, train_r=0.9, extractor=xtor)
#                 c = CrossValidator(clf, dataset, train_r=0.9)
                c.run()
