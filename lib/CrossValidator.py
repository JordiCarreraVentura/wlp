import nltk
import random
import time

from nltk.corpus import brown, reuters

from random import shuffle as randomize

from collections import (
    Counter,
    defaultdict as deft
)


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
        self.__configure_iterations()
        self.__build()
        self.shuffle()
        self.__folds()
    
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
        for i, _id in enumerate(d.fileids()):
            tags = d.categories(_id)
            self.documents.append(d.words(_id))
            if not isinstance(tags, list):
                tags = [tags]
            for tag in [tags[0]]:                   #   If a document has several tags,
                self.subsamples[tag].append(i)      #   we only add it for one.
    
    def run(self):
        for fold in range(self.nfolds):
            start = time.time()
            train, test = self.__collect(fold)
            self.classifier.train(train)
            guesses = self.classifier.test(test)
            print self.dataset, \
                  self.classifier.name, \
                  fold, len(train), \
                  len(test), \
                  self.accuracy(test, guesses, topn=1), \
                  round(time.time() - start, 2)
    
    def accuracy(self, test, guesses, topn=1):
        hits = 0
        for i, guess in enumerate(guesses):
            best = guess[0][1]
            tag = test[i][2]
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
            train += [(i, ' '.join(self.documents[i]), tag) for i in tag_train]
            test += [(i, ' '.join(self.documents[i]), tag) for i in tag_test]
        
        all = train + test
        train = all[:int(len(all) * self.r)]
        test = all[int(len(all) * self.r):]
        return train, test


if __name__ == '__main__':
    import nltk
    from nltk.corpus import reuters, brown

    #    We will re-use the SimpleCorpusReader class we saw when introducing
    #    our datasets, and extend it with .categories(), .words(), etc. meth-
    #    ods like those in NLTK's CorpusReaders so that we can use them with-
    #    in the same training and testing workflow:
    from TwentyNewsgroupsCorpusWrapper import TwentyNewsgroupsCorpusWrapper as twenty_newsgroups

    datasets = [brown, reuters, twenty_newsgroups()]
#     datasets = [brown, reuters]
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

    lr = Classifier(
        classifier='lr',
        min_df=3,
        ngram_range=(1, 3),
    )

    nb = Classifier(
        classifier='mnb',
        min_df=3,
        ngram_range=(1, 3),
    )

    clfs = [nb, lr]
#     clfs = [nb]
#     clfs = [lr]


    #    Experimental workflow:
    for clf in clfs:
        for dataset in datasets:
            print dataset
            c = CrossValidator(clf, dataset, train_r=0.9)
            c.run()
