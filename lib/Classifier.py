from __future__ import division
import numpy
import re
import sklearn
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Classifier:

    def __init__(
        self,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        classifier='lr'
    ):
        self.name = classifier
        self.vectorizer = TfidfVectorizer(
            input='content',
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            use_idf=True
        )
        if classifier == 'lr':
            self.classifier = LogisticRegression(
                class_weight='balanced',
#                 solver='liblinear',
#                 solver='newton-cg',
                solver='lbfgs', 		# much better than newton-cg
                multi_class='ovr', 	    # w/ lbfgs, much better than multinomial
#                 multi_class='multinomial',
                max_iter=100
            )
        elif classifier == 'nb':
            self.classifier = GaussianNB()
        elif classifier == 'mnb':
            self.classifier = MultinomialNB()
        elif classifier == 'svm':
            self.classifier = SVC(
                probability=True,
                kernel='poly',		#  'linear', 'poly', 'rbf', 'sigmoid'
                class_weight='balanced',
                max_iter=100,
                #decision_function_shape='ovo'	# 'ovo', 'ovr', or None
            )

    def __str__(self):
        return self.name

    def train(self, tuples, vectorized=False):
        start = time.time()
        if tuples:
            text_ids, examples, labels = zip(*tuples)
#             matrix = self.vectorizer.fit_transform(tqdm(examples))
            matrix = self.__vectorize(vectorized, examples, 'train')          
            if isinstance(self.classifier, GaussianNB):
                self.classifier.fit(
                    matrix.toarray(), labels
                )
            else:
                self.classifier.fit(
                    matrix, labels
                )
#             print 'took %.2f seconds to train' % (time.time() - start)
            return True
        return False
    
    def __vectorize(self, vectorized, examples, stage):
        if not vectorized and stage == 'train':
            return self.vectorizer.fit_transform(examples)
        elif not vectorized and stage == 'test':
            return self.vectorizer.transform(examples)
        else:
            return examples

    def test(self, tuples, vectorized=False, proba=True):
        start = time.time()
        text_ids, examples, labels = zip(*tuples)
        if examples:
            self.vectors = self.__vectorize(vectorized, examples, 'test')
            self.labels = labels
        else:
            exit('Missing vectors')
        if isinstance(self.classifier, GaussianNB):
            return self.classifier.predict(self.vectors.toarray())
        else:
            predictions = self.classifier.predict_proba(self.vectors)
            predictions = [ sorted(zip(prediction, self.classifier.classes_), 
                                  reverse=True)
                           for prediction in predictions]
            return predictions


# def f_score(accuracy, recall):
#     return 2 * ((accuracy * recall) / (accuracy + recall))
