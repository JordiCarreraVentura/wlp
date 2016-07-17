import re

from afinn import Afinn

from collections import defaultdict as deft

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from senticnet.senticnet import Senticnet


SENTIWORDNET = 'data/SentiWordNet_3.0.0_20130122.txt'
# SENTIWORDNET = '../data/SentiWordNet_3.0.0_20130122.txt'
RECORD = re.compile('[anv]\t')



class SentiWordNet:

    def __init__(self):
        self.sentiment = deft(float)
        with open(SENTIWORDNET, 'rb') as rd:
            for l in rd:
                if not RECORD.match(l):
                    continue
                row = l.decode('utf-8').strip().split('\t')
                w = row[4].split('#')[0]
                pos = float(row[2])
                neg = float(row[3])
                self.sentiment[w] = pos - neg
    
    def __call__(self, word):
        return self.sentiment[word]
    


class Vader:

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def __call__(self, text):
        scores = self.analyzer.polarity_scores(text)
        return scores['pos'] - scores['neg']

    


class Sentiment:
    
    def __init__(self, dictionary):
        self.name = dictionary
        self.sentiment = None
        self.__configure()
    
    def __str__(self):
        return self.name
    
    def __configure(self):
        if self.name == 'afinn':
            self.sentiment = Afinn().score
        elif self.name == 'sentic':
            self.sentiment = Senticnet().polarity
        elif self.name == 'sentiwordnet':
            self.sentiment = SentiWordNet()
        elif self.name == 'vader':
            self.sentiment = Vader()
    
    def word_sentiment(self, word):
        return self.__rescale(self.sentiment(word))
    
    def __rescale(self, score):
        score = score * 100
        if self.name == 'afinn':
            min, max = -300, 400
        elif self.name == 'sentic':
            min, max = -100, 100
        elif self.name == 'sentiwordnet':
            min, max = 0, 100
        elif self.name == 'vader':
            min, max = 0, 100
        r = 0
        ranged = range(min, max)
        for i, x in enumerate(ranged):
            if score >= x:
                r = (i + 1) / float(len(ranged))
            else:
                break
        return r


    def text_sentiment(self, text):
        if self.name == 'afinn' or self.name == 'vader':
            return self.__rescale(self.sentiment(text))
        elif self.name == 'sentiwordnet':
            sentims = []
            for w in text.lower().split():
                try:
                    s = self.__rescale(self.sentiment(w))
                except Exception:
                    continue
                sentims.append(s)
            if not sentims:
                return 0.5
            else:
                return sum(sentims) / len(sentims)
        elif self.name == 'sentic':
            raise Exception('Unsupported library for this method')
        




    
