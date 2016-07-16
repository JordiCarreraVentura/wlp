import csv
import nltk
import re

from collections import (
    Counter,
    defaultdict as deft
)

from copy import deepcopy as cp

from nltk import (
    ngrams,
    sent_tokenize as splitter,
    wordpunct_tokenize as tokenizer
)

from nltk.corpus import stopwords

from nltk.probability import FreqDist


NUM = re.compile('[0-9]')
NON_ALPHA = re.compile('[^a-z]', re.IGNORECASE)

STOPWORDS = deft(bool)
for w in stopwords.words('english'):
    STOPWORDS[w] = True


def decode(string):
    try:
        return string.decode('utf-8')
    except Exception:
        return string


def encode(string):
    try:
        return string.encode('utf-8')
    except Exception:
        return string


def remove_nonwords(items):
    rmd = []
    for item in items:
        if not NON_ALPHA.search(item):
            rmd.append(item)
    return rmd


def to_csv(data, path):
    with open(path, 'wb') as wrt:
        wrtr = csv.writer(wrt, quoting=csv.QUOTE_MINIMAL)
        for row in data:
            wrtr.writerow(tuple([encode(field) for field in row]))


def from_csv(path, delimiter=None):
    rows = []
    d = decode
    with open(path, 'rb') as rd:
        if delimiter:
            rdr = csv.reader(rd, delimiter=delimiter)
        else:
            rdr = csv.reader(rd)
        for row in rdr:
            yield row


def strip_punct(string):
    start, end = 0, len(string)

    for i, char in enumerate(string):
        start = i
        if not NON_ALPHA.match(char):
            break
    if start == end - 1:
        return ''

    while True:
        end -= 1
        if end < 0:
            return ''

        if not NON_ALPHA.match(string[end]):
            break

    return string[start:end + 1]


def median(values):
    vv = cp(values)
    vv.sort()
    return vv[int(len(vv) / 2)]


def average(values):
    return sum(values) / len(values)
