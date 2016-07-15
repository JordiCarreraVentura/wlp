import nltk

from nltk.corpus import reuters



from collections import (
    Counter,
    defaultdict as deft
)

#    Home-made search engine using BoWs -the "hack way"

#    First, we build the index of documents by their words
#    and compute their BoWs at the same time:

documents_by_word = deft(list)    # We will use an inverted index to speed up lookup

bows = dict([])
for i in reuters.fileids():
    bow = dict([])
    for word, weight in prob_dist.most_tfidf(i):
        bow[word] = weight
        documents_by_word[word].append(i)
    bows[i] = bow


#    After that, we iterate over the documents again
#    and retrieve the most similar documents:

#    A function to get all BoWs sharing some word
#    with the input BoW:
def documents_with_some_word_in_common(bow, top=None):
    if not None:
        docs = set([])
        for w in bow.keys():
            docs.update(documents_by_word[w])
        return docs
    else:
        docs = Counter()
        for w in bow.keys():
            docs.update(documents_by_word[w])
        return set([doc for doc, f in docs.most_common(top)])


#    A function to remind us of the top-5 words
#    by TFIDF in a particular BoW:
def top5(bow):
    return sorted(
        bow.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]


#    A function to calculate the similarity
#    between two BoWs, for now simply as the
#    ratio between the TFIDF of all words
#    shared between the two, and the total
#    TFIDF of all their words, averaged, and
#    minus the average of difference between
#    the two:
def similarity(bow, i):
    sim = 0.0
    _sim = 0.0
    _bow = bows[i]
    shared_words = set(bow.keys()).intersection(set(_bow.keys()))
    total_tfidf = sum(bow.values())
    _total_tfidf = sum(_bow.values())
    matches = []
    for w in shared_words:
        sim += bow[w] / total_tfidf
        _sim += _bow[w] / _total_tfidf
        matches.append((w, bow[w]))
    return (((sim + _sim) / 2) - (abs(sim - _sim) / 2),
            sorted(matches, key=lambda x: x[1], reverse=True)[:5])


#    A function to see the first 20 words of a
#    particular Reuters docuemnt:
def txt(i):
    return ' '.join(list(reuters.words(i))[:100])





import numpy, scipy

from sklearn.metrics.pairwise import cosine_similarity as cos

A = numpy.array
dimensions = {
    w: i for i, w in enumerate(set(reuters.words()))
}
words = {
    i: w for w, i in dimensions.items()
}



#    Function to transform a our current BoW objects
#    (Python dictionaries) into vectors:
def bow2vec(bow):
    vector = [0.0 for i in range(len(dimensions.keys()))]
    for w, weight in bow.items():
        word_index = dimensions[w]
        vector[word_index] = weight
    return A(vector).reshape(1, -1)

                               
#    Home-made search engine using BoWs -the cool way
for i, bow in bows.items()[:5]:
    print i, len(candidate_bows)
    candidate_bows = documents_with_some_word_in_common(bow, top=10)
    sim_bows = sorted(
        [(candidate, cos(bow2vec(bow), bow2vec(bows[candidate])), txt(candidate))
         for candidate in list(candidate_bows)],
        key=lambda x: x[1],
        reverse=True
    )
    print i
    print txt(i)
    print reuters.categories(i)
    print '---'
    for candidate, sim, text in sim_bows[1:4]:
        print '  ', candidate
        print '  ', reuters.categories(candidate)
        print '  ', sim
        print '  ', text
        print '  ', '----'
    print '====='
    print
