import codecs
import os
import re

from collections import (
    Counter
)

from Tools import (
    decode,
    encode,
    strip_punct
#     splitter,
#     tokenizer
)


PATH_TWENTY_NEWSGROUPS = 'data/20_newsgroups/'
PATH_TWENTY_NEWSGROUPS = '../data/20_newsgroups/'

SEPARATOR = re.compile('.{1,30}:')



class TwentyNewsgroupsCorpusWrapper:

    def __init__(self):
        self.root = PATH_TWENTY_NEWSGROUPS
        self.documents = []
        self.paths = []
        self.tags = []
        self.i_by_tag = dict([])
        self.tagdist = Counter()
        self.__load()

    
    def __load(self):
        for category in os.listdir(self.root):
            category_folder = '%s%s' % (self.root, category)
            if not os.path.isdir(category_folder):
                continue
            for document_path in os.listdir(category_folder):
                document_path = '%s/%s' % (category_folder, document_path)
                text = self.__read(document_path)

                try:
                    codecs.utf_8_decode(text)
                except Exception:
                    continue

#                 print text[:1500]
#                 print category
#                 print
                if not text:
                    continue
                self.i_by_tag[document_path] = len(self.paths)
                self.paths.append(document_path)
                self.tags.append(category)
                self.documents.append(text)
                self.tagdist[category] += 1
                    
                    
    def __read(self, document_path):
        with open(document_path, 'rb') as rd:
            lines = []
            for line in rd:
                try:
                    line = encode(line)
                    if SEPARATOR.match(line) or line.startswith('In article <') or \
                    line.startswith('>In article <'):
                        continue
                    lines.append(line)
                except Exception:
                    pass
            return ''.join(lines)


    def fileids(self):
        for path in self.paths:
            yield path
    

    def words(self, path=None):

        if not path:
            space = self.i_by_tag.values()
        else:
            space = [self.i_by_tag[path]]

        _words = []
        for i in space:
            text = self.documents[i]
            for w in text.split():
                token = strip_punct(w).lower()
                if token:
                    _words.append(token)

        return _words


    def categories(self, path=None):

        if not path:
            space = self.i_by_tag.values()
        else:
            space = [self.i_by_tag[path]]

        categories = []        
        for i in space:
            categories.append(self.tags[i])

        return categories
