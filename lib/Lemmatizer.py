
#    Using spacy
import spacy

import pattern

from pattern.en import (
    parse
)


class PatternLemmatizer:
    
    def __init__(self):
        return

    def __call__(self, text):
        return [
            morph.split('/')[-1]
            for morph in str(parse(text, relations=True, lemmata=True)).split()
        ]


class SpaCyLemmatizer:
    
    def __init__(self):
        self.en = spacy.load('en')
    
    def __call__(self, text):
        doc = self.en(text)
        return [doc.vocab[token.lemma].orth_ for token in doc]


if __name__ == '__main__':
    l = PatternLemmatizer()
    print l(u'Escalation unto death The nuclear war is already being fought , except that the')
    print l(u'For a neutral Germany Soviets said to fear resurgence of German militarism to th')
