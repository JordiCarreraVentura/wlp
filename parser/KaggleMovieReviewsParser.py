

class KaggleMovieReviewsParser:
    
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        with open(self.source, 'rb') as rd:
            for line in rd:
                try:
                    yield self(line)
                except Exception:
                    pass
            
    def __call__(self, line):
        fields = line.strip().split('\t')
        tag = int(fields[1])
        text = str(fields[2])
        return tag, text
        
