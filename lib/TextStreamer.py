# from lib.Tools import (
from Tools import (
    decode as d,           # Auxiliary method to decode UTF-8 (when reading from a file)
    encode as e            # Auxiliary method to encode UTF-8 (when writing to a file or stdout)
)

class TextStreamer:
    
    def __init__(self, source, parser=None):
        self.source = source
        if parser:
            self.parser = parser(self.source)
        else:
            self.parser = None

    def __iter__(self):
        if not self.parser:
            with open(self.source, 'rb') as rd:
                for line in rd:
                    if not line.strip():
                        continue
                    yield d(line)
        else:
            for parsed in self.parser:
                yield d(parsed)
