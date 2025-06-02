from physiosignal.utils import _checking, _sort

class Annotations:

    def __init__(self, onset, duration, description, ch_names):
        self. onset, self.duration, self.description, self.ch_names = _checking(onset, duration, description, ch_names)
        
        self._sort()

    def add(self):
        pass

    def remove(self):
        pass

    def get_annotations(self):
        pass

    def find(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
    