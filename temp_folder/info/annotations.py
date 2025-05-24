from dataclasses import dataclass
from typing import List

@dataclass
class Annotations:
    onset:List[float]
    duration:List[float]
    description:List[str]

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
    pass