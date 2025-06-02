from physiosignal.utils import _checking, _sort
import numpy as np

class Annotations:

    def __init__(self, onset, duration, description, ch_names=None):
        # Hago comprobaciones de los parámetros
        self. onset, self.duration, self.description, self.ch_names = _checking(onset, duration, description, ch_names)
         
        # Ordeno los parámetros a fin de no tener errores
        _sort(self)

    def add(self, onset, duration, description, ch_names=None, duplicates=True):
        on, dur, des, ch = _checking(onset, duration, description, ch_names)

        self.onset = np.append(self.onset, on)
        self.duration = np.append(self.duration, dur)
        self.description = np.append(self.description, des) 

        if ch_names is not None:
            self.ch_names = np.append(self.ch_names, ch)

    def remove(self, onset=None, duration=None, description=None, ch_names=None):
        pass

    def get_annotations(self):

        import pandas as pd
        
        df = pd.DataFrame.from_dict(self.getInfo(), orient="columns")

        if self.ch_names is not None:
            return df
        else:
            df = df.drop(["ch_names"], axis=1)
            return df

    def find(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
    
    def getInfo(self):
        return {
            'onset': self.onset.tolist(),
            'duration': self.duration.tolist(),
            'description': self.description.tolist(),
            'ch_names': self.ch_names.tolist() if self.ch_names is not None else [None]*len(self.onset)
        }