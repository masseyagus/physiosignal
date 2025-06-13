from __future__ import annotations
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from physiosignal.info import Info

from physiosignal.info import Info
from physiosignal.info import Annotations
from physiosignal.logger import log_config
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configuración global del logger
log_config(see_log=True)

logger = logging.getLogger(__name__)

# Futura implementación
class RawSignal:

    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, first_samp:int=0):

        self.data = data # Matriz con forma (n_canales, n_muestras)
        self.info = info # Objeto Info
        self.antoaciones = anotaciones # Objeto Annotations
        self.sfreq = self.info.sfreq if sfreq is None else sfreq
        self.first_samp = first_samp # Índice de la primera muestra


    def get_data(self, picks:str|np.array=None, start:float=0, stop:float=0, reject:float=0, times:bool=False):
        
        begin, end = int(start * self.sfreq), int(stop * self.sfreq)

        if start and stop:
            return self.data[:, begin:end]
        elif start:
            return self.data[:, begin:]
        elif stop:
            return self.data[:, :end]
        else:
            sec = int(10 * self.sfreq)
            return self.data[:, :sec]
            
            

    def drop_channels(self, ch_names) -> RawSignal:
        pass

    def crop(self, tmin, tmax) -> RawSignal:
        pass

    def describe(self):
        pass

    def filter(self, low_freq, high_freq, notch_freq, order) -> RawSignal:
        pass

    def pick(self, picks) -> RawSignal:
        pass

    def set_anotaciones(self, anotaciones):
        pass

    def plot(self, picks, start, duration, show_anotaciones):
        pass  

    def __getitem__(self):
        pass