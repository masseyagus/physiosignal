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
        
        duration = self.data.shape[1]/self.sfreq

        if start < 0 or stop < 0 or start > stop or stop > duration:
            raise ValueError(f"Valores inválidos: start debe estar entre [0, {round(duration, 1)}], " 
                             f"y stop no debe exceder {round(duration, 1)}")
        
        begin, end = int(start * self.sfreq), int(stop * self.sfreq)

        if start and stop:
            data = self.data[:, begin:end]
        elif start:
            data = self.data[:, begin:]
        elif stop:
            data = self.data[:, :end]
        else:
            sec = int(10 * self.sfreq) # Para 0, muestro 10 seg
            data = self.data[:, :sec]

        if picks is None:
            return data

        if isinstance(picks, str): # Canal en formato str
            try:
                idx = self.info.ch_names.index(picks)
            except:
                raise ValueError(f"El canal {picks} no existe dentro de ch_names")
            return data[idx, :]

        elif isinstance(picks, (list, tuple)): # varios canales
            idx = []
            for ch in picks:
                if isinstance(ch, str):
                    try:
                        idx.append(self.info.ch_names.index(ch))    
                    except:
                        raise ValueError(f"El canal {ch} no se encuentra dentro de ch_names")

                elif isinstance(ch, int):
                    idx.append(ch)

            return data[idx,:]     
        
        elif isinstance(picks, int):
            return data[picks, :]
        
        else:
            raise TypeError(f"El parámetro 'picks' debe ser str, int, list o tuple")

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