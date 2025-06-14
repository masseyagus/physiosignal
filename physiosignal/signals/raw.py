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


    def get_data(self, picks:str|np.array=None, start:float=None, stop:float=None, reject:float=None, times:bool=False):
        """
        Extrae datos de señales con opciones de recorte temporal, filtrado por amplitud y selección de canales.

        Args:
            picks: Canal(es) a seleccionar (nombre, índice o lista de ellos). Si None, se usan todos.
            start: Tiempo de inicio en segundos (None para iniciar desde el comienzo).
            stop: Tiempo de fin en segundos (None para llegar hasta el final).
            reject: Umbral de amplitud pico a pico para descartar canales (None para desactivar).
            times: Si es True, retorna también el vector de tiempos correspondiente.

        Returns:
            data: Array con forma (n_canales, n_muestras) o (n_muestras,) si un solo canal.
            time_vector (opcional): Vector 1D de tiempos en segundos si times=True.

        Notes:
            - Si se usa `reject`, se descartan canales cuya amplitud (máximo - mínimo) supere el umbral dado.
            - La recortación por `start` y `stop` se aplica antes de la selección de canales.
            - El vector de tiempos parte desde `start` si se especifica, o desde 0 por defecto.
            - El retorno será una tupla (data, times) solo si se solicita explícitamente.
        """
        duration = self.data.shape[1]/self.sfreq # Hallo la duración de la señal

        # Chequeo tiempos y genero las muestras en caso dado
        if start is not None and stop is not None:

            if start < 0 or stop < 0 or start > stop or stop > duration:
                raise ValueError(f"Valores inválidos: start debe estar entre [0, {round(duration, 1)}], " 
                                f"y stop no debe exceder {round(duration, 1)}")
            
            begin, end = int(start * self.sfreq), int(stop * self.sfreq)
        
        # Filtro por amplitud
        if reject is not None:

            if reject < 0:
                raise ValueError("reject debe ser >= 0")
            
            index = self.data.max(axis=1) - self.data.min(axis=1) # Obtengo el valor del pico
            pic_to_pic = index <= reject # Verifico que cumpla

            data = self.data[pic_to_pic,:]
        else:
            data = self.data # Si no se solicita filtro, genero la variable data con toda la información

        # Aplico segmentación si se solicita
        if start and stop:
            data = data[:, begin:end]
        elif start:
            data = data[:, begin:]
        elif stop:
            data = data[:, :end]

        # Selecciono los canales dados
        if picks is not None:
            if isinstance(picks, str): # Canal en formato str
                try:
                    idx = self.info.ch_names.index(picks)
                except:
                    raise ValueError(f"El canal {picks} no existe dentro de ch_names")
                data = data[idx, :]

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

                data = data[idx,:]     
            
            elif isinstance(picks, int):
                data = data[picks, :]
            
            else:
                raise TypeError(f"El parámetro 'picks' debe ser str, int, list o tuple")

        # Genero vector de tiempos en 1D  
        if times:
            time_sec = start if start else 0
            muestras = data.shape[1]

            time_vector = np.arange(muestras) / self.sfreq + time_sec  # Genero muestras uniformemente espaciadas y divido
                                                                         # por freq (sumo tiempo en caso de inicio distinto de 0)

            return data, time_vector

        return data

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