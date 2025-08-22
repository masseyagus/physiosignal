from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import numpy as np
import logging

# Futura implementación
class EMG(RawSignal):
    """
    Representa una señal de Electromiografía (EMG) con detección de activaciones musculares.

    Key Features:
        - Configuración de umbral de activación
        - Detección de segmentos de contracción
        - Espectrogramas tiempo-frecuencia para análisis

    Attributes:
        data : np.ndarray
            Señal EMG cruda de forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Metadatos de canales.
        anotaciones : Annotations
            Eventos temporales asociados.
        first_samp : int
            Índice de la primera muestra respecto al inicio original.
        activation_threshold : float | None
            Umbral de detección de activación en la señal.
        activation_times : np.ndarray | list | None
            Índices de muestras donde la señal supera `activation_threshold`.
    """
    
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True, activation_threshold: float = None, activation_times: np.ndarray | list = None):
        """
        Inicializa una instancia de EMGSignal.

        Parameters:
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras) con la señal EMG cruda.
            sfreq : float, optional
                Frecuencia de muestreo en Hz. Si None, se usa info.sfreq.
            info : Info, optional
                Metadatos de canales.
            anotaciones : Annotations, optional
                Eventos temporales asociados.
            first_samp : int, optional
                Índice de la primera muestra respecto al registro original.
            see_log : bool, optional
                Activa/desactiva logging interno.
            activation_threshold : float, optional
                Umbral de activación muscular en unidades de amplitud.
            activation_times : np.ndarray o list, optional
                Índices de muestras donde la señal supera el umbral.
        """
        
        super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)
        self.activation_threshold = activation_threshold
        self.activation_times = activation_times

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo