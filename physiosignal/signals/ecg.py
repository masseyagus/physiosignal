from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import numpy as np

# Futura implementación
class ECG(RawSignal):
    """
    Representa una señal de Electrocardiografía (ECG) con detección de picos R y cálculo de frecuencia cardiaca.

    Key Features:
        - Detección automática de picos R
        - Cálculo de frecuencia cardiaca (BPM)
        - Espectrogramas tiempo-frecuencia para análisis

    Attributes:
        data : np.ndarray
            Señal ECG cruda de forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Metadatos de canales.
        anotaciones : Annotations
            Eventos temporales asociados.
        first_samp : int
            Índice de la primera muestra respecto al inicio original.
        r_peaks : np.ndarray | None
            Índices de muestras donde se detectaron los picos R.
        heart_rate : float | None
            Frecuencia cardiaca en latidos por minuto (BPM).
    """
    
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True, r_picks:np.ndarray=None, heart_rate:int=None):
        """
        Inicializa una instancia de ECGSignal.

        Parameters:
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras) con la señal ECG cruda.
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
            r_peaks : np.ndarray, optional
                Índices de muestras con picos R detectados.
            heart_rate : float, optional
                Frecuencia cardiaca calculada en BPM.
        """
        
        super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)
        self.r_picks = r_picks
        self.heart_rate = heart_rate

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo