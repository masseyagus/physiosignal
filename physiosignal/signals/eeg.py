from __future__ import annotations

from physiosignal.signals import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import numpy as np

# Futura implementación
class EEG(RawSignal):
    """
    Representa una señal de Electroencefalografía (EEG) con herramientas específicas de análisis.

    Key Features:
        - Cambio de referencia (promedio, canal, laplaciano, etc.)
        - Filtro Laplaciano
        - Cálculo y visualización de espectro de Fourier
        - Generación de gráficas tiempo-frecuencia (espectrogramas)
        - Transformada de Hilbert (envolvente y fase)

    Attributes:
        data : np.ndarray
            Señal EEG cruda de forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Metadatos de canales (nombres, tipos, etc.).
        anotaciones : Annotations
            Anotaciones de eventos temporales.
        first_samp : int
            Índice de la primera muestra respecto al inicio original.
        reference : str
            Referencia actual aplicada ('promedio', 'canal', 'laplaciano', etc.).
    """
    
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True, reference:str='promedio'):
        """
        Inicializa una instancia de EEGSignal.

        Parameters:
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras) con la señal EEG cruda.
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
            reference : str, optional
                Tipo de referencia inicial ('promedio', 'canal', 'laplaciano', etc.).
        """
        
        super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)
        self.reference = reference

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo


    def change_reference(self):
        pass

    def laplace(self):
        pass

    def fft(self):
        pass

    def freq_time(self):
        pass

    def hilbert(self):
        pass