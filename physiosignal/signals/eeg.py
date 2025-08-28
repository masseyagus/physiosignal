from __future__ import annotations

from .raw import RawSignal
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

    def change_reference(self, reference:str, dic_ref:str, plot:bool, ch:str='Cz'):

        import json

        reference = reference.lower()

        if reference == 'canal':   
            ref_data = self.data[ch, :] # Canal de referencia
            new_data = self.data - ref_data[None, :] # Nueva referencia para todos los canales ref[None, :] inserta una nueva dimension

            self.data_canal = new_data

            if plot:
                pass

        elif reference == 'laplaciano':

            with open(dic_ref, "r") as f:
                ref_dic = json.load(f)

            name_to_idx = {ch: idx for idx, ch in enumerate(self.info.ch_names)}

            laplace = np.zeros_like(self.data, dtype=float)

            for idx, ch_name in enumerate(self.info.ch_names):

                if ch_name in ref_dic:
                    values_ch = ref_dic[ch_name]

                    neigh_idx = [name_to_idx[neigh] for neigh in values_ch if isinstance(neigh, str) and neigh in name_to_idx]

                    if len(neigh_idx) == 0:
                        laplace[idx, :] = self.data[idx, :].astype(float)
                    else:
                        laplace[idx, :] = self.data[idx, :].astype(float) - np.mean(self.data[neigh_idx, :], axis=0)
                else:
                    laplace[idx, :] = self.data[idx, :].astype(float)

            self.data_laplaciano = laplace

            if plot:
                pass

        elif reference == "promedio":

            ch_prom = np.mean(self.data, axis=0)

            avg_ref = self.data - ch_prom

            self.avg_ref = avg_ref

            if plot:
                pass

    def fft(self):
        pass

    def freq_time(self):
        pass

    def hilbert(self):
        pass