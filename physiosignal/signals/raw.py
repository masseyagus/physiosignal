from __future__ import annotations

from physiosignal.info import Info
from physiosignal.info import Annotations
from physiosignal.logger import log_config
import matplotlib.pyplot as plt
import numpy as np
import logging

class RawSignal:
    """
    Representa una señal de datos crudos (por ejemplo, EEG) junto con su
    información de muestreo, metadatos de canales y anotaciones de eventos.

    Attributes
    ----------
    data : np.ndarray
        Matriz de forma (n_canales, n_muestras) con los valores de la señal.
    sfreq : float
        Frecuencia de muestreo en Hz.
    info : Info
        Objeto Info que contiene metadatos de los canales (nombres, tipos, etc.).
    anotaciones : Annotations
        Objeto Annotations con las marcas de eventos o anotaciones temporales.
    first_samp : int
        Índice de la primera muestra de `data` respecto al inicio de la grabación
        original. 
    """
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True):
        """
        Inicializa una instancia de RawSignal.

        Args:
            data : np.ndarray, optional
                Matriz con los datos de la señal, de forma (n_canales, n_muestras). Por defecto None.
            sfreq : float, optional
                Frecuencia de muestreo en Hz. Si se omite (None), se toma de `info.sfreq`. Por defecto None.
            info : Info, optional
                Objeto Info con metadatos de canales (nombres, tipos, etc.). Por defecto None.
            anotaciones : Annotations, optional
                Objeto Annotations con las marcas de eventos o anotaciones temporales. Por defecto None.
            first_samp : int, optional
                Índice de la primera muestra de `data` con respecto al inicio del registro original. Por defecto 0.
            see_log : bool, optional
                Si es True, activa la salida de mensajes del sistema de logging con
                nivel INFO. Si es False, suprime los mensajes. Por defecto True.
        """
        self.data = data # Matriz con forma (n_canales, n_muestras)
        self.info = info # Objeto Info
        self.anotaciones = anotaciones # Objeto Annotations
        self.sfreq = self.info.sfreq if sfreq is None else sfreq
        self.first_samp = first_samp # Índice de la primera muestra

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__)

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
            muestras = data.shape[1]
            offset_sec = self.first_samp / self.sfreq

            time_vector = np.arange(muestras) / self.sfreq + offset_sec  # Genero muestras uniformemente espaciadas y divido
                                                                                    # por freq (sumo tiempo en caso de inicio distinto de 0)

            return data, time_vector

        return data

    def drop_channels(self, ch_names:str|list|tuple, inplace:bool=True) -> RawSignal:
        """
        Elimina uno o varios canales de la señal.

        Args:
            ch_names: Nombre(s) de canal(es) a descartar.
                - str: un único canal.
                - list o tuple de str: varios canales.
            inplace: Si True, modifica el objeto actual y devuelve `self`.
                Si False, no altera el original y retorna una nueva instancia.

        Returns:
            RawSignal:
                - Si inplace=True: el mismo objeto (`self`) con los canales eliminados.
                - Si inplace=False: una nueva instancia de RawSignal con los cambios.

        Raises:
            TypeError:
                - Si `ch_names` no es str, list ni tuple.
                - Si algún elemento de `ch_names` no es str.
            ValueError:
                - Si algún nombre de canal no existe en `self.info.ch_names`.

        Notes:
            - Convierte siempre `ch_names` a lista de str antes de procesar.
            - Busca los índices de cada canal y los ordena en orden inverso
            para evitar desplazamientos al hacer pop.
            - Tanto `self.info.ch_names` como `self.info.ch_types` se copian y
            actualizan sin desordenar el original (salvo que inplace=True).
            - Se elimina la fila correspondiente en `self.data` usando `np.delete`.
        """
        if isinstance(ch_names, str):
            ch_names = [ch_names]
        elif isinstance(ch_names, (list, tuple)):
            ch_names = list(ch_names)
        else:
            raise TypeError(f"El parámetro ch_names debe ser list, tuple o str")

        idx = []

        for ch in ch_names:
            if isinstance(ch, str):
                try:
                    idx.append(self.info.ch_names.index(ch))
                except:
                    raise TypeError(f"Cada ítem en ch_names debe ser str o int; se recibió {type(ch).__name__}")
                
        drop_idx = sorted(idx, reverse=True)

        info_ch_names = list(self.info.ch_names)
        info_ch_types = list(self.info.ch_types)
        data = self.data.copy()

        for i in drop_idx:
            info_ch_names.pop(i)
            info_ch_types.pop(i)
            data = np.delete(data, i, axis=0)

        newInfo = Info(ch_names=info_ch_names, ch_types=info_ch_types, sfreq=self.sfreq)
        newRaw = RawSignal(data=data, info=newInfo, anotaciones=self.anotaciones, first_samp=self.first_samp)

        if inplace:
            self.info = newInfo
            self.data = data
            logging.info(f"Canales {ch_names} dropeados correctamente")
            return self
        else:
            logging.info(f"Nueva instancia RawSignal sin los canales: {ch_names}")
            return newRaw

    def crop(self, tmin, tmax) -> RawSignal:
        """
        Recorta la señal en un intervalo de tiempo especificado.

        Args:
            tmin: Tiempo de inicio en segundos (incluido).
            tmax: Tiempo de fin en segundos (excluido).

        Returns:
            RawSignal: Nueva instancia con los datos recortados entre tmin y tmax.

        Raises:
            ValueError: Si tmin < 0, tmax > duración total o tmin >= tmax.
        """

        if tmin is None:
            begin = 0
        else:
            begin = int(tmin * self.sfreq)

        crop_data = self.get_data(start=tmin, stop=tmax)

        new_first_samp = self.first_samp + begin

        return RawSignal(data=crop_data, sfreq=self.sfreq, info=self.info, anotaciones=self.anotaciones, first_samp=new_first_samp)

    def describe(self):
        pass

    def filter(self, low_freq, high_freq, notch_freq, order) -> RawSignal:
        pass

    def pick(self, picks) -> RawSignal:
        """
        Selecciona y extrae un subconjunto de canales de la señal.

        Args:
            picks: Canal(es) a seleccionar. Puede especificarse como:
                - str: nombre único de canal.
                - int: índice de canal.
                - list o tuple de str o int: múltiples canales.
                - None: todos los canales.

        Returns:
            RawSignal: Nueva instancia que contiene únicamente los canales seleccionados.
            
        Raises:
            TypeError: Si `picks` no es str, int, list o tuple.
            ValueError: Si algún canal o índice no existe en `self.info.ch_names`.
        """
        channels = self.get_data(picks=picks)

        return RawSignal(data=channels, sfreq=self.sfreq, info=self.info, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def set_anotaciones(self, anotaciones):
        pass

    def plot(self, picks, start, duration, show_anotaciones):
        pass  

    def __getitem__(self): # [canal, muestras], si no hay devuelvo array vacío
        pass