from __future__ import annotations

from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

def _type_check(datos, info, anotaciones, first_samp) -> tuple:
    """
    Valida los tipos y dimensiones de los argumentos requeridos por la clase RawSignal.

    Args:
        datos : np.ndarray
            Matriz bidimensional con los datos de la señal en forma (n_canales, n_muestras).
        info : Info
            Instancia de la clase Info que contiene metadatos de la señal.
        anotaciones : Annotations
            Instancia de la clase Annotations con eventos asociados a la señal.
        first_samp : int
            Índice de la primera muestra respecto al registro original.

    Returns:
        tuple
            Tupla con los parámetros validados en el siguiente orden: (datos, info, anotaciones, first_samp).

    Raises:
        TypeError:
            - Si `datos` no es un arreglo de NumPy.
            - Si `info` no es una instancia de la clase Info.
            - Si `anotaciones` no es una instancia de la clase Annotations.
            - Si `first_samp` no es un entero.
        ValueError:
            - Si `datos` no tiene exactamente dos dimensiones.

    Notes:
        Esta función se utiliza internamente por RawSignal para asegurar que los parámetros críticos
        cumplen con los requisitos mínimos antes de instanciar el objeto.
    """
    if not isinstance(datos, np.ndarray):
        raise TypeError(f"datos debe ser de tipo np.ndarray, pero se recibió: {type(datos).__name__}")
    
    if datos.ndim != 2:
        raise ValueError(f"La dimensión de los datos debe ser de 2, (n_canales, n_muestras), pero se recibió: {datos.ndim}")
    
    if not isinstance(info, Info):
        raise TypeError(f"info debe ser una instancia de la clase Info, pero se recibió: {type(info).__name__}")
    
    if not isinstance(anotaciones, Annotations):
        raise TypeError(f"anotacioens deben ser una instancia de la clase Annotations," 
                        f"pero se recibió: {type(anotaciones).__name__}")
    
    if not isinstance(first_samp, int):
        raise TypeError(f"first_samp debe ser un int, pero fue dado: {type(first_samp).__name__}")
    
    if first_samp >= datos.shape[1]:
        raise ValueError(f"first_samp debe ser menor a la cantidad de muestras: {datos.shape[1]}")
    
    return datos, info, anotaciones, first_samp

class RawSignal:
    """
    Representa una señal de datos crudos (por ejemplo, EEG) junto con su
    información de muestreo, metadatos de canales y anotaciones de eventos.

    Key Features:
        - Soporte para operaciones temporales (crop, segmentación)
        - Manejo integrado de anotaciones
        - Selección flexible de canales
        - Filtrado por amplitud y frecuencia
        - Visualización avanzada con PyQtGraph

    Attributes:
        data : np.ndarray
            Matriz de forma (n_canales, n_muestras) con los valores de la señal.
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Objeto Info que contiene metadatos de los canales.
        anotaciones : Annotations
            Objeto con marcas de eventos temporales.
        first_samp : int
            Índice de la primera muestra respecto al inicio original.

    Usage Examples:
        >>> # Crear señal desde numpy array
        >>> data = np.random.randn(64, 512*60)  # 64 canales, 1 minuto @512Hz
        >>> info = Info(ch_names=canales, ch_types=['eeg']*64, sfreq=512)
        >>> raw = RawSignal(data=data, info=info)
        >>>
        >>> # Recortar segmento
        >>> segmento = raw.crop(tmin=10, tmax=30)  # 20 segundos
        >>>
        >>> # Filtrar señal
        >>> filtrada = raw.filter(low_freq=1, high_freq=40, notch_freq=50)
    """
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True):
        """
        Inicializa una instancia de RawSignal.

        Parameters:
            data : np.ndarray
                Matriz de forma (n_canales, n_muestras) con los valores de la señal.
            sfreq : float, optional
                Frecuencia de muestreo en Hz. Si es None, se usará info.sfreq.
            info : Info
                Objeto Info con metadatos de canales (nombres, tipos, etc.).
            anotaciones : Annotations
                Objeto Annotations con eventos temporales asociados.
            first_samp : int
                Índice de la primera muestra respecto al registro original.
            see_log : bool, optional
                Activa o desactiva el logging interno (default=True).

        Raises:
            TypeError:
                - Si `data` no es np.ndarray.
                - Si `info` no es una instancia de Info.
                - Si `anotaciones` no es una instancia de Annotations.
                - Si `first_samp` no es int.
            ValueError:
                - Si `data` no es bidimensional (n_canales, n_muestras).

        Implementation Notes:
            - Se invoca `_type_check` para validar y asignar `data`, `info`,
            `anotaciones` y `first_samp`.
            - Si `sfreq` es None, se toma de `info.sfreq`; de lo contrario,
            se usa el valor suministrado.
            - Configura el sistema de logging según `see_log`.
        """
        self.data, self.info, self.anotaciones, self.first_samp = _type_check(data, info, anotaciones, first_samp)
        
        self.sfreq = self.info.sfreq if sfreq is None else float(sfreq)

        self.is_filtered = False # Se marca True al filtrar con filter()

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo

    def get_data(self, picks:str|np.array=None, start:float=None, stop:float=None, reject:float=None, times:bool=False):
        """
        Extrae datos de señales con recorte temporal, selección de canales y
        filtrado por amplitud pico-a-pico.

        Processing Pipeline:
            1. Recorte temporal según start/stop (s).
            2. Selección de canales (picks).
            3. Filtrado por amplitud (reject).

        Args:
            picks : str, int, list o tuple, optional
                Canal(es) a seleccionar. None = todos.
            start : float, optional
                Tiempo de inicio en segundos; None = 0.0.
            stop : float, optional
                Tiempo de fin en segundos; None = duración completa.
            reject : float, optional
                Umbral pico-a-pico en µV para descartar canales ruidosos.
            times : bool, optional
                Si True, también devuelve vector de tiempos.

        Returns:
            np.ndarray or (np.ndarray, np.ndarray):
                - data: Array de forma (n_canales, n_muestras) o (1, n_muestras) si un canal.
                - time_vector: Vector de tiempos en s (solo si times=True).

        Raises:
            ValueError: Si start/stop fuera de rango o reject < 0.
            TypeError: picks no es str, int, list ni tuple.

        Notes:
            - El filtrado por amplitud se aplica después de la selección de canales.
            - El time_vector está desplazado por first_samp.
        """
        duration = self.data.shape[1]/self.sfreq # Hallo la duración de la señal
        data = self.data.copy()

        start = 0.0 if start is None else float(start)
        stop = duration if stop is None else float(stop) 

        # Chequeo tiempos y genero las muestras en caso dado
        if start < 0 or stop < 0 or start > stop or stop > duration:
            raise ValueError(f"Valores inválidos: start debe estar entre [0, {round(duration, 1)}], " 
                            f"y stop no debe exceder {round(duration, 1)}")
            
        begin, end = int(start * self.sfreq), int(stop * self.sfreq)

        # Aplico segmentación si se solicita
        if start is not None and stop is not None:
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
            
        # Filtro por amplitud
        if reject is not None:

            if reject < 0:
                raise ValueError("reject debe ser >= 0")
            
            index = data.max(axis=1) - data.min(axis=1) # Obtengo el valor del pico
            pic_to_pic = index <= reject # Verifico que cumpla

            data = data[pic_to_pic,:]
        # else:
        #     data = data # Si no se solicita filtro, genero la variable data con toda la información

        # Genero vector de tiempos en 1D  
        if times:
            muestras = data.shape[1]
            offset_sec = self.first_samp / self.sfreq

            time_vector = np.arange(muestras) / self.sfreq + offset_sec  # Genero muestras uniformemente espaciadas y divido
                                                                         # por freq (sumo tiempo en caso de inicio distinto de 0)

            return np.atleast_2d(data), time_vector

        return np.atleast_2d(data)

    def drop_channels(self, ch_names:str|list|tuple, inplace:bool=True) -> RawSignal:
        """
        Elimina uno o varios canales de la señal.

        Args:
            ch_names: Nombre(s) de canal(es) a descartar.
                - str: un único canal.
                - list o tuple de str: varios canales.
            inplace: Si True, modifica el objeto actual y devuelve self.
                Si False, retorna una nueva instancia sin modificar el original.

        Returns:
            RawSignal:
                - Si inplace=True: el mismo objeto (self) actualizado.
                - Si inplace=False: nueva instancia RawSignal sin los canales.

        Raises:
            TypeError: Si ch_names no es str, list ni tuple, o contiene elementos no str.
            ValueError: Si algún nombre de canal no existe en info.ch_names.

        Notes:
            - Actualiza tanto los datos como los metadatos de canales.
            - Mantiene la integridad de las anotaciones.
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
                    raise TypeError(f"Cada ítem en ch_names debe ser str; se recibió {type(ch).__name__}")
                
        drop_idx = sorted(idx, reverse=True) # Índice a dropear

        info_ch_names = list(self.info.ch_names) # Nombre de canales en self.info
        info_ch_types = list(self.info.ch_types) # Tipos de canales en self.info
        data = self.data.copy()

        for i in drop_idx:
            info_ch_names.pop(i) # Quito nombre de canal de la variable info_ch_names
            info_ch_types.pop(i) # Quito tipo de canal de la variable info_ch_types
            data = np.delete(data, i, axis=0) # Elimino el canal de los datos

        # Nuevas instancias
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

    def crop(self, tmin:int=None, tmax:int=None) -> RawSignal:
        """
        Recorta la señal en un intervalo de tiempo.

        Temporal Processing:
            - Recorta datos entre tmin y tmax.
            - Ajusta anotaciones restando tmin.
            - Actualiza first_samp.

        Args:
            tmin : float, optional
                Tiempo de inicio en s (inclusive); None = 0.
            tmax : float, optional
                Tiempo de fin en s (exclusive); None = fin.

        Returns:
            RawSignal: Nueva instancia con datos recortados.

        Raises:
            ValueError: Si tmin > tmax.

        Notes:
            - Anotaciones fuera de rango se eliminan.
            - La columna `ch_names` es opcional y se incluirá solo si está presente en las anotaciones.
        """
        crop_data = self.get_data(start=tmin, stop=tmax)

        duration = self.data.shape[1]/self.sfreq
        tmin = 0.0 if tmin is None else float(tmin)
        tmax = duration if tmax is None else float(tmax)        
        new_first_samp = self.first_samp + int(tmin * self.sfreq)

        # Actualizo anotaciones
        df = self.anotaciones.get_annotations()
        required_cols = ['onset', 'duration', 'description']

        if df is not None and not df.empty:
            # Verifico la existencia de las columnas onset, duration, description
            if not set(required_cols) <= set(df.columns):
                raise ValueError(f"Las anotaciones deben poseer las siguientes columnas: {required_cols}")
            
            mask = (df['onset'] >=tmin) & (df["onset"] <= tmax)
            filtered = df[mask].copy()

            filtered.loc[:, 'onset'] = filtered['onset'] - tmin

        else:
            filtered = df

        onset = filtered["onset"]
        duration = filtered['duration']
        description = filtered['description']
        ch_names = filtered['ch_names'] if "ch_names" in filtered.columns else None

        new_ann = Annotations(onset=onset, duration=duration, description=description, ch_names=ch_names)
        logging.info("Señal recortada correctamente")

        crop_raw = RawSignal(data=crop_data, sfreq=self.sfreq, info=self.info, anotaciones=new_ann, first_samp=new_first_samp)

        if self.is_filtered:
            crop_raw.is_filtered = True

        return crop_raw

    def describe(self, channels:str|list):
        """
        Genera estadísticas descriptivas para los canales especificados.

        Args:
            channels: Nombre(s) de canal(es) a analizar.

        Returns:
            pd.DataFrame: DataFrame con estadísticas por canal:
                - min: Valor mínimo
                - Q1: Primer cuartil (25%)
                - mediana: Mediana (50%)
                - Q3: Tercer cuartil (75%)
                - max: Valor máximo

        Example Output:
                   FP1       FP2
            min    -125.32   -118.45
            Q1      -15.21    -12.33
            mediana   0.05      0.12
            Q3       18.76     15.89
            max     105.89     92.17

        Notes:
            - Las estadísticas se calculan sobre toda la duración de la señal.
            - Para un solo canal, el DataFrame tiene una sola columna.
        """    
        segment = self.pick(channels)
        data = segment.data

        if data.ndim == 1:
            data = data[np.newaxis, :] # np.newaxis, incrementa en 1 la dimensión, volviendo array 2D.

        results = {}

        for name, ch_data in zip(segment.info.ch_names, data):
            results[name] = {
                "min": float(np.min(ch_data)),
                "Q1": float(np.percentile(ch_data, 25)),
                "mediana": float(np.percentile(ch_data, 50)),
                "Q3": float(np.percentile(ch_data, 75)),
                "max": float(np.max(ch_data)),
            }

        df = pd.DataFrame(results)  # columnas → canales, filas → estadísticas
        return df

    def filter(self, low_freq:float = None, high_freq:float = None, order:int = 4, 
               notch_freq:float|list[float] = None, q:int =30, inplace:bool=False, smooth:bool=False) -> RawSignal:
        """
        Aplica filtros de frecuencia (Notch y/o Butterworth) y un suavizado final.

        Esta función procesa la señal `self.data` aplicando hasta tres etapas de
        filtrado en cascada:
        1. Uno o más filtros Notch (IIR) para eliminar frecuencias específicas
           (ej. 50Hz y sus armónicos).
        2. Un filtro Butterworth (pasa-alto, pasa-bajo o pasa-banda) para 
           aislar la banda de interés.
        3. Un filtro de suavizado Savitzky-Golay (savgol_filter) final.

        La función puede modificar la instancia actual (`inplace=True`) o devolver 
        una nueva instancia `RawSignal` con los datos filtrados.

        Args:
            low_freq (float | None, optional): 
                Frecuencia de corte (Hz) para el filtro pasa-alto o el límite 
                inferior del filtro pasa-banda.
            high_freq (float | None, optional): 
                Frecuencia de corte (Hz) para el filtro pasa-bajo o el límite 
                superior del filtro pasa-banda.
            notch_freq (float | list | None, optional): 
                Frecuencia o lista de frecuencias (Hz) a eliminar con el filtro(s) notch.
            order (int, optional): 
                Orden del filtro Butterworth. Por defecto 4.
            q (float, optional): 
                Factor de calidad (Q) para el/los filtros notch. Por defecto 30.
            inplace (bool, optional): 
                Si es True, modifica `self.data` en la instancia actual y la marca
                como filtrada. Si es False (por defecto), retorna una nueva instancia
                `RawSignal` con los datos filtrados.

        Returns:
            RawSignal | self: 
                Una nueva instancia `RawSignal` con los datos filtrados (si `inplace=False`), 
                o la propia instancia modificada (si `inplace=True`).

        Raises:
            ValueError: 
                - Si no se especifica ningún parámetro de filtrado (low_freq, high_freq, o notch_freq).
                - Si alguna frecuencia está fuera del rango válido (0 a Nyquist).
                - Si `low_freq >= high_freq`.
        
        Behavior / Notes:
            - Orden de Aplicación: Los filtros se aplican en serie: 
              1º Filtro(s) Notch.
              2º Filtro Butterworth.
              3º Filtro Savitzky-Golay.
            - Tipo de Filtro: El tipo de Butterworth (pasa-alto, pasa-bajo, pasa-banda) 
              se determina automáticamente según qué combinaciones de `low_freq` y 
              `high_freq` se provean.
            - Filtro Notch: Si se provee una lista, se aplica un filtro por cada 
              frecuencia en cascada, uno después del otro.
            - Suavizado Final: Se aplica un filtro `savgol_filter`, en caso de `smooth=True`,
              (con window_length=11, polyorder=3) a la señal resultante de los 
              filtros anteriores.
            - Estado: La instancia (nueva o actual) se marca con `self.is_filtered = True`.
            - Phase Shift: Todos los filtros (Notch y Butterworth) se aplican usando
              `signal.filtfilt` para evitar introducir desfase en la señal.
        """
        import scipy.signal as signal

        # Validación de parámetros
        if low_freq is None and high_freq is None and notch_freq is None:
            raise ValueError("Debe especificar al menos una frecuencia de corte (low_freq, high_freq o notch_freq)")

        if low_freq is not None and low_freq < 0:
            raise ValueError("low_freq debe ser mayor o igual a 0")

        if high_freq is not None and low_freq is not None and low_freq >= high_freq:
            raise ValueError("low_freq debe ser menor que high_freq")

        if high_freq is not None and high_freq > self.sfreq / 2:
            raise ValueError(f"high_freq no puede exceder la frecuencia de Nyquist ({self.sfreq/2} Hz)")
        
        # Copia de la señal para procesar
        processed_signal = self.data.copy()
        
        for notch in notch_freq if isinstance(notch_freq, list) else [notch_freq]:

            if notch <= 0 or notch >= self.sfreq / 2:
                raise ValueError("notch_freq debe estar entre 0 y la frecuencia de Nyquist")
            
            # Crear filtro notch
            b_notch, a_notch = signal.iirnotch(notch, q, self.sfreq)
            
            # Aplicar filtro notch
            processed_signal = signal.filtfilt(b_notch, a_notch, processed_signal, axis=-1)

        b, a = None, None

        # Aplicar filtro Butterworth según los parámetros
        if low_freq is not None and high_freq is not None:
            # Filtro pasa-banda
            b, a = signal.butter(order, [low_freq, high_freq], btype='band', fs=self.sfreq)
        elif low_freq is not None:
            # Filtro pasa-alto
            b, a = signal.butter(order, low_freq, btype='high', fs=self.sfreq)
        elif high_freq is not None:
            # Filtro pasa-bajo
            b, a = signal.butter(order, high_freq, btype='low', fs=self.sfreq)

        if b is not None:
        # Aplicar el filtro Butterworth
            filtered_signal = signal.filtfilt(b, a, processed_signal, axis=-1)
        else:
            filtered_signal = processed_signal

        if smooth:
            filtered_signal = signal.savgol_filter(filtered_signal, window_length=11, polyorder=3)

        if inplace:
            self.data = filtered_signal
            self.is_filtered = True
            return self
        
        # Crear nueva instancia con la señal filtrada
        filtered_signal = RawSignal(data=filtered_signal, sfreq=self.sfreq, info=self.info, 
                                    anotaciones=self.anotaciones, first_samp=self.first_samp)
        
        filtered_signal.is_filtered = True

        return filtered_signal

    def pick(self, picks) -> RawSignal:
        """
        Selecciona y extrae un subconjunto de canales de la señal.

        Args:
            picks: Canal(es) a seleccionar. Puede ser:
                - str: nombre de un canal
                - int: índice de un canal
                - list: múltiples canales (nombres o índices)
                - None: todos los canales

        Returns:
            RawSignal: Nueva instancia con solo los canales seleccionados.
            
        Raises:
            TypeError: Si picks tiene un tipo no soportado.
            ValueError: Si algún canal o índice no existe.

        Notes:
            - Actualiza los metadatos de canales en la nueva instancia.
            - Realiza una copia profunda de los metadatos.
        """
        from copy import deepcopy
        # Obtengo la info de los canales (n_canales, n_muestras)
        channels = self.get_data(picks=picks)

        # Actualizo los canales 
        new_info = deepcopy(self.info)
        new_info._select(select=picks)

        return RawSignal(data=channels, sfreq=self.sfreq, info=new_info, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def set_anotaciones(self, anotaciones:Annotations):
        """
        Asigna nuevas anotaciones a la señal.

        Args:
            anotaciones : Annotations
                Objeto con eventos temporales.

        Raises:
            TypeError: anotaciones no es Annotations.
            ValueError: onset fuera de rango.

        Notes:
            - Valida onset entre 0 y duración.
        """
        if isinstance(anotaciones, Annotations):
            raise TypeError(f"El parámetro 'anotaciones' debe ser una instancia de 'Annotations'")
        
        time_signal = self.data.shape[1]/self.sfreq
        max_onset = anotaciones.onset.max()
        min_onset = anotaciones.onset.min()
        
        if max_onset > time_signal:
            raise ValueError(f"El 'onset' de la anotación excede el tiempo de la señal: {max_onset} vs {time_signal}")
        
        if min_onset < 0.0:
            raise ValueError(f"El 'onset' es menor al tiempo de la señal: {min_onset} vs {max_onset}")
        
        self.anotaciones = anotaciones

        logging.info("Anotación asignada correctamente")
    
    def plot(self, picks=None, start=None, duration=None, show_anotaciones: bool = True):
        """
        Visualización interactiva con PyQtGraph.

        Características:
            - Scroll vertical de canales.
            - Líneas de anotación.
            - Escalado Y por tipo de señal (adaptativo y seguro frente a NaN/inf/outliers).
            - Ventana adaptable al número de canales.
            - Detección automática de unidades/escala implícita (intenta evitar rangos absurdos).

        Args:
            picks : str, list[str] optional
                Canales a visualizar. Si es None se muestran todos los canales disponibles.
                Si se pasa una lista, los nombres que no existan se ignorarán.
            start : float, optional
                Tiempo inicial en s. Si es None, empieza en 0.0 s.
            duration : float, optional
                Duración del segmento en s. Si es None se muestra hasta el final.
            show_anotaciones : bool, optional
                Mostrar líneas de eventos/anotaciones en la ventana.

        Implementation Details:
            - Usa PyQt5 y pyqtgraph para la interfaz y el renderizado.
            - La función obtiene datos mediante `self.get_data(picks=..., start=..., stop=..., times=True)`.
            - Se aplican medidas de saneamiento a las señales antes de calcular límites Y:
                * Se filtran NaN/inf con np.nan_to_num.
                * Se calculan percentiles (1 y 99) y padding adaptativo.
                * Se usan fallbacks basados en desviación estándar si la señal es constante.
                * Se aplican topes absolutos para evitar rangos numéricos excesivos que causen
                overflows/RuntimeWarning en pyqtgraph..
            - Si hay anotaciones, se crea una leyenda horizontal con colores asignados por
            descripción de evento.
            - La ventana creada usará la instancia de QApplication existente si ya existe.

        Returns:
            None (muestra la ventana interactiva). La ejecución termina en sys.exit(app.exec_())
            cuando la ventana se cierra.
        """
        import pyqtgraph as pg
        import sys
        from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                                    QScrollArea, QVBoxLayout, QWidget, QDesktopWidget, QLabel, QHBoxLayout)
        from PyQt5.QtCore import Qt
        import itertools

        # 1. Configuración inicial de datos
        max_time = self.data.shape[1] / self.sfreq
        start = 0.0 if start is None else start
        stop = start + duration if duration and start + duration < max_time else max_time
        
        data, times = self.get_data(picks=picks, start=start, stop=stop, times=True)
        n_chan, n_samp = data.shape

        # 2. Manejo de anotaciones
        ann_df = self.anotaciones.get_annotations() # Obtengo el DataFrame de anotaciones
        mask = (ann_df['onset'] >= start) & (ann_df['onset'] <= stop) # Máscara booleana dentro del intervalo
        ann_filtered = ann_df.loc[mask] # Filtro el DataFrame
        ann_rel = (ann_filtered['onset'] - start).values # Calculo tiempos relativos de cada anotación
        ann_desc = ann_filtered['description'].values # Obtengo las descripciones del DataFrame filtrado
        
        colores_disponibles = ["#FF0000", "#9000FF", "#0000FF", "#FFA500", "#800080"] # Plantilla de colores
        color_cycle = itertools.cycle(colores_disponibles) # Creo secuencia infinita del iterable dado
        descripciones_unicas = np.unique(ann_desc) # Elimino descripciones repetidas
        color_dict = {desc: next(color_cycle) for desc in descripciones_unicas} # A cada descripción le asigno un color del iterable

        # 3. Configuración de la interfaz gráfica
        app = QApplication.instance() or QApplication(sys.argv) # Genero la instancia de QApplication o uso la existente
        main_win = QMainWindow() # Creo la ventana principal de la app
        main_win.setWindowTitle(f"Visualizador de Señales - {self.__class__.__name__}") # Asigno nombre a la ventana usando el nombre de la clase
        
        # Obtener tamaño de pantalla disponible
        screen = QDesktopWidget().availableGeometry() # Halla el espacio utilizable de mi pantalla (objeto QRect)
        screen_width = screen.width() # Obtengo el ancho útil de la pantalla
        screen_height = screen.height() # Obtengo la altura útil de la pantalla

        # Calcular dimensiones de ventana basadas en número de canales
        BASE_HEIGHT_PER_CHANNEL = 250  # Asigno la altura base con la que se muestra cada canal
        MIN_WINDOW_HEIGHT = 450        # Altura mínima que tiene la ventana que abre Qt
        MAX_WINDOW_HEIGHT = int(screen_height * 0.9)  # Limito la altura a un 90% útil de la pantalla
        
        # Calculo altura de la ventana Qt
        window_height = min(
            n_chan * BASE_HEIGHT_PER_CHANNEL + 100,  # Base + espacio extra
            MAX_WINDOW_HEIGHT
        )
        window_height = max(window_height, MIN_WINDOW_HEIGHT) # Si la altura es muy chica, tomo el mínimo
        
        # Establecer tamaño de ventana
        main_win.resize(1200, window_height) # 1200 px de altura y ancho de px calculados

        # Configurar fondo blanco global para PyQtGraph
        pg.setConfigOption('background', 'w') # Aplico fondo blanco a todo
        pg.setConfigOption('foreground', 'k')  # Aplico negro a texto, ejes y líneas

        # Widget central con área de scroll
        scroll = QScrollArea() # Creo la barra de scroll
        scroll.setWidgetResizable(True) # Permito que la barra se pueda redimensionar
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn) # Fuerzo a que la barra vertical siempre se visualice
        scroll.setStyleSheet("background-color: black;")  # Asigno color a la barra
        
        # Contenedor para los canales
        container = QWidget() # Creo un Widget para que contenga a los gráficos
        container.setStyleSheet("background-color: white;") # Aplico fondo blanco al Widget
        container_layout = QVBoxLayout(container) # Organiza los Widgets en vertical (un canal debajo del otro)
        container_layout.setAlignment(Qt.AlignTop) # Alinea todos los elementos
        container_layout.setSpacing(5)  # Establezco el espaciado entre canales o gráficas
        
        # 4. Parámetros visuales ajustables
        # Calcular altura por canal basada en número de canales
        MIN_CHANNEL_HEIGHT = 250 # Altura mínima del gráfico para cada canal
        MAX_CHANNEL_HEIGHT = 450 # Altura máxima del gráfico para cada canal
        
        if n_chan <= 4:
            # Para pocos canales, calculo dinámicamente la altura de cada canal
            CHANNEL_HEIGHT = min(MAX_CHANNEL_HEIGHT, int(window_height / n_chan * 0.8))
        else:
            # Para varios canales, tomo el mínimo de altura
            CHANNEL_HEIGHT = MIN_CHANNEL_HEIGHT
            
        # Asegurar altura mínima y máxima. Du´plico código para evitar errores de visualización
        CHANNEL_HEIGHT = max(CHANNEL_HEIGHT, MIN_CHANNEL_HEIGHT)
        CHANNEL_HEIGHT = min(CHANNEL_HEIGHT, MAX_CHANNEL_HEIGHT)
        
        SPACING = 5  # Espacio entre canales. 

        # 5. Función para determinar límites Y inteligentes
        def get_ylimits(signal):
            """
            Calcula límites Y adaptativos y seguros para plotear una señal 1D.

            - signal: array-like 1D (valores de la señal)
            - ch_type: string con tipo de señal ('ecg','eeg', etc.) (no obligatorio)
            Devuelve: (y_min, y_max) como floats seguros y finitos.
            """
            # ---- 0) convertir y sanear entrada ----
            sig = np.asarray(signal, dtype=np.float64)
            if sig.size == 0:
                return (-1.0, 1.0)

            # Reemplaza NaN/inf por valores finitos controlados
            # posinf/neginf se sustituyen por valores grandes pero finitos
            finfo = np.finfo(np.float64)
            LARGE = min(finfo.max / 1e6, 1e12)  # límite grande pero no extremo
            sig = np.nan_to_num(sig, nan=0.0, posinf=LARGE, neginf=-LARGE)

            # ---- 1) percentiles y estadísticas robustas ----
            try:
                p1, p99 = np.percentile(sig, [1, 99])
            except Exception:
                # si falla con percentiles (por ejemplo señal plana muy rara), usamos min/max
                p1, p99 = float(np.nanmin(sig)), float(np.nanmax(sig))

            median = float(np.nanmedian(sig))
            span = float(p99 - p1)

            # ---- 2) fallback si la señal es esencialmente constante ----
            if np.isclose(span, 0.0):
                std = float(np.nanstd(sig))
                if std > 0:
                    # rango basado en desviación estándar
                    padding = max(std * 3.0, 1e-6)
                    y_min, y_max = median - padding, median + padding
                else:
                    # señal completamente plana: darle un rango pequeño relativo a la magnitud
                    mag = max(abs(median), 1.0)
                    tiny = max(1e-6, 0.01 * mag)  # 1% de la magnitud como margen mínimo
                    y_min, y_max = median - tiny, median + tiny
            else:
                # ---- 3) padding adaptativo en caso normal ----
                # padding mínimo relativo a la magnitud para señales muy pequeñas
                padding = max(0.3 * span, 0.01 * max(abs(median), 1.0))
                y_min, y_max = p1 - padding, p99 + padding

            # ---- 4) evitar límites excesivos (cap absoluto) ----
            ABS_CAP = 1e9  # tope absoluto razonable (1e9 unidades).
            # aplica cap de manera que los límites no causen overflow interno al comparar
            y_min = float(np.clip(y_min, -ABS_CAP, ABS_CAP))
            y_max = float(np.clip(y_max, -ABS_CAP, ABS_CAP))

            # ---- 5) asegurar orden y separación mínima ----
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                # fallback seguro
                y_min, y_max = -1.0, 1.0

            if y_max <= y_min:
                # crear un pequeño margen si quedaron iguales o invertidos
                mag = max(abs(median), 1.0)
                delta = max(1e-6, 0.01 * mag)
                y_min, y_max = median - delta, median + delta

            # ---- 7) devolver floats seguros ----
            return float(y_min), float(y_max)

        # 6. Crear gráfico para cada canal con escalado inteligente
        ch_names = self.info.ch_names if picks is None else [ch for ch in self.info.ch_names if ch in picks]

        # Itero sobre cada canal
        for idx, ch_name in enumerate(ch_names):
            # Obtengo el tipo de canal (si está disponible)
            ch_type = self.info.ch_types[idx].lower() if idx < len(self.info.ch_types) else 'unknown'
            
            # Widget para cada canal
            channel_widget = pg.PlotWidget() # Creo un widget para contener al canal
            channel_widget.setBackground('w')  # Le aplico fondo blanco
            
            # Configurar colores de ejes
            channel_widget.getAxis('left').setPen(pg.mkPen('k')) # Coloreo en negro el eje izquirdo (Eje Y)
            channel_widget.getAxis('bottom').setPen(pg.mkPen('k')) # Coloreo en negro el eje debajo (Eje X)

            # Defino la altura para ese canal en específico
            channel_widget.setMinimumHeight(CHANNEL_HEIGHT) # Altura mínima
            channel_widget.setMaximumHeight(CHANNEL_HEIGHT) # Altura máxima
            
            # Grafico la señal
            plot_item = channel_widget.plot(times, data[idx, :], 
                            pen=pg.mkPen("#1f77b4", width=1.5)) # Pen especifica el trazado de la línea
            
            # Configuración de ejes
            channel_widget.setLabel('left', ch_name, 
                                **{'color': 'k', 'font-size': '12pt',
                                   'font-family':'Times New Roman'}) # Configuro el nombre del eje de ese canal y su formato
            
            # Configurar límites Y adaptativos (uso función get_ylimits())
            y_min, y_max = get_ylimits(data[idx, :])
            channel_widget.setYRange(y_min, y_max)
            
            # Mostrar solo el eje X en el último canal
            if idx == n_chan - 1: # Verifico que sea la última gráfica (último canal)
                # Personalizo el eje, selecciono color y texto
                channel_widget.setLabel('bottom', 'Tiempo (s)', **{'color':'k', 'font-size':'12pt', 
                                                                   'font-family':'Times New Roman',
                                                                   'font-style': 'italic'})
                channel_widget.showAxis('bottom') # Muestro el eje
            else:
                channel_widget.hideAxis('bottom') # Si no es el úlltimo canal, lo oculto
            
            # Añadir anotaciones
            if show_anotaciones:
                for onset_rel, desc in zip(ann_rel, ann_desc): 
                    color = color_dict[desc] # Obtengo el color asignado a al descripción
                    # Creo una línea vertical infinita con vline para la anotación
                    vline = pg.InfiniteLine(
                        pos=onset_rel, #La posiciono en el tiempo relativo
                        angle=90, # Ángulo de la línea (vertical)
                        pen=pg.mkPen(color, style=pg.QtCore.Qt.DashLine, width=1.5) # Indico grosor y 
                                                                                    # color de línea (mismo que su descripción)
                    )
                    channel_widget.addItem(vline) # Añado la línea al gráfico del canal
            
            # Añado el gráfico al layut vertical
            container_layout.addWidget(channel_widget)

        # 7. Leyenda (solo si se muestran anotaciones y hay eventos)
        if show_anotaciones and len(color_dict) > 0:
            legend_container = QWidget() # Creo un widget que contenga a la leyenga (legend())
            legend_container.setStyleSheet("background-color: white;") # Indico el color de fondo

            hbox = QHBoxLayout(legend_container) # Creo un layout horizontal asociado al widget legend_container
            hbox.setContentsMargins(10,10,10,10) # Ajusto los márgenes interiores
            hbox.setSpacing(15)  # Espacio de 15 px entre cada widget
            
            # Añadir todos los ítems de la leyenda
            for desc, color in color_dict.items(): # Recorro cada descripción y su color asignado
                # un rectángulo de color
                color_rect = QLabel() # Creo un widget vacío
                color_rect.setFixedSize(20, 10) # Asigno tamaño a QLabel
                color_rect.setStyleSheet(f"background-color: {color}; border: 1px solid black;") # Aplico estilo, color y grosor
                # la etiqueta de texto
                text_lbl = QLabel(desc) # Creo otro widget para el texto
                text_lbl.setStyleSheet("color: black;") # Aplico color negro al texto
                hbox.addWidget(color_rect) # Añado el color del rectángulo al layout
                hbox.addWidget(text_lbl) # Añado el texto al lado del widget con el color
            
            # Añadir al layout
            container_layout.addWidget(legend_container) # Añado todo al layout principal (el que contiene los canales)
            hbox.setAlignment(Qt.AlignLeft)
        
        # 8. Configuración final del scroll
        scroll.setWidget(container)
        main_win.setCentralWidget(scroll)
        main_win.show() # Muestro la ventana en pantalla
        
        # 9. Ajustar tamaño del contenedor
        legend_height = 60 if (show_anotaciones and len(color_dict) > 0) else 0 # Estimo la altura de la leyenda
        total_height = n_chan * CHANNEL_HEIGHT + legend_height + (n_chan * SPACING) # Calculo la altura total del contenedor de leyenda
        container.setMinimumHeight(total_height) # Ajusto el contenedor si el contenido excede el alto
        
        # Ejecuto la ventana y cierro al cerrar la ventana
        sys.exit(app.exec_()) 
    
    def plot_filtered(self, filtered_signal, tmin:int=0, tmax:int=10, channel:int=0):
        """
        Grafica comparación señal original vs filtrada.

        Args:
            filtered_signal : np.ndarray
                Señal filtrada (n_canales, n_muestras).
            tmin : float, optional
                Tiempo inicial en s (inclusive).
            tmax : float, optional
                Tiempo final en s (exclusive).
            channel : int, optional
                Índice del canal a mostrar.

        Raises:
            ValueError: tmin >= tmax o tmin < 0.

        Notes:
            - Ajusta límites Y según tipo de canal.
        """
        n_samps = self.data.shape[1]
        t = np.arange(n_samps) / self.sfreq
        
        if tmin >= tmax or tmin < 0:
            raise ValueError(f"tmin >= tmax o tmin < 0")
        
        min_sample = tmin * self.sfreq
        max_sample = (n_samps/self.sfreq) if tmax > (n_samps/self.sfreq) else tmax * self.sfreq

        # Graficamos la señal original y la señal filtrada
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t[min_sample:max_sample], self.data[channel, min_sample:max_sample], label='Señal Original', color='blue')
        plt.title('Señal Original')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        if self.info.ch_types[0] == "ecg":
            plt.ylim(-200, 400)
        elif self.info.ch_types[0] == "eeg":
            plt.ylim(-100, 100)
        else:
            plt.ylim(-300, 300)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t[min_sample:max_sample], filtered_signal[channel, min_sample:max_sample], label='Señal Filtrada', color='red')
        plt.title('Señal Filtrada')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.tight_layout()
        if self.info.ch_types[0] == "ecg":
            plt.ylim(-200, 400)
        elif self.info.ch_types[0] == "eeg":
            plt.ylim(-100, 100)
        else:
            plt.ylim(-300, 300)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_spectrum(self, filtered_signal, low_freq, high_freq, notch_freq, ch_idx):
        """
        Grafica PSD original vs filtrada de un canal.

        Args:
            filtered_signal : np.ndarray
                Datos filtrados (n_canales, n_muestras).
            low_freq : float
                Corte inferior en Hz.
            high_freq : float
                Corte superior en Hz.
            notch_freq : float
                Frecuencia del notch.
            ch_idx : int
                Índice de canal a mostrar.

        Notes:
            - PSD en dB (μV²/Hz).
            - Marca frecuencias de corte.
        """
        import scipy.signal

        f_orig, psd_orig = scipy.signal.welch(self.data, fs=self.sfreq, nperseg=1024, axis=-1)
        f_filt, psd_filt = scipy.signal.welch(filtered_signal, fs=self.sfreq, nperseg=1024, axis=-1)

        # Convertir a dB (con referencia estándar de 1 μV²/Hz)
        psd_orig_db = 10 * np.log10(psd_orig)  # dB re: 1 μV²/Hz
        psd_filt_db = 10 * np.log10(psd_filt)  # dB re: 1 μV²/Hz

        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(f_orig, psd_orig_db[ch_idx,:], 'b', label='Espectro Original')
        plt.plot(f_filt, psd_filt_db[ch_idx,:], 'r', label='Espectro Filtrado')

        # Añadir líneas de referencia para los filtros
        plt.axvline(low_freq, color="#288603FF", linestyle='--', alpha=0.7, label=f'LPF {low_freq}Hz')
        plt.axvline(high_freq, color='#288603FF', linestyle='--', alpha=0.7, label=f'HPF {high_freq}Hz')
        plt.axvline(notch_freq, color="#D400FFEB", linestyle='--', alpha=0.7, label=f'Notch {notch_freq}Hz')

        plt.title('Densidad Espectral de Potencia')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('PSD (dB)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def fft(self, pick_channel:list[int]|int=0, plot:bool=True, low_freq:float=None, high_freq:float=None):
        """
        Calcula la Transformada Rápida de Fourier (FFT) usando el método de Welch y visualiza el espectro de potencia.

        Estima la densidad espectral de potencia (PSD) de uno o varios canales seleccionados (por índice o nombre).
        Permite visualizar el espectro en un rango de frecuencias específico, convirtiendo los resultados a escala
        logarítmica (dB) y almacenándolos en los atributos de la instancia.

        Args:
            pick_channel (int | str | list[str], opcional): Canal(es) a analizar.
                - Si es `int`: Índice del canal.
                - Si es `str`: Nombre del canal (ej. 'Cz').
                - Si es `list[str]`: Lista de nombres de canales (ej. ['Fp1', 'Fp2']).
                Por defecto 0.
            plot (bool, opcional): Si es True, genera un gráfico del espectro de potencia. Por defecto True.
            low_freq (float, opcional): Frecuencia mínima (Hz) para el gráfico y análisis.
                                        Si es None, se usa 0 Hz. Por defecto None.
            high_freq (float, opcional): Frecuencia máxima (Hz) para el gráfico y análisis.
                                         Si es None, se usa la frecuencia de Nyquist (sfreq / 2). Por defecto None.

        Raises:
            ValueError: Si se proporciona un nombre de canal que no existe en `self.info.ch_names`.
            IndexError: Si el índice proporcionado está fuera del rango.

        Returns:
            None: Actualiza `self.fft_psd` y `self.fft_freq`. Si `plot=True`, muestra el gráfico.

        Examples:
            >>> # Por índice
            >>> signal.fft(pick_channel=0)
            >>> # Por nombre
            >>> signal.fft(pick_channel='Cz')
            >>> # Lista de nombres
            >>> signal.fft(pick_channel=['Fp1', 'Fp2'], low_freq=1.0, high_freq=30.0)
        """
        from scipy.signal import welch
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        data = np.atleast_2d(self.data)

        # --- Selección de canales ---
        if isinstance(pick_channel, int):
                    ch_idx = [pick_channel]

        elif isinstance(pick_channel, list):
            ch_idx = []
            for ch in pick_channel:
                if ch in self.info.ch_names:
                    ch_idx.append(self.info.ch_names.index(ch))
                else:
                    raise ValueError(f"El canal '{ch}' no existe en la señal.")
        
        elif isinstance(pick_channel, str):
            if pick_channel in self.info.ch_names:
                ch_idx = [self.info.ch_names.index(pick_channel)]
            else:
                raise ValueError(f"El canal '{pick_channel}' no existe en la señal.")
        
        else:
            ch_idx = [0] # Por defecto, primer canal

        data_ch = data[ch_idx, :]
        freqs, psd = welch(data_ch, fs=self.sfreq, nperseg=1024, axis=1)

        self.fft_psd = 10 * np.log10(psd + 1e-12)
        self.fft_freq = freqs

        if plot:
            # Configuración de colores
            if len(ch_idx) <= 8:
                distinct_colors = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
                ]
                colors = distinct_colors[:len(ch_idx)]
            else:
                colormap = cm.get_cmap('tab20', len(ch_idx))
                colors = [mcolors.to_hex(colormap(i)) for i in range(len(ch_idx))]

            if low_freq is not None and high_freq is not None:
                band_mask = (self.fft_freq >= low_freq) & (self.fft_freq <= high_freq)
            else:
                low_freq = 0
                high_freq = self.sfreq / 2
                band_mask = (self.fft_freq >= low_freq) & (self.fft_freq <= high_freq)

            fig, ax = plt.subplots(figsize=(10, 5))

            for i, channel_idx in enumerate(ch_idx):
                channel_name = self.info.ch_names[channel_idx]
                y_data = self.fft_psd[i, band_mask]

                from scipy.ndimage import gaussian_filter1d
                y_smooth = gaussian_filter1d(y_data, sigma=0.8)

                ax.plot(self.fft_freq[band_mask], y_smooth, color=colors[i], label=f'Canal {channel_name}', linewidth=2)

            if self.is_filtered:
                title = f'Espectro de Potencia de Señal Filtrada - Banda ({low_freq}-{high_freq} Hz)'
            else:
                title = f'Espectro de Potencia de Señal Cruda - Banda ({low_freq}-{high_freq} Hz)'
        
            ax.set_title(title)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Densidad Espectral de Potencia (dB)')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    def __getitem__(self, idx): 
        """
        Permite acceso tipo obj[canal, muestras].

        Args:
            idx : int, str, list o tuple
                - int/str/list: picks de canales.
                - slice o int en segunda posición: muestras.

        Returns:
            np.ndarray:
                Segmento solicitado. Si `idx` no corresponde a un canal/muestras válidos
                (p.ej. tipo no soportado), devuelve un array vacío de forma (0, 0).

        Raises:
            IndexError:
                Si `idx` es tupla y `len(idx) != 2`.
            ValueError:
                Si `picks` no es `int`, `str` ni `list`.
        """
        # Si es un solo índice, devuelvo todo el canal completo
        if not isinstance(idx, tuple): # obj[x]
            picks = idx
            muestras = None

        else: # obj[x, y]
            # idx = (picks, slice) o (picks, int)
            if len(idx) != 2:
                raise IndexError("Se debe indexar como [canal, muestras]")
            
            picks, muestras = idx
        
        if not isinstance(picks, (str, list, int)):
            raise ValueError(f"Al indexar, canales debe ser un 'int', 'str' o 'list' y fue dado: {type(picks)}")
    
        if muestras is not None:
            data = self.get_data(picks=picks)
            if data.ndim == 1:
                return data[muestras]
            else:
                return data[:, muestras]

        elif isinstance(picks, (str, list, int)):
            return self.get_data(picks=idx)
        
        else:
            return np.empty(shape=(0,0))

    def _getInfo(self):
        """
        Genera información estadística sobre los datos (uso interno).

        Returns:
            dict: Diccionario con estadísticas descriptivas:
                - name: Nombres de canales
                - type(s): Tipos únicos de canales
                - min: Valor mínimo global
                - Q1: Primer cuartil global
                - mediana: Mediana global
                - Q3: Tercer cuartil global
                - max: Valor máximo global

        Notes:
            - Método de uso interno, no diseñado para uso público.
            - Estadísticas calculadas sobre toda la señal (todos los canales y muestras).
        """
        dic = {"name":self.info.ch_names,
               "type(s)":list(set(self.info.ch_types)),
               "min":self.data.min(),
               "Q1": np.percentile(self.data, q=25),
               "mediana": np.median(self.data),
               "Q3":np.percentile(self.data, q=75),
               "max": self.data.max()}
        
        return dic
