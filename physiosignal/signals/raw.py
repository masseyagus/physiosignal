from __future__ import annotations

from physiosignal.info import Info
from physiosignal.info import Annotations
from physiosignal.logger import log_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import logging

class RawSignal:
    """
    Representa una señal de datos crudos (por ejemplo, EEG) junto con su
    información de muestreo, metadatos de canales y anotaciones de eventos.

    Key Features:
        - Soporte para operaciones temporales (crop, segmentación)
        - Manejo integrado de anotaciones
        - Selección flexible de canales
        - Filtrado por amplitud

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
    """
    def __init__(self, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True):
        """
        Inicializa una instancia de RawSignal.

        Parameters:
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras)
            sfreq : float, optional
                Frecuencia de muestreo (obtenida de info si es None)
            info : Info, optional
                Metadatos de canales
            anotaciones : Annotations, optional
                Eventos temporales asociados
            first_samp : int, optional
                Índice de primera muestra (default=0)
            see_log : bool, optional
                Activa/desactiva logs (default=True)

        Implementation Notes:
            - Si sfreq es None, se obtiene de info.sfreq
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

        Processing Pipeline:
            1. Filtrado por amplitud (si reject)
            2. Recorte temporal (si start/stop)
            3. Selección de canales (si picks)

        Args:
            picks: Canal(es) a seleccionar (nombre, índice o lista de ellos). Si None, se usan todos.
            start: Tiempo de inicio en segundos (None para iniciar desde el comienzo).
            stop: Tiempo de fin en segundos (None para llegar hasta el final).
            reject: Umbral de amplitud pico a pico para descartar canales (None para desactivar).
            times: Si es True, retorna también el vector de tiempos correspondiente.

        Returns:
            data: Array con forma (n_canales, n_muestras) o (n_muestras,) si un solo canal.
            time_vector (opcional): Vector 1D de tiempos en segundos si times=True.

        Usage Examples:
            >>> # Extraer canales FP1-FP2 entre 10-20s con umbral 150μV
            >>> datos = raw.get_data(
            >>>     picks=['FP1','FP2'],
            >>>     start=10,
            >>>     stop=20,
            >>>     reject=150
            >>> )

        Notes:
            - Si se usa `reject`, se descartan canales cuya amplitud (máximo - mínimo) supere el umbral dado.
            - La recortación por `start` y `stop` se aplica antes de la selección de canales.
            - El vector de tiempos parte desde `start` si se especifica, o desde 0 por defecto.
            - El retorno será una tupla (data, times) solo si se solicita explícitamente.
        """
        duration = self.data.shape[1]/self.sfreq # Hallo la duración de la señal

        start = 0.0 if start is None else float(start)
        stop = duration if stop is None else float(stop) 

        # Chequeo tiempos y genero las muestras en caso dado
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

        # Genero vector de tiempos en 1D  
        if times:
            muestras = self.data.shape[1]
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

    def crop(self, tmin:int=None, tmax:int=None) -> RawSignal:
        """
        Recorta la señal en un intervalo de tiempo especificado.

        Temporal Processing:
            - Recorta señal y ajusta anotaciones al nuevo intervalo
            - Normaliza tiempos relativos al nuevo inicio

        Args:
            tmin: Tiempo de inicio en segundos (incluido). Si None, usa 0.
            tmax: Tiempo de fin en segundos (excluido). Si None, usa el final.

        Returns:
            RawSignal: Nueva instancia con los datos recortados entre tmin y tmax.

        Edge Cases:
            - Si tmin > tmax: ValueError
            - Si tmax excede duración: ajusta al final
            - Anotaciones fuera del rango: eliminadas

        Example:
            >>> recortada = raw.crop(tmin=5, tmax=15)  # 10 segundos
            >>> print(recortada.data.shape)
            (n_canales, 5120)  # 10s * 512Hz

        Notes:
            - Las anotaciones se filtran para incluir solo aquellas cuyo onset
              esté dentro del intervalo [tmin, tmax].
            - Los onset de las anotaciones se ajustan restando tmin para que sean
              relativos al nuevo inicio del segmento recortado.
            - Las anotaciones que comienzan antes de tmin o después de tmax se descartan.
        """
        crop_data = self.get_data(start=tmin, stop=tmax)

        duration = self.data.shape[1]/self.sfreq
        tmin = 0.0 if tmin is None else float(tmin)
        tmax = duration if tmax is None else float(tmax)        
        new_first_samp = self.first_samp + int(tmin * self.sfreq)

        # Actualizo anotaciones
        df = self.anotaciones.get_annotations()

        if df is not None and not df.empty:
            mask = (df['onset'] >=tmin) & (df["onset"] <= tmax)
            filtered = df[mask].copy()

            filtered.loc[:, 'onset'] = filtered['onset'] - tmin

        else:
            filtered = df

        onset = filtered["onset"]
        duration = filtered['duration']
        description = filtered['description']
        ch_names = filtered['ch_names']

        new_ann = Annotations(onset=onset, duration=duration, description=description, ch_names=ch_names)

        return RawSignal(data=crop_data, sfreq=self.sfreq, info=self.info, anotaciones=new_ann, first_samp=new_first_samp)

    def describe(self, channels:str|list):
        """
        Genera estadísticas descriptivas para los canales especificados.

        Args:
            channels: Nombre(s) de canal(es) para analizar.

        Returns:
            pd.DataFrame: DataFrame con estadísticas descriptivas (min, Q1, mediana, Q3, max)
            para cada canal especificado.

        Example Output:
                   FP1       FP2
            min    -125.32   -118.45
            Q1      -15.21    -12.33
            mediana   0.05      0.12
            Q3       18.76     15.89
            max     105.89     92.17

        Notes:
            - Las estadísticas se calculan sobre toda la duración de la señal.
            - Para un solo canal, se devuelve un DataFrame con una columna.
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

    def filter(self, low_freq:float = 1, high_freq:float = 25, order:int = 4, 
               notch_freq:float = 50.0, q:int = 30) -> RawSignal:
        """
        Aplica filtrado a la señal (no implementado actualmente).

        Args:
            low_freq: Frecuencia de corte baja.
            high_freq: Frecuencia de corte alta.
            notch_freq: Frecuencia de filtro notch.
            order: Orden del filtro.

        Returns:
            RawSignal: Nueva instancia con la señal filtrada.

        Notes:
            - Este método está pendiente de implementación.
        """
        if low_freq < 1 or low_freq >= high_freq:
            raise ValueError(f"low_freq debe ser mayor o igual a 1, y menor a high_freq")

        # Vector de tiempo para gráficas
        n_samps = self.data.shape[1]
        t = np.arange(n_samps) / self.sfreq

        # Obtengo los coeficientes del filtro notch
        b_notch, a_notch = scipy.signal.iirnotch(notch_freq, q, self.sfreq)

        # Aplico el filtro notch a la señal
        notch_signal = scipy.signal.filtfilt(b_notch, a_notch, self.data, axis=-1)

        # Design the bandpass filter
        b_butter, a_butter = scipy.signal.butter(order, [low_freq, high_freq], btype='band', fs=self.sfreq)

        # Aplico el filtro Butterworth a la señal filtrada
        filtered_signal = scipy.signal.filtfilt(b_butter, a_butter, notch_signal, axis=-1)

        # Devuelvo nuevo objeto con señal filtrada
        return RawSignal(data=filtered_signal, sfreq=self.sfreq, info=self.info, anotaciones=self.anotaciones, first_samp=self.first_samp)

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

        Notes:
            - Actualiza la información de los canales en el objeto info.
        """
        from copy import deepcopy
        # Obtengo la info de los canales (n_canales, n_muestras)
        channels = self.get_data(picks=picks)

        # Actualizo los canales 
        new_info = deepcopy(self.info)
        new_info._select(picks)

        return RawSignal(data=channels, sfreq=self.sfreq, info=new_info, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def set_anotaciones(self, anotaciones):
        """
        Establece las anotaciones para la señal (no implementado actualmente).

        Args:
            anotaciones: Objeto Annotations con las nuevas anotaciones.

        Notes:
            - Este método está pendiente de implementación.
        """
        pass

    def plot(self, picks, start, duration, show_anotaciones):
        """
        Genera una visualización de la señal (no implementado actualmente).

        Args:
            picks: Canales a visualizar.
            start: Tiempo de inicio para el plot.
            duration: Duración del segmento a visualizar.
            show_anotaciones: Si es True, muestra las anotaciones en el plot.

        Notes:
            - Este método está pendiente de implementación.
        """
        pass  
    
    def plot_filtered(self, filtered_signal):

        n_samps = self.data.shape[1]
        t = np.arange(n_samps) / self.sfreq

        # Graficamos la señal original y la señal filtrada
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, self.data[0,:], label='Señal Original', color='blue')
        plt.title('Señal Original')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.ylim(-200, 200)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, filtered_signal[0,:], label='Señal Filtrada', color='red')
        plt.title('Señal Filtrada')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.tight_layout()
        plt.ylim(-200, 200)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_spectrum(self, filtered_signal, low_freq, high_freq, notch_freq):

        f_orig, psd_orig = scipy.signal.welch(self.data, fs=self.sfreq, nperseg=1024, axis=-1)
        f_filt, psd_filt = scipy.signal.welch(filtered_signal, fs=self.sfreq, nperseg=1024, axis=-1)

        # Convertir a dB (con referencia estándar de 1 μV²/Hz)
        psd_orig_db = np.log10(psd_orig)  # dB re: 1 μV²/Hz
        psd_filt_db = np.log10(psd_filt)  # dB re: 1 μV²/Hz

        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(f_orig, psd_orig_db[0,:], 'b', label='Espectro Original')
        plt.plot(f_filt, psd_filt_db[0,:], 'r', label='Espectro Filtrado')

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

    def __getitem__(self): # [canal, muestras], si no hay devuelvo array vacío
        """
        Permite el acceso por índices a los datos de la señal (no implementado actualmente).

        Returns:
            np.ndarray: Segmento solicitado de los datos.

        Notes:
            - Este método está pendiente de implementación.
        """
        pass

    def _getInfo(self):
        """
        Genera información estadística sobre los datos (uso interno).

        Returns:
            dict: Diccionario con estadísticas descriptivas de la señal.

        Notes:
            - Método de uso interno, no diseñado para ser llamado directamente.
            - Las estadísticas se calculan sobre todos los canales y muestras.
        """
        dic = {"name":self.info.ch_names,
               "type(s)":list(set(self.info.ch_types)),
               "min":self.data.min(),
               "Q1": np.percentile(self.data, q=25),
               "mediana": np.median(self.data),
               "Q3":np.percentile(self.data, q=75),
               "max": self.data.max()}
        
        return dic



    