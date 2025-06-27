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
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras) con datos de señal.
            sfreq : float, optional
                Frecuencia de muestreo en Hz. Si es None, se usa info.sfreq.
            info : Info, optional
                Metadatos de canales (nombres, tipos, etc.).
            anotaciones : Annotations, optional
                Eventos temporales asociados a la señal.
            first_samp : int, optional
                Índice de la primera muestra respecto al registro original (default=0).
            see_log : bool, optional
                Activa/desactiva mensajes de logging (default=True).

        Implementation Notes:
            - Si sfreq es None, se obtiene de info.sfreq
            - Configura el sistema de logging según see_log
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
            1. Recorte temporal (si start/stop)
            2. Selección de canales (si picks)
            3. Filtrado por amplitud (si reject)

        Args:
            picks: Canal(es) a seleccionar. Puede ser:
                - str: nombre de un canal
                - int: índice de un canal
                - list: múltiples canales (nombres o índices)
                - None: todos los canales (default)
            start: Tiempo de inicio en segundos (None = inicio de la señal).
            stop: Tiempo de fin en segundos (None = fin de la señal).
            reject: Umbral de amplitud pico a pico (en μV) para descartar canales ruidosos.
            times: Si True, retorna también el vector de tiempos.

        Returns:
            data: Array con forma (n_canales, n_muestras) o (n_muestras,) si un solo canal.
            time_vector (opcional): Vector 1D de tiempos en segundos (solo si times=True).

        Raises:
            ValueError: Si los parámetros start/stop son inválidos o reject < 0.
            TypeError: Si picks tiene un tipo no soportado.

        Usage Examples:
            >>> # Extraer canales FP1-FP2 entre 10-20s con umbral 150μV
            >>> datos = raw.get_data(
            >>>     picks=['FP1','FP2'],
            >>>     start=10,
            >>>     stop=20,
            >>>     reject=150
            >>> )

        Notes:
            - El filtrado por amplitud se aplica después de la selección de canales.
            - El vector de tiempos incluye el offset de first_samp.
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
        else:
            data = data # Si no se solicita filtro, genero la variable data con toda la información

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
            - Recorta datos entre tmin y tmax
            - Ajusta anotaciones al nuevo intervalo
            - Actualiza first_samp

        Args:
            tmin: Tiempo de inicio en segundos (inclusive). None = 0.
            tmax: Tiempo de fin en segundos (exclusive). None = fin de la señal.

        Returns:
            RawSignal: Nueva instancia con los datos recortados.

        Edge Cases:
            - Si tmin > tmax: ValueError
            - Si tmax excede duración: se ajusta al final
            - Anotaciones fuera del rango: eliminadas

        Example:
            >>> recortada = raw.crop(tmin=5, tmax=15)  # 10 segundos
            >>> print(recortada.data.shape)
            (n_canales, 5120)  # 10s * 512Hz

        Notes:
            - Las anotaciones se ajustan restando tmin para mantener referencia temporal.
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

    def filter(self, low_freq:float = 1, high_freq:float = 25, order:int = 4, 
               notch_freq:float = 50.0, q:int = 30) -> RawSignal:
        """
        Aplica filtrado pasa-banda y notch a la señal EEG.

        Filtrado:
            1. Filtro notch para eliminar interferencia de línea eléctrica
            2. Filtro pasa-banda Butterworth

        Args:
            low_freq: Frecuencia de corte inferior (Hz) para filtro pasa-banda.
            high_freq: Frecuencia de corte superior (Hz) para filtro pasa-banda.
            order: Orden del filtro Butterworth (default=4).
            notch_freq: Frecuencia central del filtro notch (default=50 Hz).
            q: Factor de calidad del filtro notch (default=30).

        Returns:
            RawSignal: Nueva instancia con la señal filtrada.

        Raises:
            ValueError: Si low_freq < 0 o low_freq >= high_freq.

        Notes:
            - Utiliza filtrado zero-phase (filtfilt) para evitar distorsión de fase.
            - Conserva metadatos y anotaciones del objeto original.
        """
        if low_freq < 0 or low_freq >= high_freq:
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
        new_info._select(picks)

        return RawSignal(data=channels, sfreq=self.sfreq, info=new_info, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def set_anotaciones(self, anotaciones:Annotations):
        """
        Establece las anotaciones para la señal.

        Args:
            anotaciones: Objeto Annotations con las nuevas anotaciones.

        Raises:
            TypeError: Si anotaciones no es instancia de Annotations.
            ValueError: Si los onset están fuera del rango de la señal.

        Notes:
            - Valida que los onset estén dentro de la duración de la señal.
            - Actualiza la referencia interna a las anotaciones.
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
    
    def plot(self, picks=None, start=None, duration=None, show_anotaciones: bool = True):
        """
        Visualización interactiva de señales usando PyQtGraph.

        Características:
            - Desplazamiento vertical entre canales
            - Visualización de anotaciones como líneas verticales
            - Escalado automático por tipo de señal
            - Interfaz con scroll para muchas señales

        Args:
            picks: Canales a visualizar (None = todos).
            start: Tiempo inicial en segundos (None = 0).
            duration: Duración a mostrar en segundos (None = señal completa).
            show_anotaciones: Mostrar marcadores de eventos (default=True).

        Implementation Details:
            - Usa PyQt5 para la interfaz gráfica
            - Autoajusta límites Y según tipo de señal (EEG, ECG, etc.)
            - Leyenda interactiva para anotaciones
            - Diseño responsivo adaptable al número de canales
        """
        import pyqtgraph as pg
        import sys
        import numpy as np
        from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                                    QScrollArea, QVBoxLayout, QWidget, QDesktopWidget)
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QLabel, QHBoxLayout
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
        BASE_HEIGHT_PER_CHANNEL = 220  # Asigno la altura base con la que se muestra cada canal
        MIN_WINDOW_HEIGHT = 400        # Altura mínima que tiene la ventana que abre Qt
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
        MIN_CHANNEL_HEIGHT = 220 # Altura mínima del gráfico para cada canal
        MAX_CHANNEL_HEIGHT = 400 # Altura máxima del gráfico para cada canal
        
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
        def get_ylimits(signal, ch_type):
            """
            Calcula límites Y adaptativos basados en tipo de señal y percentiles
            
            Args:
                signal: Array 1D con datos del canal
                ch_type: Tipo de señal (eeg, ecg, emg, etc.)
            
            Returns:
                tuple: (y_min, y_max)
            """
            # Rangos típicos para diferentes tipos de señales
            type_ranges = {
                'eeg': (-50, 50),   # μV
                'ecg': (-150, 350),     # mV
                'emg': (-150, 100),     # mV
                'eog': (-500, 500), # μV
            }
            
            # Uso rango predefinido si el tipo es conocido
            if ch_type in type_ranges:
                return type_ranges[ch_type]
            
            # Para tipos desconocidos: uso percentiles 1 y 99 con padding
            p1 = np.percentile(signal, 1)
            p99 = np.percentile(signal, 99)
            padding = 0.3 * (p99 - p1)  # 30% de padding
            
            # Manejar caso de señal constante
            if padding == 0:
                padding = 1 if signal[0] == 0 else abs(signal[0]) * 0.5
                
            return (p1 - padding, p99 + padding)

        # 6. Crear gráfico para cada canal con escalado inteligente
        # Itero sobre cada canal
        for idx in range(n_chan):
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
            channel_widget.setLabel('left', self.info.ch_names[idx], 
                                **{'color': 'k', 'font-size': '12pt',
                                   'font-family':'Times New Roman'}) # Configuro el nombre del eje de ese canal y su formato
            
            # Configurar límites Y adaptativos (uso función get_ylimits())
            y_min, y_max = get_ylimits(data[idx, :], ch_type)
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
    
    def plot_filtered(self, filtered_signal):
        """
        Grafica comparación entre señal original y filtrada.

        Args:
            filtered_signal: Array con datos de señal filtrada.

        Notes:
            - Muestra primer canal de ambas señales.
            - Utilizado para diagnóstico interno de filtrado.
        """
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

    def plot_spectrum(self, filtered_signal, low_freq, high_freq, notch_freq, ch_idx):
        """
        [Método interno] Grafica comparación de espectros antes/después de filtrar.

        Args:
            filtered_signal: Array con datos de señal filtrada.
            low_freq: Frecuencia de corte inferior usada en filtrado.
            high_freq: Frecuencia de corte superior usada en filtrado.
            notch_freq: Frecuencia de notch usada en filtrado.
            ch_idx: Índice del canal que se quiere mostrar.

        Notes:
            - Convierte PSD a escala dB (referencia: 1 μV²/Hz)
            - Marca frecuencias de corte con líneas verticales
            - Muestra solo el primer canal
        """
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

    def __getitem__(self, idx): # [canal, muestras], si no hay devuelvo array vacío
        """
        Permite el acceso por índices a los datos de la señal (no implementado actualmente).

        Returns:
            np.ndarray: Segmento solicitado de los datos.

        Notes:
            - Este método está pendiente de implementación.
        """
        # Si es un solo índice, devuelvo todo el canal completo
        if not isinstance(idx, tuple):
            picks = idx
            muestras = None
        else:
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
