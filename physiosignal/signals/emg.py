from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import matplotlib.pyplot as plt
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
    
    def __init__(self, raw:RawSignal=None, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True, activation_threshold:float=None, activation_times:np.ndarray|list=None,
                 is_filtered:bool=False):
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
        from copy import deepcopy
        if raw is not None:
            super().__init__(deepcopy(raw.data),
                            deepcopy(raw.sfreq),
                            deepcopy(raw.info),
                            deepcopy(raw.anotaciones),
                            deepcopy(raw.first_samp),
                            see_log)
            self.is_filtered = deepcopy(raw.is_filtered)
        else:
            super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)
            self.is_filtered = is_filtered  # Por defecto, la señal se asume no filtrada

        self.activation_threshold = activation_threshold
        self.activation_times = activation_times

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo

    def variation(self):
        """
        Función para el análisis de la variación temporal del EMG.
        """
        pass

    def tkeo_envelope(self, traditional:bool=False, channel:int|str=0, tmin:int=0, tmax:int=10):
        """
        Calcula y visualiza la envolvente de la señal EMG utilizando el operador de energía Teager-Kaiser (TKEO).

        Aplica el operador TKEO para estimar la energía instantánea de la señal, rectifica el resultado y
        suaviza mediante un filtro paso bajo Butterworth de segundo orden. Genera una gráfica comparativa
        con doble eje Y para contrastar la amplitud original con la energía calculada, permitiendo visualizar
        mejor los cambios bruscos de actividad (onsets).

        Args:
            traditional (bool): Si es True, calcula y grafica también la envolvente tradicional (rectificación
                                + filtro paso bajo) para comparación. Por defecto False.
            channel (int, str): Índice o nombre del canal a analizar. Por defecto 0.
            tmin (int): Tiempo inicial del segmento en segundos. Por defecto 0.
            tmax (int): Tiempo final del segmento en segundos. Por defecto 10.

        Raises:
            ValueError: Si el canal no existe, si los tiempos de ventana son inconsistentes (`tmin >= tmax`)
                        o si la ventana solicitada no contiene datos.

        Returns:
            None: Muestra una gráfica comparativa (Amplitud vs Energía) mediante Matplotlib.
        """
        import scipy.signal

        if len(self.data.shape) > 1:
            if channel >= self.data.shape[0]:
                raise ValueError("El canal especificado excede el número de canales en la señal EMG.")
            
            if isinstance(channel, str):
                if channel in self.info.ch_names:
                    channel = self.info.ch_names.index(channel)
                else:
                    raise ValueError(f"El canal '{channel}' no existe en la señal EMG.")
                
            signal_1d = np.asarray(self.data[channel]).ravel()

        else:
            if self.data.ndim > 1:
                logging.info("La señal EMG tiene múltiples canales, se usará el canal 0 por defecto.")
        
        tmin_samps_global = int(tmin * self.sfreq)
        tmax_samps_global = int(tmax * self.sfreq)

        if tmin_samps_global >= tmax_samps_global:
            raise ValueError("tmin debe ser menor que tmax y dentro del rango de la señal.")
        
        tmin_local = tmin_samps_global - int(self.first_samp)
        tmax_local = tmax_samps_global - int(self.first_samp)

        n_local = signal_1d.shape[0]
        slice_start = max(0, tmin_local)
        slice_end = min(n_local, tmax_local)

        if slice_start >= slice_end:
            raise ValueError("La ventana solicitada no intersecta la señal almacenada en esta instancia (revisar first_samp).")  

        signal_1d = signal_1d[slice_start:slice_end]
        tkeo = signal_1d.copy()

        # Teager–Kaiser Energy operator 
        tkeo[1:-1] = signal_1d[1:-1]**2 - signal_1d[0:-2]*signal_1d[2:]

        # Corrección de bordes
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]

        # Rectificado de TKEO
        tkeo_rectified = np.abs(tkeo)

        cut_off = 8 # Frecuencia de corte de la envolvente
        sos_env = scipy.signal.butter(2, cut_off, btype='low', fs=self.sfreq, output='sos')

        # Filtro en la señal TKEO rectificada
        envelope_tkeo = scipy.signal.sosfiltfilt(sos_env, tkeo_rectified)

        start_time_segundos = (slice_start + self.first_samp) / self.sfreq
        t = (np.arange(len(signal_1d)) / self.sfreq) + start_time_segundos

        fig, ax = plt.subplots(figsize=(12, 6))

        # --- Eje Izquierdo (Amplitud: Original + Tradicional) ---
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Amplitud / Energía [µV]')
        ax.tick_params(axis='y')

        ln1 = ax.plot(t, signal_1d, label='Señal EMG Original', color="#4F4E4E", alpha=0.5, zorder=2)

        if traditional:
            # Comparación con el método tradicional
            sos_raw = scipy.signal.butter(2, cut_off, btype='low', output='sos', fs=self.sfreq)
            envelope_traditional = scipy.signal.sosfiltfilt(sos_raw, np.abs(signal_1d))
            ln2 = ax.plot(t, envelope_traditional, label='Envolvente Tradicional', color="#0000FF", linestyle='--', alpha=0.6, zorder=2)
       
        # --- Eje Derecho (Energía: TKEO + Env TKEO) ---
        ax2 = ax.twinx()
        ax2.set_ylabel('Energía TKEO [µV²]')
        ax2.tick_params(axis='y')

        ln3 = ax2.plot(t, tkeo_rectified, label='TKEO (Rectificado)', color="#FE6600", alpha=0.6, zorder=3)

        ln4 = ax2.plot(t, envelope_tkeo, label='Envolvente Final (TKEO + LowPass)', color="#3BE966", linewidth=2, zorder=4)

        lines = ln1 + (ln2 if traditional else []) + ln3 + ln4
        labels = [l.get_label() for l in lines]

        ax.legend(lines, labels, loc='upper right')
        ax.grid(True, alpha=0.6)
        plt.title('Comparativa: EMG Original vs Envolvente Tradicional vs TKEO')
        plt.show()

    def hilbert(self, tmin:int=0, tmax:int=10, channels:str|int|list[str|int]=0, 
                low_freq:float=0.5, high_freq:float=150):
        """
        Calcula y visualiza la envolvente de la señal EMG usando la Transformada de Hilbert.

        Aplica un filtro paso banda (Butterworth de 4to orden) a la señal para aislar las frecuencias
        musculares relevantes y luego calcula la señal analítica mediante la Transformada de Hilbert.
        La magnitud de esta señal analítica (la envolvente) representa la amplitud instantánea, útil para
        estimar la fuerza muscular manteniendo las unidades originales (µV).

        Args:
            tmin (float, opcional): Tiempo de inicio del análisis en segundos. Por defecto 0.
            tmax (float, opcional): Tiempo de fin del análisis en segundos. Por defecto 10.
            channels (int | str | list[int|str], opcional): Canal(es) a analizar. Puede ser un índice,
                                                             un nombre o una lista de ellos. Por defecto 0.
            low_freq (float, opcional): Frecuencia de corte inferior para el filtro paso banda (Hz).
                                        Por defecto 0.5.
            high_freq (float, opcional): Frecuencia de corte superior para el filtro paso banda (Hz).
                                         Por defecto 150.

        Raises:
            ValueError: Si el canal no existe, si los tiempos son inválidos (`tmin >= tmax`), si la ventana
                        no intersecta con los datos o si los parámetros de frecuencia violan Nyquist.

        Returns:
            tuple:
                - hilbert_signal (np.ndarray): Señal analítica compleja (parte real es la señal filtrada).
                - envelope (np.ndarray): Envolvente de amplitud (magnitud de la señal analítica).
                Ambos arrays tienen la forma (n_canales_seleccionados, n_muestras_recortadas).

        Notes:
            - El filtro aplicado es un Butterworth pasa-banda de 4to orden, sin fase (`filtfilt`).
            - Si `high_freq` supera la frecuencia de Nyquist, se ajusta automáticamente a 0.99 * Nyquist.
            - Genera un gráfico por cada canal seleccionado comparando la señal filtrada y su envolvente.

        Examples:
            >>> # Envolvente del canal 0 entre 5s y 15s
            >>> analytic, env = emg.hilbert(tmin=5, tmax=15, channels=0)
            >>>
            >>> # Envolvente de un canal específico con filtro personalizado
            >>> _, env = emg.hilbert(channels='EMG1', low_freq=10, high_freq=200)
        """
        from scipy.signal import hilbert, butter, filtfilt

        # ------ Normalización de Canales ------
        ch_idx = []
        ch_labels = []

        if isinstance(channels, (int, str)):
            input_list = [channels]
        else:
            input_list = channels

        for ch in input_list:

            if isinstance(ch, int):
                if ch < 0 or ch >= self.data.shape[0]:
                    raise ValueError(f"El índice de canal {ch} está fuera de rango.")
                
                ch_idx.append(ch)
                ch_labels.append(self.info.ch_names[ch])
            
            elif isinstance(ch, str):
                if ch in self.info.ch_names:
                    idx = self.info.ch_names.index(ch)
                    ch_idx.append(idx)
                    ch_labels.append(ch)

                else:
                    raise ValueError(f"El canal '{ch}' no existe en la señal EMG.")
            
            else:
                 raise ValueError("Los canales deben ser int o str.")

        # ------ Tiempos ------
        tmin_samps_global = int(tmin * self.sfreq)
        tmax_samps_global = int(tmax * self.sfreq)

        if tmin_samps_global >= tmax_samps_global:
            raise ValueError("tmin debe ser menor que tmax.")

        tmin_local = tmin_samps_global - int(self.first_samp)
        tmax_local = tmax_samps_global - int(self.first_samp)

        n_local = self.data.shape[1] if self.data.ndim > 1 else self.data.shape[0]
        
        slice_start = max(0, tmin_local)
        slice_end = min(n_local, tmax_local)

        if slice_start >= slice_end:
            raise ValueError("La ventana solicitada no intersecta la señal almacenada.")  

        data = self.data[ch_idx, slice_start:slice_end]
        data = np.atleast_2d(data)

        nyquist = self.sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Validar que high < 1.0 (Nyquist)
        if high >= 1.0:
            high = 0.99 
            print("Warning: high_freq ajustada al límite de Nyquist.")

        b_butter, a_butter = butter(4, [low, high], btype='band')
        data_filtered = filtfilt(b_butter, a_butter, data, axis=1)

        # Transformada de Hilbert
        hilbert_signal = hilbert(data_filtered, axis=1)
        envelope = np.abs(hilbert_signal)

        n_segment = data.shape[1]
        start_time_abs = (slice_start + self.first_samp) / self.sfreq
        t = (np.arange(n_segment) / self.sfreq) + start_time_abs

        for i, idx_ch in enumerate(ch_idx):
            plt.figure(figsize=(12, 6))

            # Grafico la señal filtrada
            plt.plot(t, np.real(hilbert_signal[i]), label='Señal Filtrada (Bandpass)', color="#002FFF", alpha=0.6)

            # Grafico la envolvente
            plt.plot(t, envelope[i], label='Envolvente (Hilbert)', color="#FF7300", linewidth=2)
            
            plt.title(f"Transformada de Hilbert - Canal: {ch_labels[i]}")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Amplitud [µV]")
            plt.legend()
            plt.grid(True, alpha=0.4, color='#000000')
            plt.tight_layout()
            plt.show()

        return hilbert_signal, envelope

    def segment(self, umbral:float, channel:int=0, tmin:int=0, tmax:int=10):
        """
        Segmenta y visualiza períodos de activación muscular basándose en un umbral de amplitud.

        Analiza un segmento específico de la señal y destaca visualmente las regiones donde la amplitud
        supera el umbral definido. Utiliza `matplotlib.pyplot.fill_between` para generar un sombreado
        continuo sobre los intervalos de activación activa, facilitando la identificación visual de
        contracciones musculares.

        Args:
            umbral (float): Valor de amplitud (µV) mínimo para considerar una activación muscular.
            channel (int): Índice del canal a analizar. Por defecto 0.
            tmin (int): Tiempo inicial del segmento en segundos. Por defecto 0.
            tmax (int): Tiempo final del segmento en segundos. Por defecto 10.

        Raises:
            ValueError: Si el canal especificado no es válido o si el intervalo temporal es incorrecto.

        Returns:
            None: Muestra la gráfica con las zonas de activación sombreadas mediante Matplotlib.
        """
        if len(self.data.shape) > 1:
            if channel >= self.data.shape[0]:
                raise ValueError("El canal especificado excede el número de canales en la señal EMG.")
            
            if isinstance(channel, str):
                if channel in self.info.ch_names:
                    channel = self.info.ch_names.index(channel)
                else:
                    raise ValueError(f"El canal '{channel}' no existe en la señal EMG.")
                
            signal_1d = np.asarray(self.data[channel]).ravel()

        else:
            if self.data.ndim > 1:
                logging.info("La señal EMG tiene múltiples canales, se usará el canal 0 por defecto.")

        tmin_samps_global = int(tmin * self.sfreq)
        tmax_samps_global = int(tmax * self.sfreq)

        if tmin_samps_global >= tmax_samps_global:
            raise ValueError("tmin debe ser menor que tmax y dentro del rango de la señal.")
        
        tmin_local = tmin_samps_global - int(self.first_samp)
        tmax_local = tmax_samps_global - int(self.first_samp)

        n_local = signal_1d.shape[0]
        slice_start = max(0, tmin_local)
        slice_end = min(n_local, tmax_local)

        signal_1d = signal_1d[slice_start:slice_end]

        min_value = np.min(signal_1d)
        max_value = np.max(signal_1d)
        
        # Umbral menor que el mínimo de la señal (Todo está pintado)
        if umbral < min_value:
            logging.info(f"El umbral ({umbral}) es menor que toda la señal. Se seleccionará todo el tramo")
        
        # Umbral mayor que el máximo (Nada está pintado)
        if umbral > max_value:
            logging.info(f"En el intervalo {tmin}-{tmax}s, la señal (max: {max_value:.2f}) no supera el umbral ({umbral}). No hay contracciones")
        
        start_time_segundos = (slice_start + self.first_samp) / self.sfreq
        t = (np.arange(len(signal_1d)) / self.sfreq) + start_time_segundos

        plt.figure(figsize=(12, 6))

        # Grafica de la señal EMG completa
        plt.plot(t, signal_1d, label='Señal EMG', color="#0022FF", alpha=0.7, zorder=5)

        plt.fill_between(
            t, 
            np.min(signal_1d), # Valor mínimo
            np.max(signal_1d), # Valor máximo
            where=(signal_1d > umbral), # Condición para llenar
            color="#9500FF", 
            alpha=0.3, 
            label=f'Sobre Umbral ({umbral})',
            transform=plt.gca().get_xaxis_transform(), # Truco para llenar todo el alto Y
        )
        
        # Línea punteada del umbral definido
        plt.axhline(umbral, color='red', linestyle='--', alpha=0.5, label='Umbral')

        plt.title('Segmentación de EMG: Contracciones Detectadas')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [µV]')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def freq_time():
        """
        Función para generar espectrogramas tiempo-frecuencia de la señal EMG.
        """


        pass    
    
    def plotSignal(self, raw_emg:np.ndarray, tmin:int=0, tmax:int=10, show_ann:bool=True, channel:int=0):
        """
        Grafica un segmento de la señal EMG cruda y la señal filtrada (self.data).

        Muestra un gráfico superpuesto de la señal EMG original (`raw_emg`) y la 
        versión procesada y filtrada (`self.data`) para un canal específico. 
        Permite la visualización de anotaciones (`self.anotaciones`) y la selección 
        de un intervalo de tiempo absoluto.

        Args:
            raw_emg (np.ndarray): 
                La señal EMG cruda original. Debe ser compatible en forma y origen
                (mismo first_samp) con la señal procesada `self.data`.
            tmin (int or float, optional):
                Tiempo inicial en segundos ABSOLUTO (respecto al inicio del registro original).
                Por defecto 0.
            tmax (int or float, optional):
                Tiempo final en segundos ABSOLUTO (respecto al inicio del registro original).
                Debe cumplirse tmin < tmax. Por defecto 10.
            show_ann (bool, optional):
                Si True, dibuja líneas/zonas para las anotaciones en `self.anotaciones`.
                Por defecto True.
            channel (int, optional):
                Índice del canal a graficar. Por defecto 0.

        Returns:
            None: Muestra el gráfico correspondiente mediante Matplotlib.

        Raises:
            ValueError:
                - Si tmin >= tmax.
                - Si la ventana [tmin, tmax] no intersecta la señal almacenada en la instancia
                (revisar `first_samp` y el rango de la señal procesada).
            RuntimeError:
                - Si `self.is_filtered` es False, ya que no existe señal filtrada para mostrar.

        Behavior / Notes:
            - Conversión muestras/tiempo:
                `tmin` y `tmax` se interpretan como segundos absolutos.
                `start_global = int(tmin * self.sfreq)`
                `end_global   = int(tmax * self.sfreq)`
            - Traducción a coordenadas locales (relativas a `self.data`):
                `start_local = start_global - int(self.first_samp)`
                `end_local   = end_global - int(self.first_samp)`
            - Manejo de límites (Clamping):
                Si la ventana solicitada (ej: `tmin=0s`) comienza antes que los datos 
                almacenados (ej: `self.first_samp=3s`), `start_local` será negativo. 
                El método ajusta esto usando `slice_start = max(0, start_local)`, por lo que
                el gráfico comenzará en el primer dato disponible (3s absolutos).
            - Eje de Tiempo:
                El eje X del gráfico se genera calculando el tiempo absoluto real del 
                inicio del *slice* (`real_start_time_sec`), asegurando que la 
                visualización sea coherente con los tiempos absolutos.
            - Anotaciones:
                `self.anotaciones.onset` y `duration` se interpretan en segundos absolutos
                para dibujarlas correctamente en la ventana.
        """
        if self.is_filtered:

            ch = channel
            if self.data.ndim == 2:
                emg_clean = np.asarray(self.data[ch]).ravel()
                emg_raw = np.asarray(raw_emg[ch]).ravel()
            else:
                emg_clean = self.data
                emg_raw = raw_emg

            tmin_samps_global = int(tmin * self.sfreq)
            tmax_samps_global = int(tmax * self.sfreq)

            if tmin_samps_global >= tmax_samps_global:
                raise ValueError("tmin debe ser menor que tmax y dentro del rango de la señal.")
            
            tmin_local = tmin_samps_global - int(self.first_samp)
            tmax_local = tmax_samps_global - int(self.first_samp)

            n_local = emg_clean.shape[0]
            slice_start = max(0, tmin_local)
            slice_end = min(n_local, tmax_local)

            if slice_start >= slice_end:
                raise ValueError("La ventana solicitada no intersecta la señal almacenada en esta instancia (revisar first_samp).")        

            emg_clean_slice = emg_clean[slice_start:slice_end]
            emg_raw_slice = emg_raw[slice_start:slice_end] if raw_emg is not None else None

            real_start_samps_global = slice_start + int(self.first_samp)
            real_start_time_sec = real_start_samps_global / self.sfreq
            real_end_time_sec = real_start_time_sec + (len(emg_clean_slice) / self.sfreq)

            fig, ax = plt.subplots(figsize=(12, 6))

            x_axis = np.arange(len(emg_clean_slice)) / self.sfreq + real_start_time_sec

            if emg_raw_slice is not None:
                ax.plot(x_axis, emg_raw_slice, color="#8B8383", label='Raw Signal', zorder=2)

            ax.plot(x_axis, emg_clean_slice, color="#4554F7", label='Clean Signal', alpha=0.9, zorder=5)

            # Eventos externos
            if show_ann and getattr(self, 'anotaciones', None) is not None:
                import hashlib

                def get_color_from_string(s):
                    hash_obj = hashlib.md5(s.encode())
                    hash_int = int(hash_obj.hexdigest(), 16) % (256**3)
                    r = (hash_int >> 16) & 255
                    g = (hash_int >> 8) & 255
                    b = hash_int & 255
                    return f"#{r:02x}{g:02x}{b:02x}"
            
                onsets_seconds = np.asarray(self.anotaciones.onset)  # onset en segundos absolutos
                durations_seconds = np.asarray(self.anotaciones.duration)  # duración en segundos
                descriptions = self.anotaciones.description

                for onset_sec, duration_sec, desc in zip(onsets_seconds, durations_seconds, descriptions):
                    # Calcular el tiempo de finalización
                    if duration_sec > 0:
                        end_sec = onset_sec + duration_sec
                        
                        # Convertir a muestras globales para verificar superposición con la ventana
                        onset_g = int(onset_sec * self.sfreq)
                        end_g = int(end_sec * self.sfreq)
                        
                        # Verificar si la anotación se superpone con la ventana de visualización
                        if not (end_g < tmin_samps_global or onset_g > tmax_samps_global):
                            # Calcular los límites de visualización para la franja
                            draw_start = max(onset_sec, tmin)
                            draw_end = min(end_sec, tmax)
                            
                            color = get_color_from_string(desc)

                            # Dibujar la franja con transparencia
                            ax.axvspan(draw_start, draw_end, color=color, alpha=0.2, zorder=1)
                            
                            # Dibujar línea vertical en el inicio de la anotación (opcional)
                            ax.axvline(x=onset_sec, color="#000000", linestyle="--", alpha=0.7, zorder=3)
                            
                            # Añadir texto descriptivo
                            ax.text(onset_sec, ax.get_ylim()[1]*0.9, str(desc),
                                    verticalalignment="bottom", fontsize=10, color="#000000")
                    else:
                        # Dibujar solo línea vertical para anotaciones instantáneas
                        if tmin_samps_global <= onset_g < tmax_samps_global:
                            ax.axvline(x=onset_sec, color="#00C853", linestyle="--", alpha=0.8, zorder=3)
                            ax.text(onset_sec, ax.get_ylim()[1]*0.9, str(desc),
                                    rotation=90, verticalalignment="bottom", fontsize=10, color="#00C853")

            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud (µV)')
            ax.set_title(f"Señal Cruda vs Filtrada — Ventana [{real_start_time_sec:.1f}–{real_end_time_sec:.1f}]s")
            ax.legend()

            plt.grid(True, alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            raise RuntimeError("La señal EMG no ha sido filtrada. No se puede graficar la señal filtrada.")
        
