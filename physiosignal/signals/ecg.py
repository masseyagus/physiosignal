from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Futura implementación
class ECG(RawSignal):
    """
    Representa una señal de Electrocardiografía (ECG) con utilidades para
    detección de picos R, cálculo de frecuencia cardiaca y visualización.

    Key Features:
        - Detección de picos R mediante NeuroKit2 (`nk.ecg_peaks`) — se usa únicamente como detector.
        - Almacenamiento de resultados de detección en `self.peaks`, `self.info_peaks` y `self.r_peaks`.
        - `r_peaks` contiene índices **locales** (0..N-1) relativos al array procesado.
        - Alineación con anotaciones externas mediante `first_samp` (para convertir índices locales -> globales).
        - Visualización de segmentos con marcaje de picos R y anotaciones.

    Attributes:
        data : np.ndarray
            Señal ECG cruda; forma (n_canales, n_muestras) o (n_muestras,) si mono.
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Metadatos de canales.
        anotaciones : Annotations
            Anotaciones externas (onset en segundos absolutos, duration en segundos, description).
        first_samp : int
            Offset en muestras del primer dato de `self.data` respecto al registro original.
        peaks : dict-like | None
            Resultado retornado por `nk.ecg_peaks` (puede incluir máscara binaria por muestra).
        info_peaks : dict-like | None
            Diccionario de info devuelto por `nk.ecg_peaks` (puede contener 'ECG_R_Peaks' con índices).
        r_peaks : np.ndarray | None
            Índices locales (dtype=int, 1D) de picos R detectados; array vacío si no hay picos.
        r_peaks_global : np.ndarray | None
            (Se asigna en `plot_r_peaks`) Índices de picos en coordenada **global** (local + first_samp).
        heart_rate : float | None
            Frecuencia cardiaca estimada (BPM). Inicialmente None; queda para ser poblada por la función
            `heart_rate()` si se implementa/ejecuta por separado.

    Notes:
        - Esta clase **no** hace preprocesado automático antes de `nk.ecg_peaks`. Debes preprocesar la señal
        usando los métodos de la clase RawSignal
        - `first_samp` es crucial: `r_peaks` son relativos a la señal que se pasó a `nk.ecg_peaks`;
        para mapearlos a la referencia del registro original sumá `first_samp`.
    """

    def __init__(self, raw:RawSignal=None, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True):
        """
        Inicializa una instancia de ECG.

        Parameters:
            raw : RawSignal, optional
                Si se proporciona, se copian los atributos de `raw` (data, sfreq, info, anotaciones, first_samp).
            data : np.ndarray, optional
                Señal cruda (n_canales, n_muestras) o 1D (n_muestras,).
            sfreq : float, optional
                Frecuencia de muestreo (Hz). Si `raw` fue pasado, se usa `raw.sfreq` si sfreq es None.
            info : Info, optional
                Metadatos de canales.
            anotaciones : Annotations, optional
                Eventos con columnas `onset` (s absolutos), `duration` (s) y `description`.
            first_samp : int, optional
                Offset en muestras del primer dato de `self.data` respecto al inicio del registro original.
                Por defecto 0.
            see_log : bool, optional
                Controla la configuración del logger interno.

        Notes:
            - Si se pasa `raw`, el constructor hace una copia profunda de sus atributos y preserva `first_samp`.
            - Inicialmente `peaks`, `info_peaks`, `r_peaks`, `r_peaks_global` y `heart_rate` se establecen en None o arrays vacíos.
            - `heart_rate` quedará en None hasta que se ejecute la función `heart_rate()`.
        """
        from copy import deepcopy
        if raw is not None:
            super().__init__(deepcopy(raw.data),
                            deepcopy(raw.sfreq),
                            deepcopy(raw.info),
                            deepcopy(raw.anotaciones),
                            deepcopy(raw.first_samp),
                            see_log)
        else:
            super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo

        self.peaks = None
        self.info_peaks = None
        self.r_peaks = None
        self.r_peaks_global = None
        self.heart_rate = None

    def peak_detection(self, channel:int=0):
        """
        Detecta picos R usando NeuroKit2 (`nk.ecg_peaks`) sobre un canal seleccionado.

        Args:
            channel : int, optional
                Índice del canal a procesar cuando `self.data` es multicanal (forma (n_canales, n_muestras)).
                Por defecto 0. Si `self.data` es 1D, se procesa esa señal.

        Returns:
            None

        Side effects / Atributos generados:
            - self.peaks : dict-like
                DataFrame o dict-like devuelto por `nk.ecg_peaks`. Puede contener una máscara `ECG_R_Peaks`.
            - self.info_peaks : dict-like
                Diccionario de información (puede incluir `ECG_R_Peaks` con índices).
            - self.r_peaks : np.ndarray
                Índices locales (enteros, 1D) de picos R normalizados desde `info_peaks` o desde la máscara en `peaks`.
                Si no se encuentran picos, se asigna un array vacío dtype=int.

        Raises:
            ValueError:
                Si `channel` está fuera de rango cuando `self.data` es multicanal.

        Notes:
            - Este método **no** aplica filtros ni limpieza extra; usa `nk.ecg_peaks` exclusivamente como detector.
            - `nk.ecg_peaks` puede devolver picos como índices (`info_peaks['ECG_R_Peaks']`) o como máscara
            booleana/numérica en `peaks['ECG_R_Peaks']`; el método convierte ambas formas a `self.r_peaks`.
            - Para obtener índices en la escala del registro original usar `self.r_peaks + self.first_samp`.

        Examples:
            >>> # Señal mono en instancia
            >>> ecg = ECG(data=my_ecg_1d, sfreq=512.0, first_samp=1536)
            >>> ecg.peak_detection()                    # detecta picos en el canal 0
            >>> print(ecg.r_peaks[:10])                 # índices locales (0..N-1)
            >>> print(ecg.r_peaks + ecg.first_samp)     # índices globales (registro original)

            >>> # Señal multicanal: procesar canal 1
            >>> ecg = ECG(data=my_multi_chan, sfreq=500.0, first_samp=0)
            >>> ecg.peak_detection(channel=1)
            >>> len(ecg.r_peaks)                        # número de picos detectados en el canal 1
        """
        
        if self.data.ndim == 2:
            if channel >= self.data.shape[0]:
                raise ValueError(f"Canal {channel} fuera de rango. La señal tiene {self.data.shape[0]} canales.")
            data = self.data[channel]
        else:
            data = self.data.reshape(-1)

        peaks, info_peaks = nk.ecg_peaks(data, self.sfreq, method='neurokit', correct_artifacts=True)

        self.peaks = peaks
        self.info_peaks = info_peaks

        # normalizar/extraer índices locales de picos a self.r_peaks (entero, 1D)
        if 'ECG_R_Peaks' in info_peaks:
            self.r_peaks = np.asarray(info_peaks['ECG_R_Peaks']).astype(int)

        elif 'ECG_R_Peaks' in peaks:
            mask = np.asarray(peaks['ECG_R_Peaks']).ravel()
            self.r_peaks = np.where(mask != 0)[0].astype(int)

        else:
            self.r_peaks = np.array([], dtype=int)

        self._last_channel = channel

        return peaks, info_peaks

    def plot_r_peaks(self, tmin:int=0, tmax:int=10, plot_raw:bool=False, show_ann:bool=True,
                     raw_signal=None):
        """
        Grafica un segmento de la señal limpia con los picos R detectados y las anotaciones.

        Args:
            tmin : int or float, optional
                Tiempo inicial en segundos ABSOLUTO respecto al inicio del registro original.
                Por defecto 0
            tmax : int or float, optional
                Tiempo final en segundos ABSOLUTO respecto al inicio del registro original.
                Debe cumplirse tmin < tmax.
                Por defecto 10
            plot_raw : bool, optional
                Si True, también grafica la señal cruda (`ECG_Raw`) retornada en `self.peaks`.
                Por defecto es False
            show_ann : bool, optional
                Si True, dibuja líneas verticales y etiquetas para las anotaciones en `self.anotaciones`
                que caigan dentro de la ventana [tmin, tmax].
                Por defecto True

        Returns:
            None

        Raises:
            ValueError:
                - Si no existen `self.peaks` o `self.info_peaks` (llamar primero a `peak_detection()`).
                - Si tmin >= tmax.
                - Si la ventana [tmin, tmax] no intersecta la señal almacenada en la instancia
                (revisar `first_samp` y el rango de la señal procesada).

        Behavior / Notes:
            - Conversión muestras/tiempo:
                start_global = int(tmin * sfreq)
                end_global   = int(tmax * sfreq)
            Para indexar arrays locales (los devueltos por NeuroKit2) usar:
                start_local = start_global - self.first_samp
                end_local   = end_global - self.first_samp
            El método hace clipping automático para no salir de los límites locales.
            - Picos:
                * `self.r_peaks` son índices locales (0..N-1).
                * `picks_global = self.r_peaks + self.first_samp` son índices en coordenada global.
            En el scatter se dibujan las abscisas en segundos absolutos: `picks_global / self.sfreq`.
            - Anotaciones:
                * `self.anotaciones.onset` se asume en segundos absolutos. Se convierten a muestras
                globales con `onsets_global = (onsets_seconds * sfreq).astype(int)` y solo se muestran
                las que caen dentro de la ventana solicitada.
            - El método asigna `self.r_peaks_global = r_peaks_local + int(self.first_samp)` como atributo
            auxiliar disponible tras el plot.

        Examples:
            >>> # Mostrar 30–36 s del registro (tmin/tmax son tiempos absolutos)
            >>> ecg = ECG(data=my_ecg_1d, sfreq=512.0, first_samp=1536)
            >>> ecg.peak_detection()
            >>> ecg.plot_r_peaks(tmin=30, tmax=36, plot_raw=True, show_ann=True)

            >>> # Si la señal procesada no comienza en 0 (first_samp>0), tmin/tmax siguen siendo absolutos:
            >>> # Para ver la porción local correspondiente, el método calcula start_local = int(30*sfreq) - first_samp.
            >>> ecg.plot_r_peaks(tmin=30, tmax=36)

            >>> # Si solo querés ver la señal limpia (sin la cruda)
            >>> ecg.plot_r_peaks(tmin=10, tmax=12, plot_raw=False, show_ann=False)
        """

        if not hasattr(self, 'peaks') or not hasattr(self, 'info_peaks'):
             raise ValueError("No existe el atributo peaks/info_peaks: llamá primero a process_signal().")

        ch = getattr(self, '_last_channel', 0)
        if self.data.ndim == 2:
            ecg_clean = np.asarray(self.data[ch]).ravel()
            ecg_raw = np.asarray(raw_signal[ch]).ravel()
        else:
            ecg_clean = self.data.reshape(-1)
            ecg_raw = raw_signal

        # Ventana a mostrar en muestras
        tmin_samps_global = int(tmin * self.sfreq)
        tmax_samps_global = int(tmax * self.sfreq)

        if tmin_samps_global >= tmax_samps_global:
            raise ValueError("tmin debe ser menor que tmax y dentro del rango de la señal.")
        
        tmin_local = tmin_samps_global - int(self.first_samp)
        tmax_local = tmax_samps_global - int(self.first_samp)

        n_local = ecg_clean.shape[0]
        slice_start = max(0, tmin_local)
        slice_end = min(n_local, tmax_local)

        if slice_start >= slice_end:
            raise ValueError("La ventana solicitada no intersecta la señal almacenada en esta instancia (revisar first_samp).")        

        ecg_clean_slice = ecg_clean[tmin_local:tmax_local]
        ecg_raw_slice = ecg_raw[tmin_local:tmax_local]

        r_peaks_local_all = self.r_peaks  # local indices

        r_peaks_global_all = r_peaks_local_all + int(self.first_samp)
        self.r_peaks_global = r_peaks_global_all

        mask = (r_peaks_global_all >= tmin_samps_global) & (r_peaks_global_all < tmax_samps_global)
        picks_global = r_peaks_global_all[mask]

        picks_in_window = picks_global - tmin_samps_global

        fig, ax = plt.subplots(figsize=(12, 6))

        x_axis = np.arange(tmin_samps_global, tmin_samps_global + len(ecg_clean_slice)) / self.sfreq

        if ecg_raw_slice is not None and plot_raw:
            ax.plot(x_axis, ecg_raw_slice, color="#8B8383", label='Raw Signal', zorder=1)

        ax.plot(x_axis, ecg_clean_slice, color="#4554F7", label='Clean Signal', alpha=0.7, zorder=2)

        if picks_in_window.size > 0:
            amp = ecg_clean_slice[picks_in_window]

            ax.scatter(picks_global/self.sfreq, amp, color="#FF0000",
                       label='R Peaks', zorder=5, alpha=0.7, marker='*')
        else:
            ax.text(0.02, 0.95, "No R-peaks in plotted window", transform=ax.transAxes,
                    fontsize=10, color="#A2A4A5", verticalalignment='top')

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
                        ax.axvspan(draw_start, draw_end, color=color, alpha=0.2, zorder=3)
                        
                        # Dibujar línea vertical en el inicio de la anotación (opcional)
                        ax.axvline(x=onset_sec, color="#000000", linestyle="--", alpha=0.7, zorder=4)
                        
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
        ax.set_title(f"Detección de Picos R  — Ventana [{tmin:.1f}–{tmax:.1f}]s")
        ax.legend()

        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()  

    def find_heart_rate(self):
        """
        Calcula la frecuencia cardíaca promedio (BPM) basada en los picos R detectados.

        Returns:
            float: Frecuencia cardíaca en beats por minuto (BPM).

        Raises:
            ValueError:
                Si no se han detectado picos R previamente (`self.r_peaks` es None).

        Side effects / Atributos generados:
            - self.heart_rate : float
                La frecuencia cardíaca calculada en BPM.

        Notes:
            - Este método depende de que `peak_detection()` haya sido ejecutado.
            - Internamente llama a `_compute_heart_rate()` para obtener el valor.
        """
        if self.r_peaks is None:
            raise ValueError("Primero debe detectar los picos R usando peak_detection()")

        return self._compute_heart_rate()

    def plot_heart_rate(self, before:float=0.2, after:float=0.5):
        """
        Grafica latidos individuales extraídos alrededor de cada pico R detectado,
        junto con la forma de latido promedio y la frecuencia cardíaca promedio.

        Args:
            before : float, optional
                Tiempo en segundos a incluir antes del pico R. Por defecto 0.2 s.
            after : float, optional
                Tiempo en segundos a incluir después del pico R. Por defecto 0.5 s.

        Raises:
            ValueError:
                Si no se han detectado picos R previamente (`self.r_peaks` es None).

        Side effects / Atributos generados:
            - self.heart_rate : float
                Se asegura de estar calculada antes de graficar.

        Notes:
            - Se generan segmentos de ECG centrados en cada pico R mediante `_extract_segment`.
            - Cada segmento se grafica con transparencia y se superpone la curva promedio.
            - La frecuencia cardíaca promedio se muestra en el título del gráfico.
        """
        if self.r_peaks is None:
            raise ValueError("Primero debe detectar los picos R usando peak_detection()")
        
        segments, time_vector = self._extract_segment(before=before, after=after)
        segments_array = np.array(segments)
        
        # Hallo el latido promedio
        avg_beat = np.nanmean(segments_array, axis=0)

        # Aseguramps que avg_beat tenga la misma longitud que time_vector
        if len(avg_beat) != len(time_vector):
            # Si hay discrepancia, truncar o interpolar para que coincidan
            min_length = min(len(avg_beat), len(time_vector))
            avg_beat = avg_beat[:min_length]
            time_vector = time_vector[:min_length]
        
        fig, ax = plt.subplots(figsize=(12, 6))

        for segment in segments:
            if len(segment) != len(time_vector):
                # Grafico cada segmento o latido
                ax.plot(time_vector, segment, color="#9B9797", alpha=0.5, zorder=1, linewidth=0.5)
            else:
                ax.plot(time_vector, segment, color='#9B9797', alpha=0.5, zorder=1, linewidth=0.5)

        # Grafico el latido promedio
        ax.plot(time_vector, avg_beat, color="#8000FF", linewidth=2, zorder=3, label='Forma de ritmo promedio')
        ax.axhline(0, color="#000000", linestyle='--', alpha=0.5)
        ax.axvline(0, color="#000000", linestyle='--', alpha=0.5)
        
        # Configurar la gráfica
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (µV)')

            # Asegurarse de que heart_rate esté calculado
        if not hasattr(self, 'heart_rate') or self.heart_rate is None:
            self._compute_heart_rate()

        ax.set_title(f'Latidos Individuales (FC Promedio: {self.heart_rate:.1f} bpm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def freq_time(self):
        pass

    def _compute_heart_rate(self):
        """
        Método privado. Calcula la frecuencia cardíaca promedio en BPM
        a partir de los intervalos RR obtenidos de `self.r_peaks`.

        Returns:
            float: Frecuencia cardíaca promedio en beats por minuto (BPM).

        Side effects / Atributos generados:
            - self.heart_rate : float
                Se almacena la frecuencia cardíaca calculada.

        Notes:
            - Devuelve 0.0 si no hay suficientes picos R para calcular intervalos.
            - Utiliza el promedio de los intervalos RR para calcular BPM.
        """
        if self.r_peaks is None or len(self.r_peaks) < 2:
            return 0.0
        
        intervals = np.diff(self.r_peaks) / self.sfreq
        self.heart_rate = 60.0 / np.mean(intervals)

        return self.heart_rate
    
    def _extract_segment(self, before:float=0.2, after:float=0.5):
        """
        Método privado. Extrae segmentos de ECG centrados en cada pico R,
        convirtiendo la señal en latidos individuales.

        Args:
            before : float, optional
                Tiempo en segundos a incluir antes del pico R. Por defecto 0.2 s.
            after : float, optional
                Tiempo en segundos a incluir después del pico R. Por defecto 0.5 s.

        Returns:
            tuple:
                - list of np.ndarray: Lista de segmentos individuales del ECG.
                - np.ndarray: Vector de tiempo correspondiente a los segmentos, centrado en 0 s.

        Notes:
            - Rellena con NaN los segmentos que no puedan completar la ventana solicitada
            por estar al inicio o final de la señal.
            - Utiliza `self._last_channel` si la señal es multicanal.
        """
        samp_before = int(before * self.sfreq)
        samp_after = int(after * self.sfreq)
        total_length = samp_before + samp_after

        if self.data.ndim == 2:
            ch = getattr(self, '_last_channel', 0)
            data = self.data[ch]
        else:
            data = self.data

        segments = []
        for r_peak in self.r_peaks:

            start = max(0, r_peak - samp_before)
            end = min(len(data), r_peak + samp_after)

            segment = data[start:end]

            # Rellenar con NaN si es necesario para mantener la longitud constante
            if len(segment) < total_length:
                # Calcular cuántas muestras faltan al inicio y al final
                missing_start = max(0, samp_before - r_peak)
                missing_end = total_length - len(segment) - missing_start
                
                # Rellenar con NaN
                segment = np.concatenate([
                    np.full(missing_start, np.nan),
                    segment,
                    np.full(missing_end, np.nan)
                ])
        
            segments.append(segment)

        time_vector = np.linspace(-before, after, samp_before+samp_after)

        return segments, time_vector
