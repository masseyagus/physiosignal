from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

class ECG(RawSignal):
    """
    Representa una señal de Electrocardiografía (ECG) con utilidades para
    detección de picos R, cálculo de frecuencia cardiaca, evaluación de calidad
    de señal y visualizaciones avanzadas.

    Key Features:
        - Detección de picos R mediante NeuroKit2 (`nk.ecg_peaks`)
        - Cálculo de intervalos RR y frecuencia cardíaca
        - Evaluación de calidad de señal basada en métricas de detección
        - Delineado de ondas P, Q, R, S, T
        - Visualización: plot de picos R, segmentación de latidos, Poincaré con SD1/SD2
        - Estimación automática de ventanas temporales óptimas para análisis
    
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
            Resultado retornado por `nk.ecg_peaks`.
        info_peaks : dict-like | None
            Diccionario de info devuelto por `nk.ecg_peaks`.
        r_peaks : np.ndarray | None
            Índices locales de picos R detectados.
        r_peaks_global : np.ndarray | None
            Índices de picos en coordenada global (local + first_samp).
        heart_rate : float | None
            Frecuencia cardiaca estimada (BPM).
        rr_intervals : np.ndarray | None
            Intervalos RR en segundos, calculados durante la detección de picos.
        rr_times : np.ndarray | None
            Tiempos absolutos de los intervalos RR.

    Notes:
        - El preprocesamiento debe hacerse usando los métodos de la clase RawSignal.
        - `first_samp` se interpreta como número de muestras de offset.
        - Los métodos de calidad de señal ayudan a determinar la confiabilidad de los resultados.
        - Los métodos de estimación de ventanas proporcionan valores por defecto inteligentes.
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
                **Nota:** se asume que `onset` y `duration` están en segundos; si están en muestras,
                convertílos a segundos antes de crear la instancia.
            first_samp : int, optional
                Offset en muestras del primer dato de `self.data` respecto al inicio del registro original.
                Por defecto 0. Se almacena como `int(self.first_samp)`.
            see_log : bool, optional
                Controla la configuración del logger interno.

        Behavior / Side effects:
            - Inicializa atributos: `peaks`, `info_peaks`, `r_peaks`, `r_peaks_global`, `heart_rate`.
            - No ejecuta detección de picos automáticamente: llamá a `peak_detection()` cuando quieras poblar `r_peaks`.
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
        Detecta picos R usando NeuroKit2 (`nk.ecg_peaks`) sobre un canal seleccionado y guarda
        índices locales y globales.

        Args:
            channel : int, optional
                Índice del canal a procesar cuando `self.data` es multicanal (forma (n_canales, n_muestras)).
                Por defecto 0. Si `self.data` es 1D, se procesa esa señal.

        Returns:
            tuple: (peaks, info_peaks)
                - peaks: DataFrame/dict-like devuelto por `nk.ecg_peaks`.
                - info_peaks: diccionario de información de NeuroKit2.

        Side effects / Atributos generados:
            - self.peaks : dict-like
            - self.info_peaks : dict-like
            - self.r_peaks : np.ndarray (índices locales, enteros)
            - self.r_peaks_global : np.ndarray (índices globales, enteros) == self.r_peaks + int(self.first_samp)
            - self._last_channel : canal procesado (int)

        Raises:
            ValueError:
                Si `channel` está fuera de rango cuando `self.data` es multicanal.

        Notes:
            - `r_peaks` son relativos a la porción de señal pasada al detector (índices locales).
            - `r_peaks_global` está en la referencia del registro original y es el valor que
            debés usar para comparar con `self.anotaciones` (que se asumen en segundos).
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

        try:
            self.r_peaks_global = (self.r_peaks.astype(int) + int(self.first_samp))
        except Exception:
            self.r_peaks_global = None

        # Calculo y almaceno los intervalos RR y sus tiempos ABSOLUTOS
        if len(self.r_peaks) >= 2:
            self.rr_intervals , self.rr_times = self.get_rr_intervals(return_times=True)
        else:
            self.rr_intervals = np.array([], dtype=float)
            self.rr_times = np.array([], dtype=float)

        self._last_channel = channel

        return peaks, info_peaks

    def plot_r_peaks(self, tmin:int=0, tmax:int=10, plot_raw:bool=False, show_ann:bool=True,
                     raw_signal=None):
        """
        Grafica un segmento de la señal limpia con los picos R detectados y las anotaciones.

        Args:
            tmin : int or float, optional
                Tiempo inicial en segundos ABSOLUTO respecto al inicio del registro original.
                Por defecto 0.
            tmax : int or float, optional
                Tiempo final en segundos ABSOLUTO respecto al inicio del registro original.
                Debe cumplirse tmin < tmax. Por defecto 10.
            plot_raw : bool, optional
                Si True, también grafica la señal cruda (`raw_signal`) si se provee.
            show_ann : bool, optional
                Si True, dibuja líneas/zonas para las anotaciones en `self.anotaciones`.
            raw_signal : np.ndarray | None, optional
                Señal cruda asociada (misma forma que self.data) para superponer si `plot_raw=True`.

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
                start_global = int(tmin * self.sfreq)
                end_global   = int(tmax * self.sfreq)
            Para indexar arrays locales (los devueltos por NeuroKit2) usar:
                start_local = start_global - int(self.first_samp)
                end_local   = end_global - int(self.first_samp)
            - El método asigna `self.r_peaks_global = self.r_peaks + int(self.first_samp)` y usa
            `picks_global / self.sfreq` para colocar los marcadores en segundos absolutos.
            - `self.anotaciones.onset` y `duration` se interpretan en segundos absolutos; las franjas
            de anotación se dibujan si se solapan con la ventana [tmin, tmax].
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

        ax.plot(x_axis, ecg_clean_slice, color="#4554F7", label='Clean Signal', alpha=0.9, zorder=2)

        if picks_in_window.size > 0:
            amp = ecg_clean_slice[picks_in_window]

            ax.scatter(picks_global/self.sfreq, amp, color="#FF0000",
                       label='R Peaks', zorder=5, alpha=0.8, marker='*')
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
            - Internamente llama a `_compute_heart_rate()` que usa `get_rr_intervals()` para
            obtener intervalos RR (en segundos) y asegura que los tiempos absolutos están disponibles.
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
                Si no se han detectado picos R previamente (`self.r_peaks` is None).

        Side effects / Atributos generados:
            - self.heart_rate : float
                Se asegura de estar calculada antes de graficar.

        Notes:
            - Este método utiliza `extract_segment()` para generar latidos individuales y `get_instantaneous_hr()`
            para obtener HR por latido y sus tiempos absolutos (útiles para análisis por evento).
            - En presencia de `self.anotaciones`, el método puede calcular promedios de HR por evento
            usando los tiempos absolutos devueltos por `get_instantaneous_hr()`.
            - Las unidades en los ejes son segundos y µV respectivamente.
        """
        if self.r_peaks is None:
            raise ValueError("Primero debe detectar los picos R usando peak_detection()")
        
        segments, time_vector = self.extract_segment(before=before, after=after)
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

    def poincare(self, by_event: bool = False, annotate_values: bool = True):
        """
        Poincaré plot de RR_n vs RR_{n+1} con SD1/SD2. Opcionalmente colorea puntos por anotaciones
        (self.anotaciones) y calcula SD1/SD2 por evento.

        Args:
            by_event : bool, optional
                Si True, colorea cada punto según la anotación (se asigna cada punto al evento cuyo
                tiempo (segundo pico del par RR) satisface onset <= t < onset+duration).
            annotate_values : bool, optional
                Si True, muestra SD1/SD2 globales en el gráfico.

        Returns:
            dict: métricas calculadas
                {'SD1_global': float, 'SD2_global': float, 'per_event': {label: (sd1, sd2, n_points)}}.

        Raises:
            ValueError:
                Si `self.r_peaks` no contiene suficientes picos (necesita al menos 3 picos para un Poincaré).

        Notes:
            - Cada punto representa un par de intervalos consecutivos: (RR_n, RR_{n+1}).
            - El tiempo asociado a cada punto es el tiempo absoluto del segundo pico del par:
                t_point = (r_peaks[i+1] + first_samp) / sfreq.
            - SD1 mide la variabilidad de corto plazo (dispersión perpendicular a la identidad).
            - SD2 mide la variabilidad de largo plazo (dispersión a lo largo de la identidad).
            - Leyenda: las etiquetas aparecen con el formato "<label> (n=NN)". Aquí, `n` es el número de puntos
            ploteados para esa etiqueta, es decir la cantidad de pares (RR_n, RR_{n+1}) asignados a esa
            condición o evento. **No** confundir con el número de picos R: si un grupo tiene `n` puntos
            corresponde a `n+1` picos R (en general) porque cada punto consume dos picos y los puntos
            consecutivos se solapan en un pico.
            - Si un grupo tiene <2 puntos, SD1/SD2 no son representativos y se devuelven NaN para esas métricas.
            - Los puntos etiquetados como 'No event' son aquellos que no caen dentro de ninguna anotación.
        """
        if self.r_peaks is None or len(self.r_peaks) < 3:
            raise ValueError("Necesitas al menos 3 picos R para Poincaré. Ejecutá peak_detection().")

        rr, times = self.rr_intervals, self.rr_times
        rr_n  = rr[:-1]
        rr_n1 = rr[1:]
        times_pairs = times[:-1]  # tiempo asociado a cada punto (opcional)

        # SD1/SD2 global
        diff = rr_n1 - rr_n
        sum_  = rr_n1 + rr_n
        SD1_global = np.std(diff) / np.sqrt(2)
        SD2_global = np.std(sum_) / np.sqrt(2)

        fig, ax = plt.subplots(figsize=(9, 8))
        cmap = None
        per_event = {}

        if by_event and getattr(self, 'anotaciones', None) is not None:
            # Construir etiquetas por intervalo
            onsets = np.asarray(self.anotaciones.onset, dtype=float)
            durations = np.asarray(self.anotaciones.duration, dtype=float)
            descs = np.asarray(self.anotaciones.description, dtype=object)

            labels = np.array(['__no_event__'] * len(times_pairs), dtype=object)

            for onset, dur, desc in zip(onsets, durations, descs):
                start = onset
                end = onset + max(0.0, dur)
                mask = (times_pairs >= start) & (times_pairs < end)
                labels[mask] = str(desc)

            unique_labels = np.unique(labels)
            # Color map
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab10', len(unique_labels))

            for i, lab in enumerate(unique_labels):
                mask = labels == lab
                if lab == '__no_event__':
                    lab_name = 'No event'
                else:
                    lab_name = lab
                ax.scatter(rr_n[mask], rr_n1[mask], label=f'{lab_name} (n={mask.sum()})',
                        alpha=0.7, s=20, zorder=2, c=[cmap(i)])
                # calcular SD1/SD2 por evento y guardarlo
                if mask.sum() >= 2:
                    d = (rr_n1[mask] - rr_n[mask])
                    s = (rr_n1[mask] + rr_n[mask])
                    sd1 = np.std(d) / np.sqrt(2)
                    sd2 = np.std(s) / np.sqrt(2)
                else:
                    sd1 = np.nan
                    sd2 = np.nan
                per_event[lab] = (sd1, sd2, int(mask.sum()))
        else:
            ax.scatter(rr_n, rr_n1, color="#63DC7B", alpha=0.6, zorder=1, label='Intervalos R-R')

        # Línea identidad
        min_rr = min(np.min(rr_n), np.min(rr_n1))
        max_rr = max(np.max(rr_n), np.max(rr_n1))
        ax.plot([min_rr, max_rr], [min_rr, max_rr], color="#0008FF", linestyle='--', label='Línea identidad', zorder=3)

        # Elipse global
        from matplotlib.patches import Ellipse
        mean_rr = np.mean(rr)
        ellipse = Ellipse(xy=(mean_rr, mean_rr), width=2*SD2_global, height=2*SD1_global,
                        angle=45, edgecolor="#000000", fc='None', lw=1.5, zorder=4, label=f'Centroide: {mean_rr:.3f}s')
        ax.add_patch(ellipse)

        # Flechas (SD2 sobre identidad, SD1 perpendicular)
        ax.arrow(mean_rr, mean_rr, SD2_global/np.sqrt(2), SD2_global/np.sqrt(2),
                color="#FF0000", width=0.001, head_width=(max_rr-min_rr)*0.01, length_includes_head=True, zorder=6,
                label= f'SD2: {SD2_global:.3f}s')
        ax.arrow(mean_rr, mean_rr, -SD1_global/np.sqrt(2), SD1_global/np.sqrt(2),
                color="#B700FF", width=0.001, head_width=(max_rr-min_rr)*0.01, length_includes_head=True, zorder=6,
                label= f'SD1: {SD1_global:.3f}s')

        ax.set_xlabel(r'$RR_n$ (s)', size=12)
        ax.set_ylabel(r'$RR_{n+1}$ (s)', size=12)
        ax.set_title('Poincaré Plot con SD1 y SD2', size=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        result = {'SD1_global': float(SD1_global), 'SD2_global': float(SD2_global), 'per_event': per_event}
        if annotate_values:
            # Añadir texto resumen en la gráfica (puede adaptarse la posición)
            txt = f"SD1={SD1_global:.3f}s  SD2={SD2_global:.3f}s"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top')
        
        return result

    def dt_waves(self, channels:dict=None, low_freq:float=0.5, high_freq:float=40.0, order:int=5,
                 delineate_method:str='dwt', plot_waves:bool=True, tmin:float=0.0, tmax:float=None):
        """
        Detección y punteado de picos P/Q/R/S/T usando NeuroKit2 en una ventana temporal.

        Detecta picos R y puntea los picos de las ondas (P, Q, R, S, T) por canal en la ventana
        absoluta [tmin, tmax] (segundos desde el inicio del registro original). Devuelve
        un diccionario con resultados por canal que incluye índices locales, tiempos
        absolutos (s) y la señal filtrada utilizada para la delineación.

        Args:
            channels : iterable[int] | dict | None, optional
                Canales a procesar. Si es None se procesan todos los canales en `self.data`.
                (Aunque la firma actual use `dict` acepta también listas/tuplas/iterables.)
            low_freq : float, optional
                Corte inferior (Hz) para el filtrado Butterworth previo. Si es None no
                se aplica pasa-altos. Por defecto 0.5 Hz.
            high_freq : float, optional
                Corte superior (Hz) para el filtrado Butterworth previo. Si es None no
                se aplica pasa-bajos. Por defecto 40.0 Hz.
            order : int, optional
                Orden del filtro Butterworth. Por defecto 5.
            delineate_method : str, optional
                Método de delineación de ondas. Opciones disponibles (NeuroKit2):
                    - "dwt": Discrete Wavelet Transform (default, robusto).
                    - "peak": detección simple basada en picos.
                    - "cwt": Continuous Wavelet Transform.
            plot_waves : bool, optional
                Si True, genera una figura por canal mostrando la señal (en tiempo absoluto)
                y los marcadores de P/Q/R/S/T hallados. Por defecto True.
            tmin : float, optional
                Tiempo inicial ABSOLUTO en segundos (desde el inicio del registro original).
                Por defecto 0.0.
            tmax : float, optional
                Tiempo final ABSOLUTO en segundos. Si es None, se usa min_tmax() para hallar
                un tmax óptimo. Por defecto None.

        Returns:
            dict:
                Diccionario con una entrada por canal (clave = índice del canal) cuyo valor
                es otro diccionario con las siguientes claves principales:

                - 'r_peaks' : np.ndarray
                    Índices LOCALES (enteros) de picos R respecto a la `sig_clean` pasada a NeuroKit2.
                - 'r_peaks_global' : np.ndarray
                    Índices GLOBALES (muestras en la referencia del registro original): 
                    r_peaks_local + t_min_samps + first_samp.
                - 'P_peaks', 'Q_peaks', 'S_peaks', 'T_peaks' : np.ndarray
                    Índices LOCALES (enteros) detectados por `nk.ecg_delineate` (filtrados de NaN).
                - 'P_onsets','P_offsets',... (y similares) : np.ndarray
                    Índices LOCALES de onsets/offsets para cada onda (enteros).
                - 'times' : dict
                    Sub-diccionario con conversiones a segundos absolutos para cada array (p. ej.
                    'R_peaks', 'P_onsets', ...). Cada valor es un np.ndarray de tiempos en segundos
                    obtenido como (index_local + t_min_samps + first_samp) / sfreq.
                - 'delineate_raw' : dict
                    Salida "cruda" (normalizada) devuelta por `nk.ecg_delineate` para inspección.
                - 'filtered_signal' : np.ndarray
                    Ventana de la señal (1-D) usada para detección/delineado (sig_clean).

        Raises:
            ValueError:
                - Si `tmin` está fuera del rango de la señal.
                - Si, tras el clipping por duración, la ventana resultante queda vacía.
                - Si los parámetros de frecuencia son inválidos (lo chequea `self.filter`).

        Side effects / notas de estado:
            - No modifica `self.data`.
            - La función aplica un filtrado previo (si `self.is_filtered` es False) y luego llama
            a `nk.ecg_peaks` y `nk.ecg_delineate` sobre la **señal recortada**; por tanto los
            índices devueltos por NeuroKit2 son **locales** respecto a `sig_clean`.

        Notes:
            - Convenciones temporales (muy importantes):
                * `tmin`/`tmax` se interpretan como tiempos ABSOLUTOS desde el inicio del registro.
                * `t_min_samps = int(tmin * sfreq)` es la muestra global donde empieza la ventana.
                * `global_offset = t_min_samps + first_samp`.
                * índice_global = índice_local + global_offset.
                * tiempo (s) = índice_global / sfreq.
            - `nk.ecg_delineate` puede devolver NaN o floats; la función normaliza esas salidas
            eliminando NaN y redondeando a enteros para índices de muestra.
            - Si la ventana es muy corta (p. ej. < ~4 s) la detección/delineado puede fallar
            o devolver menos onsets/offsets (P/T pueden no detectarse). Por eso se fuerza una
            ventana mínima interna (`min_window_seconds = 4.0`).
            - `r_peaks` en el diccionario resultante **son índices locales**. Usá `r_peaks_global`
            o `times['R_peaks']` si querés coordenadas absolutas.
            - La función protege contra índices fuera de rango antes de indexar la señal para
            evitar errores por latidos parciales en los bordes de la ventana.

        Examples:
            >>> # Señal mono, tiempo absoluto y first_samp conocido
            >>> ecg = ECG(data=my_ecg_1d, sfreq=512.0, first_samp=1536)
            >>> res = ecg.dt_waves(tmin=30.0, tmax=40.0)   # procesa 30–40 s del registro original
            >>> # r_peaks locales y globales del canal 0
            >>> print(res[0]['r_peaks'][:10])              # índices locales (0..N-1)
            >>> print(res[0]['r_peaks_global'][:10])       # índices globales (muestras en el registro)
            >>> print(res[0]['times']['R_peaks'][:10])     # tiempos absolutos en segundos

            >>> # Si querés plotear pero la ventana original es muy corta, aumentá tmax o usa
            >>> # padding fuera de esta función para dar más contexto a la delineación.
        """
        # Preparo los datos
        data = np.asarray(self.data)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Preparo los canales
        n_channels, n_samples = data.shape
        if channels is None:
            channels = list(range(n_channels))

        # Si tmax no se especifica, lo calculamos automáticamente
        if tmax is None:
            tmax = tmin + self.min_tmax(n_beats=4, use_percentile=True, safety=1.5) 
            logging.info(f"tmax no especificado: usando tmax={tmax:.2f}s para al menos 4 latidos (percentil 95).")

        t_min_samps = int(max(0, np.floor(tmin * self.sfreq)))
        t_max_samps = int(np.ceil(tmax * self.sfreq))

        if t_min_samps >= n_samples:
            raise ValueError("tmin está fuera del rango de la señal.")
        if t_max_samps <= t_min_samps:
            raise ValueError("Ventana incorrecta tras clipping por duración de la señal.")

        results = {}
        
        for ch in channels:
            if not self.is_filtered:
                sig_clean = self.filter(low_freq=low_freq, high_freq=high_freq, order=order).data[ch, t_min_samps:t_max_samps]
            else:
                sig_clean = data[ch, t_min_samps:t_max_samps]

            peaks, info_peaks = nk.ecg_peaks(sig_clean, sampling_rate=self.sfreq)

            if 'ECG_R_Peaks' in info_peaks:
                r_peaks = np.asarray(info_peaks['ECG_R_Peaks']).astype(int)

            elif 'ECG_R_Peaks' in peaks:
                mask = np.asarray(peaks['ECG_R_Peaks']).ravel()
                r_peaks = np.where(mask!=0)[0].astype(int)
            else:
                r_peaks = np.array([], dtype=int)

            # Delineado de ondas
            delineate_out_raw = {}
            if r_peaks.size > 0:
                try:
                    delineate_out_raw = nk.ecg_delineate(sig_clean, rpeaks=r_peaks, sampling_rate=self.sfreq, method=delineate_method)
                except Exception as e:
                    logging.info(f"La delineación falló en {ch}: {e}")
                    delineate_out_raw = {}

            delineate_out = {}
            if isinstance(delineate_out_raw, dict):
                delineate_out = delineate_out_raw
            elif isinstance(delineate_out_raw, (tuple, list)) and len(delineate_out_raw) > 0:
                # Buscar el primer elemento que sea dict
                for el in delineate_out_raw:
                    if isinstance(el, dict):
                        delineate_out = el
                        break
                # Si no encontramos dict, intentar si el segundo elemento es el dict (común caso)
                if not delineate_out and len(delineate_out_raw) > 1 and isinstance(delineate_out_raw[1], dict):
                    delineate_out = delineate_out_raw[1]
            # si sigue vacío, delineate_out quedará {} y _getarr devolverá arrays vacío

            # Función interna para normalizado de keys para NeuroKit2
            def _getarr(d, key):
                """
                Normaliza la salida de nk.ecg_delineate para devolver índices enteros locales
                (respecto a la señal pasada a nk.ecg_delineate). Elimina NaN y redondea.
                """
                if not isinstance(d, dict):
                    return np.array([], dtype=int)

                val = d.get(key, np.array([], dtype=int))

                # Si es DataFrame/Series convertimos a numpy
                try:
                    import pandas as _pd
                    if isinstance(val, (_pd.Series, _pd.DataFrame)):
                        val = val.values.ravel()
                except Exception:
                    pass

                arr = np.asarray(val, dtype=float)  # puede contener NaN o floats (fracciones de muestra)
                if arr.size == 0:
                    return np.array([], dtype=int)

                # Filtrar valores finitos y convertir a índices de muestra enteros
                finite_mask = np.isfinite(arr)
                if not finite_mask.any():
                    return np.array([], dtype=int)

                arr_finite = arr[finite_mask]
                arr_idx = np.round(arr_finite).astype(int)
                return arr_idx
            
            P_peaks = _getarr(delineate_out, 'ECG_P_Peaks')
            P_onsets = _getarr(delineate_out, 'ECG_P_Onsets')
            P_offsets = _getarr(delineate_out, 'ECG_P_Offsets')

            Q_peaks = _getarr(delineate_out, 'ECG_Q_Peaks')
            Q_onsets = _getarr(delineate_out, 'ECG_Q_Onsets')
            Q_offsets = _getarr(delineate_out, 'ECG_Q_Offsets')

            R_peaks = r_peaks
            R_onsets = _getarr(delineate_out, 'ECG_R_Onsets')
            R_offsets = _getarr(delineate_out, 'ECG_R_Offsets')

            S_peaks = _getarr(delineate_out, 'ECG_S_Peaks')

            T_peaks = _getarr(delineate_out, 'ECG_T_Peaks')
            T_onsets = _getarr(delineate_out, 'ECG_T_Onsets')
            T_offsets = _getarr(delineate_out, 'ECG_T_Offsets')    

            # offset global en muestras para la ventana actual
            global_offset = int(t_min_samps) + int(self.first_samp)

            r_global = (R_peaks.astype(int) + global_offset) if R_peaks.size else np.array([], dtype=int)

            def _to_times(arr):
                if arr is None or arr.size == 0:
                    return np.array([], dtype=float)
                # arr está en índices locales respecto a sig_clean -> pasar a global y a segundos
                arr_local = np.asarray(arr, dtype=int)
                arr_global = arr_local + global_offset
                return arr_global / float(self.sfreq)

            times = {
                'P_peaks': _to_times(P_peaks),
                'P_onsets': _to_times(P_onsets),
                'P_offsets': _to_times(P_offsets),
                'Q_peaks': _to_times(Q_peaks),
                'Q_onsets': _to_times(Q_onsets),
                'Q_offsets': _to_times(Q_offsets),
                'R_peaks': _to_times(R_peaks),
                'R_onsets': _to_times(R_onsets),
                'R_offsets': _to_times(R_offsets),
                'S_peaks': _to_times(S_peaks),
                'T_peaks': _to_times(T_peaks),
                'T_onsets': _to_times(T_onsets),
                'T_offsets': _to_times(T_offsets),
            }

            results[ch] = {
                'r_peaks': R_peaks,
                'r_peaks_global': r_global,
                'P_peaks': P_peaks, 'P_onsets': P_onsets, 'P_offsets': P_offsets,
                'Q_peaks': Q_peaks, 'Q_onsets': Q_onsets, 'Q_offsets': Q_offsets,
                'R_onsets': R_onsets, 'R_offsets': R_offsets,
                'S_peaks': S_peaks,
                'T_peaks': T_peaks, 'T_onsets': T_onsets, 'T_offsets': T_offsets,
                'times': times,
                'delineate_raw': delineate_out,  # raw output
                'filtered_signal': sig_clean
            }

            if plot_waves:
                fig, ax = plt.subplots(figsize=(12, 6))

                # x axis en segundos alineada con los tiempos globales
                x_axis = (np.arange(sig_clean.size) + global_offset) / float(self.sfreq)
                ax.plot(x_axis, sig_clean, label=f'ECG Ch: {ch+1}', color="#000000", alpha=0.8, zorder=1)

                # marcar ondas: convertir índices locales -> globales usando global_offset
                for key, arr in [('R', R_peaks), ('P', P_peaks), ('Q', Q_peaks), ('S', S_peaks), ('T', T_peaks)]:
                    if arr is not None and arr.size > 0:
                        arr_global = (arr.astype(int) + global_offset)
                        # proteger índices fuera de rango
                        valid_mask = (arr >= 0) & (arr < sig_clean.size)
                        ax.scatter(arr_global[valid_mask] / float(self.sfreq), sig_clean[arr[valid_mask]],
                                label=f'Picos {key}', s=30, zorder=3)

                ax.set_xlabel('Tiempo (s)')
                ax.set_ylabel('Amplitud (µV)')
                ax.set_title(f'Picos de Ondas ECG - Canal {ch+1} - Ventana [{tmin:.1f}–{tmax:.1f}]s')

                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.show()
        
        return results

    def freq_time(self):
        pass

    def min_tmax(self, n_beats:int=3, safety:float=1.2, use_percentile:bool=True) -> float:
        """
        Calcula un tiempo máximo (tmax) recomendado para análisis de ECG.
        
        Este método estima la duración necesaria para capturar un número específico de latidos
        cardíacos, considerando la variabilidad natural del ritmo cardíaco.
        
        Args:
            n_beats: Número de latidos que se desea capturar (por defecto 3)
            safety: Factor de seguridad para acomodar variabilidad (por defecto 1.2 = 20% extra de tiempo)
            use_percentile: Si True, usa el percentil 95 para cubrir latidos más largos
        
        Returns:
            float: Duración estimada en segundos para la ventana de análisis
        
        Notes:
            - Si no hay intervalos RR disponibles, se asume un intervalo de 1.0s (60 LPM)
            - El percentil 95 es más conservador que la media para señales con arritmias
            - El factor de seguridad asegura que se capturen suficientes latidos incluso
            con variabilidad en el ritmo cardíaco
        """
        if self.rr_intervals is None:
            raise ValueError("Necesitas calcular los intervalos RR primero (peak_detection()).")

        if len(self.rr_intervals) == 0:
            # Valor por defecto para 60 LPM (1 latido por segundo)
            rr_value = 1.0
        else:
            if use_percentile:
                # Usar percentil 95 para ser conservador con latidos largos
                rr_value = np.percentile(self.rr_intervals, 95)
            else:
                rr_value = np.mean(self.rr_intervals)
        
        return rr_value * n_beats * safety
    
    def beat_in_interval(self, window:float):
        """
        Estima el número de latidos esperados en un intervalo de tiempo.
        
        Args:
            window: Duración del intervalo en segundos
            
        Returns:
            int: Número estimado de latidos cardíacos
        """
        if self.heart_rate is None:
            self._compute_heart_rate()

        if self.heart_rate == 0.0:
            return 0
        
        return int((self.heart_rate / 60.0) * window)

    def get_rr_intervals(self, return_times: bool = True):
        """
        Devuelve los intervalos R-R (segundos) y — opcionalmente — los tiempos absolutos (s)
        asociados a cada intervalo (tiempo del segundo pico del par), usando `first_samp`.

        Args:
            return_times : bool, optional
                Si True, devuelve (rr_intervals, times). Si False, devuelve solo rr_intervals.
                `times` corresponde al instante absoluto (segundos) del **segundo** pico en cada par RR:
                times[i] = (r_peaks[i+1] + first_samp) / sfreq.

        Returns:
            rr : np.ndarray (N-1,)
                Intervalos RR en segundos.
            times : np.ndarray (N-1,) (opcional)
                Tiempos absolutos (s) del segundo pico de cada intervalo, adecuados para comparar con `anotaciones`.

        Notes:
            - Si no hay suficientes picos (len(r_peaks) < 2) devuelve arrays vacíos.
            - `first_samp` se incorpora en la conversión a tiempos absolutos: r_global = r_local + int(first_samp).
        """
        if self.r_peaks is None or len(self.r_peaks) < 2:
            if return_times:
                return np.array([], dtype=float), np.array([], dtype=float)
            
            return np.array([], dtype=float)

        r = np.asarray(self.r_peaks).astype(int)
        rr = np.diff(r).astype(float) / float(self.sfreq)

        if return_times:
            r_global = r + int(self.first_samp)
            times = r_global[1:] / float(self.sfreq)

            return rr, times

        return rr
        
    def extract_segment(self, before:float=0.2, after:float=0.5, return_peak_times:bool=False):
        """
        Extrae segmentos de ECG centrados en cada pico R (latidos individuales).

        Args:
            before : float, optional
                Tiempo en segundos a incluir antes del pico R. Por defecto 0.2 s.
            after : float, optional
                Tiempo en segundos a incluir después del pico R. Por defecto 0.5 s.
            return_peak_times : bool, optional
                Si True, además de (segments, time_vector) devuelve un array con los tiempos absolutos
                (segundos) del pico R usado para cada segmento:
                    peak_time_i = (r_peak_i + first_samp) / sfreq.

        Returns:
            segments : list[np.ndarray]
                Lista de segmentos individuales (cada uno puede contener NaN si corta en los bordes).
            time_vector : np.ndarray
                Vector de tiempo centrado en 0 con longitud samp_before + samp_after.
            peak_times : np.ndarray (opcional)
                Tiempos absolutos en segundos del pico R para cada segmento (solo si return_peak_times True).

        Notes:
            - Rellena con NaN los segmentos que no puedan completar la ventana solicitada por
            estar al inicio o final de la señal.
            - Emplea `self._last_channel` si la señal es multicanal para seleccionar el canal procesado.
            - Los tiempos absolutos devueltos usan `first_samp` para mapear índices locales → globales.
        
        Examples:
            >>> # Extraer latidos para análisis morfológico
            >>> segments, time_vector = ecg.extract_segment(before=0.3, after=0.7)
            >>> # Calcular la media y desviación estándar de la morfología
            >>> mean_beat = np.nanmean(segments, axis=0)
            >>> std_beat = np.nanstd(segments, axis=0)
            
            >>> # Clasificar latidos por similitud morfológica
            >>> from sklearn.cluster import KMeans
            >>> kmeans = KMeans(n_clusters=3)
            >>> clusters = kmeans.fit_predict(np.array(segments))
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
        peak_times = []

        for r_peak in np.asarray(self.r_peaks).astype(int):

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

            # Tiempo absoluto del pico R (segundos) usando first_samp
            peak_times.append((int(r_peak) + int(self.first_samp)) / float(self.sfreq))

        time_vector = np.linspace(-before, after, samp_before+samp_after)

        if return_peak_times:
            return segments, time_vector, np.asarray(peak_times)
        
        return segments, time_vector

    def get_instantaneous_hr(self):
        """
        Devuelve la frecuencia instantánea por latido (BPM) y sus tiempos absolutos.

        Returns:
            hr : np.ndarray (N-1,)
                Frecuencia instantánea por intervalo en BPM (60 / RR[s]).
            times : np.ndarray (N-1,)
                Tiempos absolutos (s) asociados a cada HR (tiempo del segundo pico del par),
                calculados como (r_peaks[1:] + first_samp) / sfreq.

        Notes:
            - Usa `get_rr_intervals(return_times=True)` para garantizar que los tiempos absolutos
            incorporen `first_samp`.
            - Si no hay intervalos válidos devuelve arrays vacíos.
        """
        rr, times = self.get_rr_intervals(return_times=True)

        if rr.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        
        hr = 60.0 / rr
        return hr, times

    def _compute_heart_rate(self):
        """
        Método privado. Calcula la frecuencia cardíaca promedio en BPM a partir de los intervalos RR
        obtenidos por `get_rr_intervals()`.

        Returns:
            float: Frecuencia cardíaca promedio en beats por minuto (BPM).

        Side effects:
            - self.heart_rate : float
                Se almacena la frecuencia cardíaca calculada (0.0 si no hay intervalos válidos).

        Notes:
            - Este método usa `get_rr_intervals()` (que incorpora `first_samp`) para calcular
            la media de los intervalos RR en segundos y convertir a BPM.
        """
        if hasattr(self, 'rr_intervals') and len(self.rr_intervals) > 0:
            self.heart_rate = 60.0 / np.mean(self.rr_intervals)
        else:
            self.heart_rate = 0.0
        
        return self.heart_rate
    
