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

    def hilbert(self):
        """
        Función para el análisis de la envolvente de la señal EMG usando la transformada de Hilbert.
        """
        pass

    def segment(self, umbral:float):
        """
        Función para segmentar la señal EMG en períodos de contracción muscular basados en el umbral de activación.
        """
        min_value = np.min(self.data)
        max_value = np.max(self.data)

        if umbral < min_value or umbral >= max_value:
            raise ValueError("El umbral de activación debe estar entre {:.2f} y {:.2f}".format(min_value, max_value))

        mask = self.data > umbral
        data = self.data[mask]

        pass

    def freq_time():
        """
        Función para generar espectrogramas tiempo-frecuencia de la señal EMG.
        """
        pass    

    def plot_activation(self):
        """
        Función para graficar la señal EMG con las activaciones musculares resaltadas.
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
        
