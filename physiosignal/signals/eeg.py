from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import matplotlib.pyplot as plt
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

    def change_reference(self, reference:str, dic_ref:str, plot:bool=False, ch:str='Cz', tmin:int=10,
                         tmax:int=20):
        """
        Cambia la referencia de la señal EEG a promedio, canal específico o laplaciano espacial.

        Parámetros
        ----------
        reference : str
            Tipo de referencia a aplicar. Opciones: 'promedio', 'canal', 'laplaciano'.
        dic_ref : str
            Ruta al archivo JSON con la configuración de vecinos para el filtro laplaciano.
            Para 'canal' y 'promedio' puede ser None.
        plot : bool, opcional
            Si True, muestra gráficos comparativos y/o mapas topográficos según la referencia.
        ch : str, opcional
            Nombre del canal de referencia (solo para reference='canal'). Por defecto 'Cz'.
        tmin : int, opcional
            Tiempo inicial en segundos para graficar (solo para 'canal' y 'promedio'). Por defecto 10.
        tmax : int, opcional
            Tiempo final en segundos para graficar (solo para 'canal' y 'promedio'). Por defecto 20.

        Notas
        -----
        - Para 'laplaciano', se requiere un archivo JSON con los vecinos de cada canal.
        - Si plot=True y hay eventos, se muestran mapas topográficos y waveforms ERP.
        - Si plot=True y no hay eventos, se muestran mapas topográficos continuos por bandas de frecuencia.

        Raises
        ------
        ValueError
            Si el tipo de referencia no es válido o los parámetros de tiempo son incorrectos.

        Ejemplo
        -------
        >>> eeg.change_reference('laplaciano', 'vecinos.json', plot=True)
        >>> eeg.change_reference('canal', None, plot=True, ch='Cz', tmin=0, tmax=10)
        """
        import json

        reference = reference.lower()
        n_channels, n_samps = self.data.shape

        if reference not in ['canal', 'promedio', 'laplaciano']:
            raise ValueError("El cambio de referencia solo puede ser de tipo ['canal', 'promedio', 'laplaciano'].")

        if reference == 'canal':   
            ref_data = self.data[self.info.ch_names.index(ch), :] # Canal de referencia
            new_data = self.data - ref_data[None, :] # Nueva referencia para todos los canales ref[None, :] inserta una nueva dimension

            self.data_canal = new_data
            self.reference = ch

            if plot:
                if tmin >= tmax or tmin < 0:
                  raise ValueError(f"tmin >= tmax o tmin < 0")
                
                tmin_samps = int(tmin * self.sfreq) if tmin > 0 else 0
                tmax_samps = int(n_samps/self.sfreq) if tmax > (n_samps/self.sfreq) else tmax * self.sfreq
                crop_t = np.arange(tmin_samps, tmax_samps) / self.sfreq

                ch_reference = ['Fp1', 'Cz', 'Pz', 'Oz'] # Frontal, Central, Parietal, Occipital

                fig, ax = plt.subplots(4, 1, figsize=(7, 10))

                for i, chs in enumerate(ch_reference):

                    # Indice del canal
                    ch_idx = self.info.ch_names.index(chs)

                    # Señal original
                    ax[i].plot(crop_t, self.data[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Original', color="#0800FF")
                    
                    # Señal luego de cambio de referencia
                    ax[i].plot(crop_t, self.data_canal[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Referenciado', color="#FF0000")

                    ax[i].axhline(y=0, color="#373737", linestyle='--', alpha=0.7)

                    # Titulos
                    ax[i].set_ylabel(f'{chs} (µV)')
                    ax[i].legend(loc='upper right')
                
                ax[-1].set_xlabel('Tiempo (s)')
                plt.suptitle(f'Comparación antes y después de referencia a {ch}')
                plt.tight_layout()
                plt.show()

        elif reference == 'laplaciano':

            with open(dic_ref, "r") as f:
                ref_dic = json.load(f)

            channels = [ch.upper() for ch in ref_dic]
            self.info.ch_names = channels

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
            self.reference = 'laplaciano'

            if plot:
                import mne
                from scipy import signal

                # Diccionario para renombrar canales a formato estándar MNE
                dic = {'FP1':'Fp1', 'FPZ':'Fpz', 'FP2':'Fp2', 
                        'FZ':'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 
                        'POZ':'POz', 'OZ':'Oz'}
                
                channels = self.info.rename_channels(dic).ch_names

                # Montaje estándar de MNE
                montage = mne.channels.make_standard_montage('standard_1005')

                mne_info = mne.create_info(ch_names=channels, sfreq=self.sfreq, ch_types=self.info.ch_types)

                mne_info.set_montage(montage)

                # Convierto las anotaciones propias a eventos MNE
                events, event_id = self._from_ann_to_events()
                # logging.debug(f"Eventos encontrados: {events}, ID eventos: {event_id}")

                if len(events) == 0:
                    logging.info("No existen eventos dentro de las anotaciones. Se muestran los datos continuos")

                    freq_bands = {
                            'Delta': (1, 4),
                            'Theta': (4, 8),
                            'Alpha': (8, 13),
                            'Beta': (13, 30),
                            'Gamma': (30, 45)
                        }
                    
                    fig, ax = plt.subplots(2, 3, figsize=(15,10))
                    ax = ax.ravel()

                    for idx, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                        # Filtro los datos en la banda de frecuencia
                        nyquist = self.sfreq / 2

                                            # Orden, acote de freq, tipo
                        b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
                        filtered_data = signal.filtfilt(b, a, self.data_laplaciano, axis=1)

                        # Hallo la potencia RMS para cada canal
                        power = np.sqrt(np.mean(filtered_data**2, axis=1))
                        
                        # Genero el mapa topográfico para cada frecuencia
                        im, _ = mne.viz.plot_topomap(power, mne_info, axes=ax[idx], show=False, sensors=True, 
                                                    contours=7, cmap='RdBu_r', res=64)

                        ax[idx].set_title(f'{band_name} ({low_freq}-{high_freq}) Hz')

                    cbar = fig.colorbar(im, ax=ax[-1])
                    cbar.set_label(r'Potencia ($µV^2$)')

                    if len(freq_bands) < 6:
                        fig.delaxes(ax[-1])
                    
                    plt.suptitle('Mapas Topográficos por Bandas de Frecuencia (Laplaciano)')
                    plt.tight_layout()
                    plt.show()
                
                else:
                    # Si hay eventos, creo objeto Rae de MNE con los datos laplacianos
                    raw = mne.io.RawArray(self.data_laplaciano, mne_info) # Objeto Raw MNE

                    # Ventana temporal para los epochs
                    tmin = -0.2  # 200 ms antes del evento
                    tmax = 0.8   # 800 ms después del evento
                    baseline = (-0.2, 0)  # Período de línea base

                    # Extraigo los epochs alrededor de los eventos
                    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                                        baseline=baseline, preload=True, verbose=False)
                    
                    # Filtro por epocas: 1 es Derecha, 2 es Izquierda
                    epochs=epochs[events[:,2]==1]
                    
                    # Hallo el promedio (ERP) de los epochs
                    erp = epochs.average()

                    # Tiempos de interés para los mapas topográficos
                    time_points = [0.1, 0.2, 0.3, 0.4] # 100ms, 200ms...

                    n_times = len(time_points) 
                    n_cols = min(2, n_times)
                    n_rows = (n_times + n_cols -1) // n_cols

                    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

                    # Aseguro que ax sea siempre una lista para iterar
                    if n_times == 1:
                        ax = [ax]
                    else:
                        ax = ax.ravel()

                    # Itero sobre cada punto temporal de interés y genero el topomap
                    for idx, time_point in enumerate(time_points):
                        if idx < len(ax):
                            time_idx = np.argmin(np.abs(erp.times - time_point))

                            # Hallo las amplitudes de los canales en ese tiempo
                            amplitudes = erp.data[:, time_idx]

                            # Genero el topomap para ese tiempo
                            im, _ = mne.viz.plot_topomap(amplitudes, erp.info, axes=ax[idx],
                                                         show=False, contours=7, cmap='RdBu_r', sensors=True)
                            
                            ax[idx].set_title(f'{time_point*1000:.0f} ms')
                    
                    cbar = fig.colorbar(im, ax=ax[-1], location='right', shrink=0.9) # shrink: multiplicador de la colorbar
                    cbar.set_label('Amplitud (µV)')

                    # Elimino los ejes que sobren
                    for j in range(idx+1, len(ax)):
                        fig.delaxes(ax[j])

                    plt.suptitle('Mapas Topográficos de Amplitud ERP en Tiempos Específicos')
                    plt.tight_layout()
                    plt.show()

                    # Segundo grtáfico: Waveforms de canales específicos
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    
                    # Canales de interés
                    channels_of_interest = ['C3', 'C4', 'C5','C6']
                    
                    for ch in channels_of_interest:
                        if ch in erp.ch_names:
                            ch_idx = erp.ch_names.index(ch)
                            ax2.plot(erp.times, erp.data[ch_idx], label=ch, linewidth=2)
                    
                    # Marco los tiempos de interés en la gráfica
                    for t in time_points:
                        ax2.axvline(t, color='r', linestyle='--', alpha=0.7)

                        # Agrega el texto al lado de la línea
                        ax2.text(t + 0.01, ax2.get_ylim()[1]*0.9, f'{int(t*1000)} ms', color='black', fontsize=10)

                    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
                    ax2.axvline(0, color='k', linestyle='-', alpha=0.5)
                    ax2.set_xlabel('Tiempo (s)')
                    ax2.set_ylabel('Amplitud (µV)')
                    ax2.legend()
                    ax2.set_title('Waveforms ERP para Canales de Interés')
                    
                    plt.tight_layout()
                    plt.show()

        elif reference == "promedio":

            ch_prom = np.mean(self.data, axis=0)
            avg_ref = self.data - ch_prom

            self.avg_ref = avg_ref
            self.reference = 'promedio'

            if plot:
                if tmin >= tmax or tmin < 0:
                  raise ValueError(f"tmin >= tmax o tmin < 0")
                
                tmin_samps = int(tmin * self.sfreq) if tmin > 0 else 0
                tmax_samps = int(n_samps/self.sfreq) if tmax > (n_samps/self.sfreq) else tmax * self.sfreq
                crop_t = np.arange(tmin_samps, tmax_samps) / self.sfreq

                ch_reference = ['Fp1', 'Cz', 'Pz', 'Oz'] # Frontal, Central, Parietal, Occipital

                fig, ax = plt.subplots(4, 1, figsize=(7, 10))

                for i, chs in enumerate(ch_reference):

                    # Indice del canal
                    ch_idx = self.info.ch_names.index(chs)

                    # Señal original
                    ax[i].plot(crop_t, self.data[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Original', color="#0800FF")
                    
                    # Señal luego de cambio de referencia
                    ax[i].plot(crop_t, self.avg_ref[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Referenciado', color="#FF0000")

                    ax[i].axhline(y=0, color="#000000", linestyle='--', alpha=0.7)

                    # Titulos
                    ax[i].set_ylabel(f'{chs} (µV)')
                    ax[i].legend(loc='upper right')
                
                ax[-1].set_xlabel('Tiempo (s)')
                plt.suptitle('Comparación temporal: Original vs Referencia al Promedio')
                plt.tight_layout()
                plt.show()

    def fft(self):
        pass

    def freq_time(self):
        pass

    def hilbert(self):
        pass

    def _from_ann_to_events(self):
        """Convierte las anotaciones propias a eventos en formato MNE"""

        events, event_id = [], {}
        current_id = 1

        for onset, duration, description in zip(self.anotaciones.onset, self.anotaciones.duration, self.anotaciones.description):

            samples = int(onset * self.sfreq)

            if description not in event_id:
                event_id[description] = current_id
                current_id += 1

            events.append([samples, 0, event_id[description]])

        return np.array(events), event_id