from __future__ import annotations

from .raw import RawSignal
from physiosignal.info import Info, Annotations
from physiosignal.logger import log_config

import logging
import matplotlib.pyplot as plt
import numpy as np
import mne

def _normalize_ch_list(ch_names, montage='standard_1005'):
    """
    Normaliza una lista de nombres de canales EEG para que coincidan con los nombres canónicos 
    del montaje MNE especificado.

    Parameters:
        ch_names : list of str
            Lista de nombres de canales a normalizar.
        montage : str, optional
            Nombre del montaje estándar de MNE a utilizar para la normalización. 
            Por defecto 'standard_1005'.

    Returns:
        tuple: (normalized_list, rename_map)
            normalized_list : list of str
                Lista de nombres de canales normalizados.
            rename_map : dict
                Diccionario que mapea nombres originales a nombres canónicos {old_name: canonical_name}.

    Notes:
        La función utiliza una comparación insensible a mayúsculas/minúsculas y caracteres especiales
        como guiones, puntos y espacios.
        Si dos canales distintos mapearían al mismo nombre canónico, solo el primero se renombra
        y el segundo mantiene su nombre original para evitar duplicados.
        Los nombres de canales que no coinciden con ningún nombre canónico del montaje se mantienen
        sin cambios.

    Examples:
        >>> normalized, mapping = _normalize_ch_list(['Fp1', 'fp2', 'C3'], 'standard_1005')
        >>> print(normalized)
        ['Fp1', 'Fp2', 'C3']
        >>> print(mapping)
        {'fp2': 'Fp2'}
    """
    def _key(s):
        return s.lower().replace('-', '').replace('.', '').replace(' ', '')

    # Obtengo nombres canónicos del montaje
    mont = mne.channels.make_standard_montage(montage)
    canonical = mont.ch_names

    # Lookup para comparación segura
    lookup = { _key(c): c for c in canonical }

    normalized = []
    rename_map = {}
    used_targets = set()

    ch_names = ch_names if isinstance(ch_names, list) else [ch_names]

    for ch in ch_names:
        k = _key(ch)
        if k in lookup:
            target = lookup[k]
            if target in used_targets:
                # Conflicto: ya usamos ese canonical antes -> no renombramos este canal
                print(f"[normalize_ch_list] Conflicto: '{ch}' mapearía a '{target}', "
                      "pero ya está ocupado. Se mantiene el nombre original.")
                normalized.append(ch)
            else:
                # Renombro al canonical exacto
                normalized.append(target)
                if target != ch:
                    rename_map[ch] = target
                used_targets.add(target)
        else:
            # No hay match con el montaje -> dejamos el nombre como estaba
            normalized.append(ch)

    return normalized, rename_map

class EEG(RawSignal):
    """
    Representa una señal de Electroencefalografía (EEG) con utilidades comunes para
    preprocesado, referencia, análisis espectral y visualización.

    Al inicializar, normaliza los nombres de canales al estándar MNE ('standard_1005'),
    convierte anotaciones a eventos compatibles con MNE y configura el logger interno.
    Si se pasa un objeto `RawSignal` mediante el parámetro `raw`, la inicialización
    ejecuta la lógica base de `RawSignal.__init__` sobre copias profundas de sus
    atributos para garantizar que la ruta de inicialización sea idéntica a la de
    pasar los atributos individualmente y para evitar mutaciones sobre el objeto
    `raw` original.

    Key Features:
        - Cambio de referencia (por canal, promedio, laplaciano).
        - Filtro laplaciano espacial a partir de una configuración de vecinos.
        - Cálculo y visualización de espectro (FFT) usando Welch.
        - Análisis tiempo-frecuencia (wavelets de Morlet) y aplicación de baseline.
        - Transformada de Hilbert para obtener señal analítica y envolvente.
        - Conversión de anotaciones internas a eventos compatibles con MNE.
        - Funciones de visualización: waveforms, topomaps y gráficas tiempo-frecuencia.

    Attributes:
        data : np.ndarray
            Señal EEG cruda con forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Metadatos de canales. Los nombres se normalizan al montaje MNE
            por defecto ('standard_1005') durante la inicialización (sobre una copia
            para no mutar objetos externos).
        anotaciones : Annotations
            Anotaciones de eventos temporales (onset, duration, description).
        first_samp : int
            Índice de la primera muestra respecto al inicio original. Se conserva el
            valor provisto por `raw` si la instancia se crea con `raw=...`.
        data_ref : np.ndarray, opcional
            Señal referenciada (resultado de aplicar channel_reference, mean_reference o laplacian_filter).
            Varios métodos (por ejemplo `fft`, `freq_time`, `hilbert`) esperan que `data_ref` exista.
        reference : str
            Tipo de referencia actualmente aplicada ('promedio', 'canal', 'laplaciano', etc.).
        events : np.ndarray, opcional
            Matriz de eventos en formato MNE ([muestra, 0, código]) generada por la inicialización.
            Se crea automáticamente a partir de `anotaciones` y `first_samp` para garantizar
            consistencia en todas las rutas de creación del objeto.
        event_id : dict, opcional
            Diccionario de mapeo {descripcion: codigo} generado desde las anotaciones.
        fft_psd : np.ndarray, opcional
            Resultado de PSD (dB) tras ejecutar `fft()`. Se crea y guarda en `self.fft_psd` por el método `fft`.
        fft_freq : np.ndarray, opcional
            Vector de frecuencias asociado a `fft_psd`. Se crea y guarda en `self.fft_freq` por el método `fft`.

    Notes:
        - Muchas funciones de visualización usan matplotlib; las llamadas a `plot` muestran
        figuras en pantalla (plt.show()).
        - La normalización de nombres de canales utiliza comparaciones insensibles a
        mayúsculas/minúsculas y elimina caracteres como '-', '.' y espacios; se aplica
        sobre una copia del `info` para evitar modificar el `raw` original.
        - Si se inicializa con `raw=...`, la clase llamará a `super().__init__` con copias
        profundas de `raw.data`, `raw.sfreq`, `raw.info`, `raw.anotaciones` y `raw.first_samp`
        para garantizar que la lógica de `RawSignal.__init__` se ejecute siempre.
        - Antes de ejecutar métodos que requieren referencia (por ejemplo `fft`, `freq_time`,
        `hilbert`) se debe haber calculado y guardado `data_ref` mediante `channel_reference`,
        `mean_reference` o `laplacian_filter`; de lo contrario se lanzará un ValueError.
        - Los métodos que emplean eventos dependen de las anotaciones provistas; si se modifican
        las anotaciones tras la inicialización, llamá a `refresh_events()` para regenerar
        `self.events` y `self.event_id`.
        - Para compatibilidad con MNE se utiliza por defecto el montaje 'standard_1005'.
        - Para reducir salida de log informativa de MNE, ejecutar antes:
        
            mne.set_log_level('ERROR')

    Examples:
        >>> # Instanciación desde atributos (asegurarse de pasar first_samp si es relevante)
        >>> eeg = EEG(data=my_data, sfreq=250.0, info=my_info, anotaciones=my_annots, first_samp=my_first_samp)
        >>>
        >>> # Instanciación desde un objeto RawSignal (equivalente a pasar atributos)
        >>> eeg = EEG(raw=my_rawsignal)
        >>>
        >>> # Referencia al promedio y cálculo de FFT
        >>> eeg.mean_reference(plot=False)
        >>> eeg.fft(pick_channel='Cz', band='Alpha', plot=True)
        >>>
        >>> # Análisis tiempo-frecuencia por evento
        >>> power = eeg.freq_time(low_freq=1, high_freq=40, channels=['C3','C4'], separate_events=False)
        >>>
        >>> # Hilbert sobre un segmento
        >>> analytic, env = eeg.hilbert(channels='Cz', freq_band=(8,12), plot=True)
    """
    
    def __init__(self, raw:RawSignal=None, data:np.ndarray=None, sfreq:float=None, info:Info=None, anotaciones:Annotations=None, 
                 first_samp:int=0, see_log:bool=True, reference:str='promedio'):
        """
        Inicializa una instancia de EEGSignal.

        Comportamiento:
            - Si se pasa `raw` (objeto RawSignal), la inicialización ejecuta siempre la
            lógica base de `RawSignal.__init__` llamando a `super().__init__` con copias
            profundas (`deepcopy`) de los atributos de `raw` (data, sfreq, info,
            anotaciones y first_samp). Esto preserva `first_samp` y evita mutar el objeto
            `raw` original.
            - Si `raw` es None, se inicializa mediante `super().__init__` usando los
            parámetros `data`, `sfreq`, `info`, `anotaciones` y `first_samp` como antes.
            - Después de la inicialización base se normalizan los nombres de canales sobre
            la copia de `info` y se generan `events` y `event_id` internamente para
            garantizar consistencia entre ambas rutas de creación del objeto.

        Parameters:
            raw : RawSignal, optional
                Objeto RawSignal desde el cual inicializar la instancia. Si se proporciona,
                tiene prioridad frente a `data`, `sfreq`, `info` y `anotaciones`.
            data : np.ndarray, optional
                Matriz (n_canales, n_muestras) con la señal EEG cruda (usado si raw is None).
            sfreq : float, optional
                Frecuencia de muestreo en Hz. Si None y raw es None, se intenta usar info.sfreq.
            info : Info, optional
                Metadatos de canales (usado si raw is None).
            anotaciones : Annotations, optional
                Eventos temporales asociados (usado si raw is None).
            first_samp : int, optional
                Índice de la primera muestra respecto del registro original (usado si raw is None).
                Si se pasa `raw`, el `first_samp` empleado será `raw.first_samp`.
            see_log : bool, optional
                Activa/desactiva logging interno.
            reference : str, optional
                Tipo de referencia inicial ('promedio', 'canal', 'laplaciano', etc.).

        Notes:
            - El uso de `deepcopy` al inicializar desde `raw` garantiza que las modificaciones
            internas (por ejemplo renombrado de canales) no afecten al objeto `raw` que se
            pasó como entrada. Esto tiene un coste de memoria; si se desea evitar la copia
            profunda, ajustar la implementación con precaución.
            - La normalización de nombres de canales y la generación de `events` se realizan
            inmediatamente en la inicialización para que `events`, `event_id`, `info`
            y `first_samp` sean consistentes y reproducibles tanto si la instancia se
            creó desde `raw` como desde atributos individuales.
            - Si modificás `self.anotaciones` tras la inicialización, llamá a `refresh_events()`
            para regenerar `self.events` y `self.event_id` (no usar `_from_ann_to_events()` desde fuera).
        """
        from copy import deepcopy
        if raw is not None:
            # Inicializamos siempre llamando a super().__init__ para mantener la lógica de RawSignal
            super().__init__(deepcopy(raw.data),
                            deepcopy(raw.sfreq),
                            deepcopy(raw.info),
                            deepcopy(raw.anotaciones),
                            deepcopy(raw.first_samp),
                            see_log)
        else:
            super().__init__(data, sfreq, info, anotaciones, first_samp, see_log)

        self.reference = reference

        # Configuración inicial del logger
        log_config(see_log)  
        self._logger = logging.getLogger(__name__) # __name__ toma el nombre del submódulo

        # Normalizo los nombres de los canales a formato estándar MNE
        n_original = len(self.info.ch_names)
        normalized, rename_map = _normalize_ch_list(self.info.ch_names, montage='standard_1005')
        self.info.ch_names = normalized

        # Convierto las anotaciones propias a eventos MNE
        self.events, self.event_id = self._from_ann_to_events()

        if rename_map:
            logging.info(f"Se aplicaron renombrados (original -> canónico):")
            if len(rename_map) == n_original:
                logging.info(f"Todos los canales fueron renombrados al estándar MNE. La nueva lista es: {self.info.ch_names}")
            else:
                for old_name, new_name in rename_map.items():
                    logging.info("  %s -> %s", old_name, new_name)
                
    def channel_reference(self, ch:str='Cz', plot:bool=False, tmin:int=10, tmax:int=20, ch_reference:list|str=['Fp1', 'Cz', 'Pz', 'Oz']):
        """
        Aplica referencia a un canal específico en la señal EEG.

        Args:
            ch : str, optional
                Nombre del canal de referencia. Por defecto 'Cz'.
            plot : bool, optional
                Si True, muestra la comparación gráfica entre la señal original y la referenciada.
            tmin : int, optional
                Tiempo inicial en segundos para la visualización. Por defecto 10.
            tmax : int, optional
                Tiempo final en segundos para la visualización. Por defecto 20.
            ch_reference : list or str, optional
                Lista o nombre de canales a mostrar en la gráfica. Por defecto ['Fp1', 'Cz', 'Pz', 'Oz'].

        Raises:
            ValueError
                Si tmin >= tmax, tmin < 0, o el canal de referencia no existe.

        Notes:
            La señal referenciada se guarda en el atributo `data_canal` y la referencia actual en `reference`.

        Examples:
            >>> eeg.channel_reference(ch='Cz', plot=True)
            >>> eeg.channel_reference(ch='Pz', plot=True, tmin=0, tmax=15, ch_reference=['Cz', 'Pz'])
        """
        n_channels, n_samps = self.data.shape
        ch, _ = _normalize_ch_list(ch, montage='standard_1005')

        ref_data = self.data[self.info.ch_names.index(ch[0]), :] # Canal de referencia
        new_data = self.data - ref_data[None, :] # Nueva referencia para todos los canales ref[None, :] inserta una nueva dimension

        self.data_ref = new_data
        self.reference = 'canal'

        if plot:
            if tmin >= tmax or tmin < 0:
                raise ValueError(f"tmin >= tmax o tmin < 0")
            
            tmin_samps = int(tmin * self.sfreq) if tmin > 0 else 0
            tmax_samps = int(n_samps/self.sfreq) if tmax > (n_samps/self.sfreq) else tmax * self.sfreq
            crop_t = np.arange(tmin_samps, tmax_samps) / self.sfreq

            normalized, _ = _normalize_ch_list(ch_reference, montage='standard_1005')

            if all(ch in self.info.ch_names for ch in normalized):
                ch_reference = _normalize_ch_list(ch_reference, montage='standard_1005')[0]
            else:
                raise ValueError(f"Uno o más canales en ch_reference no existen en la señal.")

            fig, ax = plt.subplots(4, 1, figsize=(7, 10))

            for i, chs in enumerate(ch_reference):
                
                # Indice del canal
                ch_idx = self.info.ch_names.index(chs)

                # Señal original
                ax[i].plot(crop_t, self.data[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Original', color="#0800FF")
                
                # Señal luego de cambio de referencia
                ax[i].plot(crop_t, self.data_ref[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Referenciado', color="#FF0000")

                ax[i].axhline(y=0, color="#373737", linestyle='--', alpha=0.7)

                # Titulos
                ax[i].set_ylabel(f'{chs} (µV)')
                ax[i].legend(loc='upper right')
                ax[i].grid(True, alpha=0.5)
            
            ax[-1].set_xlabel('Tiempo (s)')
            plt.suptitle(f'Comparación antes y después de referencia a {ch}')
            plt.tight_layout()
            plt.show()

    def mean_reference(self, plot:bool=False, tmin:int=10, tmax:int=20, ch_reference:list|str=['Fp1', 'Cz', 'Pz', 'Oz']):
        """
        Aplica referencia al promedio de todos los canales en la señal EEG.

        Args:
            plot : bool, optional
                Si True, muestra la comparación gráfica entre la señal original y la referenciada.
            tmin : int, optional
                Tiempo inicial en segundos para la visualización. Por defecto 10.
            tmax : int, optional
                Tiempo final en segundos para la visualización. Por defecto 20.
            ch_reference : list or str, optional
                Lista o nombre de canales a mostrar en la gráfica. Por defecto ['Fp1', 'Cz', 'Pz', 'Oz'].

        Raises:
            ValueError
                Si tmin >= tmax o tmin < 0.

        Notes:
            La señal referenciada se guarda en el atributo `avg_ref` y la referencia actual en `reference`.

        Examples:
            >>> eeg.mean_reference(plot=True)
            >>> eeg.mean_reference(plot=True, tmin=0, tmax=15, ch_reference=['Cz', 'Pz'])
        """
        n_channels, n_samps = self.data.shape

        ch_prom = np.mean(self.data, axis=0)
        avg_ref = self.data - ch_prom

        self.data_ref = avg_ref
        self.reference = 'promedio'

        if plot:
            if tmin >= tmax or tmin < 0:
                raise ValueError(f"tmin >= tmax o tmin < 0")
            
            tmin_samps = int(tmin * self.sfreq) if tmin > 0 else 0
            tmax_samps = int(n_samps/self.sfreq) if tmax > (n_samps/self.sfreq) else tmax * self.sfreq

            crop_t = np.arange(tmin_samps, tmax_samps) / self.sfreq # Eje temporal acotado

            normalized, _ = _normalize_ch_list(ch_reference, montage='standard_1005')

            if all(ch in self.info.ch_names for ch in normalized):
                ch_reference = _normalize_ch_list(ch_reference, montage='standard_1005')[0]
            else:
                raise ValueError(f"Uno o más canales en ch_reference no existen en la señal.")

            fig, ax = plt.subplots(4, 1, figsize=(7, 10))

            for i, chs in enumerate(ch_reference):

                # Indice del canal
                ch_idx = self.info.ch_names.index(chs)

                # Señal original
                ax[i].plot(crop_t, self.data[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Original', color="#0800FF")
                
                # Señal luego de cambio de referencia
                ax[i].plot(crop_t, self.data_ref[ch_idx, tmin_samps:tmax_samps], alpha=0.6, label=f'{chs} Referenciado', color="#FF0000")

                ax[i].axhline(y=0, color="#000000", linestyle='--', alpha=0.7)

                # Titulos
                ax[i].set_ylabel(f'{chs} (µV)')
                ax[i].legend(loc='upper right')
                ax[i].grid(True, alpha=0.5)
            
            ax[-1].set_xlabel('Tiempo (s)')
            plt.suptitle('Comparación temporal: Original vs Referencia al Promedio')
            plt.tight_layout()
            plt.show()

    def laplacian_filter(self, dic_ref:str, topomap:bool=False, channels_of_interest:list|str=['Fp1', 'Cz', 'Pz', 'Oz'],
                         t_after_event:float=0.8, t_previous_event:float=-0.2, event_index:int=2, 
                         time_points:list[float]=[0.1, 0.2, 0.3, 0.4], waveform:bool=False):
        """
        Aplica filtro laplaciano espacial a la señal EEG.

        Args:
            dic_ref : str
                Ruta al archivo JSON con la configuración de vecinos para el filtro laplaciano.
            plot : bool, optional
                Si True, muestra visualizaciones (mapas topográficos y/o waveforms).
            channels_of_interest : list or str, optional
                Canales a mostrar en waveforms. Por defecto ['Fp1', 'Cz', 'Pz', 'Oz'].
            t_after_event : float, optional
                Tiempo después del evento para análisis (segundos). Por defecto 0.8.
            t_previous_event : float, optional
                Tiempo antes del evento para análisis (segundos). Por defecto -0.2.
            event_index : int, optional
                ID del evento a analizar. Por defecto 2.
            time_points : list of float, optional
                Tiempos específicos para visualización (segundos). Por defecto [0.1, 0.2, 0.3, 0.4].
            waveform : bool, optional
                Si True, muestra waveforms ademas de mapas topográficos.

        Returns:
            None

        Raises:
            ValueError: Si el archivo JSON no existe o parámetros son inválidos.

        Notes:
            - Aplica filtro laplaciano usando configuración de vecinos del JSON.
            - El método **usa** `self.events` y `self.event_id` generados en la inicialización.
            Por tanto, `events` reflejarán el `first_samp` empleado al crear la instancia.
            - Si `self.events` está vacío, el método muestra mapas topográficos de potencia
            RMS sobre los datos continuos.
            - Si `waveform=True` y hay eventos: muestra waveforms de canales seleccionados.
            - Los resultados se guardan en `data_ref` y se actualiza `reference`.
            - Si modificás las anotaciones tras la inicialización, llamá a `refresh_events()`
            para regenerar `self.events` antes de ejecutar este método.

        Examples:
            >>> # Filtro sin visualización
            >>> eeg.laplacian_filter('vecinos.json')
            
            >>> # Filtro con visualización completa
            >>> eeg.laplacian_filter('vecinos.json', plot=True, waveform=True)
            
            >>> # Filtro visualizando canales específicos
            >>> eeg.laplacian_filter('vecinos.json', plot=True, channels_of_interest=['Cz', 'Pz'])
        """
        import json
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

        self.data_ref = laplace
        self.reference = 'laplaciano'

        # Montaje estándar de MNE
        montage = mne.channels.make_standard_montage('standard_1005')
        mne_info = mne.create_info(ch_names=self.info.ch_names, sfreq=self.sfreq, ch_types=self.info.ch_types)
        mne_info.set_montage(montage)

        # Si hay eventos, creo objeto Raw de MNE con los datos laplacianos
        raw = mne.io.RawArray(self.data_ref, mne_info)

        # Validación de parámetros de tiempo
        if t_after_event <= 0 or t_previous_event >= 0 or t_after_event <= abs(t_previous_event):
            raise ValueError("Parámetros de tiempo inválidos. Asegúrese que t_after_event > 0, t_previous_event < 0 y t_after_event > |t_previous_event|")
        
        tmin = t_previous_event
        tmax = t_after_event
        baseline = (t_previous_event, 0.0)  # Período de línea base

        # Extraigo los epochs alrededor de los eventos
        epochs = mne.Epochs(raw, self.events, event_id=self.event_id, tmin=tmin, tmax=tmax, 
                            baseline=baseline, preload=True, verbose=False, picks='eeg')
        self.epochs = epochs
        
        # Filtro por epocas del evento seleccionado
        epochs=epochs[self.events[:,2] == event_index]
        
        # Hallo el promedio (ERP) de los epochs
        erp = epochs.average()
        
        if topomap:
            self._topomap(erp, self.events, mne_info, self.event_id, event_index, time_points)

        if waveform and erp is not None:
            self._plot_waveform(erp, self.events, self.event_id, channels_of_interest, time_points, event_index)

    def fft(self, pick_channel:list[str]|str='Cz', band:list[str]|str='Alpha', plot:bool=True, low_freq:float=None, high_freq:float=None):
        """
        Calcula la Transformada Rápida de Fourier (FFT) usando el método de Welch para la señal EEG y permite 
        visualizar el espectro de potencia para canales y bandas de frecuencia específicos.

        Args:
            pick_channel : list or str, optional
                Lista o nombre de canales a analizar. Por defecto 'Cz'.
            band : list or str, optional
                Lista o nombre de bandas de frecuencia predefinidas a visualizar. 
                Opciones: 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'. Por defecto 'Alpha'.
            plot : bool, optional
                Si True, muestra gráficos del espectro de potencia. Por defecto True.
            low_freq : float, optional
                Límite inferior de frecuencia personalizado (Hz). Si se especifica junto con high_freq, 
                ignora las bandas predefinidas. Por defecto None.
            high_freq : float, optional
                Límite superior de frecuencia personalizado (Hz). Si se especifica junto con low_freq, 
                ignora las bandas predefinidas. Por defecto None.

        Raises:
            ValueError
                Si no se ha aplicado previamente un método de referencia, si los canales especificados no existen,
                o si las bandas de frecuencia no son válidas.

        Notes:
            El espectro de potencia se calcula usando el método de Welch y se convierte a escala logarítmica (dB).
            Los resultados se guardan en los atributos `fft_psd` (densidad espectral de potencia) y `fft_freq` (frecuencias).
            Si se especifican low_freq y high_freq, se ignora el parámetro band y se usa el rango de frecuencia personalizado.
            El gráfico muestra el espectro de potencia suavizado para cada canal seleccionado.

        Examples:
            >>> eeg.fft(pick_channel='Cz', band='Alpha')
            >>> eeg.fft(pick_channel=['Cz', 'Pz'], band=['Alpha', 'Beta'])
            >>> eeg.fft(pick_channel='Oz', low_freq=8.0, high_freq=12.0)
        """
        from scipy.signal import welch
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if not hasattr(self, 'data_ref'):
            raise ValueError("Primero debe aplicar un método de referencia (canal, promedio, laplaciano) antes de calcular la FFT.")
        
        data = np.atleast_2d(self.data_ref)
        pick_channel, _ = _normalize_ch_list(pick_channel, montage='standard_1005')

        if isinstance(pick_channel, str):
            if pick_channel not in self.info.ch_names:
                raise ValueError(f"El canal {pick_channel} no existe en la señal.")
            
            ch_idx = [self.info.ch_names.index(pick_channel)]
            channel_names = [pick_channel]

        elif isinstance(pick_channel, list):
            for ch in pick_channel:
                if ch not in self.info.ch_names:
                    raise ValueError(f"El canal {ch} no existe en la señal.")
                
            ch_idx = [self.info.ch_names.index(ch) for ch in pick_channel]
            channel_names = pick_channel

        data_ch = data[ch_idx, :]
        freqs, psd = welch(data_ch, fs=self.sfreq, nperseg=1024, axis=1)

        self.fft_psd = 10 * np.log10(psd + 1e-12)
        self.fft_freq = freqs

        if plot:
            freq_bands = {
                'Delta': (1, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 45)
            } 

            # Determino si se están usando bandas predefinidas o personalizadas
            using_custom_band = low_freq is not None and high_freq is not None # Bool: True o False
            
            if using_custom_band:
                # Uso frecuencias personalizadas
                band_list = [('Personalizado', low_freq, high_freq)]
            else:
                # Uso bandas predefinidas
                band = [band.capitalize()] if isinstance(band, str) else [b.capitalize() for b in band]
                band_list = []
                
                for b in band:
                    if b not in freq_bands:
                        raise ValueError(f"La banda {b} no es válida. Use: {list(freq_bands.keys())}")
                    # Tupla con (nombre, low_freq, high_freq)
                    band_list.append((b, freq_bands[b][0], freq_bands[b][1]))

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

            for band_name, band_low_freq, band_high_freq in band_list:
                # Uso las frecuencias adecuadas (personalizadas o predefinidas)
                current_low_freq = low_freq if using_custom_band else band_low_freq
                current_high_freq = high_freq if using_custom_band else band_high_freq
                
                band_mask = (self.fft_freq >= current_low_freq) & (self.fft_freq <= current_high_freq)

                fig, ax = plt.subplots(figsize=(10, 5))

                for i, channel_idx in enumerate(ch_idx):
                    channel_name = self.info.ch_names[channel_idx]
                    y_data = self.fft_psd[i, band_mask]

                    from scipy.ndimage import gaussian_filter1d
                    y_smooth = gaussian_filter1d(y_data, sigma=0.8)

                    ax.plot(self.fft_freq[band_mask], y_smooth, color=colors[i], label=f'Canal {channel_name}', linewidth=2)

                # Crear título apropiado
                if len(channel_names) > 3:
                    channel_title = f"{len(channel_names)} canales"
                else:
                    channel_title = ', '.join(channel_names)
                
                if using_custom_band:
                    title = f'Espectro de Potencia - {channel_title} - Banda Personalizada ({current_low_freq}-{current_high_freq} Hz)'
                else:
                    title = f'Espectro de Potencia - {channel_title} - Banda {band_name} ({current_low_freq}-{current_high_freq} Hz)'
                
                ax.set_title(title)
                ax.set_xlabel('Frecuencia (Hz)')
                ax.set_ylabel('Densidad Espectral de Potencia (dB)')
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()

    def freq_time(self, low_freq:float=1, high_freq:float=60.0, channels:list[str]|str ='Cz', separate_events:bool=True, color:str='magma'):
        """
        Realiza análisis tiempo-frecuencia de los datos EEG utilizando wavelets de Morlet.

        Este método calcula y visualiza la potencia tiempo-frecuencia de las señales EEG,
        permitiendo analizar cómo diferentes bandas de frecuencia varían en el tiempo
        en respuesta a eventos específicos.

        Args:
            low_freq : float, optional
                Frecuencia inferior del rango de análisis (por defecto 1 Hz).
            high_freq : float, optional
                Frecuencia superior del rango de análisis (por defecto 60 Hz).
            channels : str or list[str], optional
                Canal(es) a analizar. Puede ser un string con el nombre de un canal
                o una lista con múltiples nombres (por defecto 'Cz').
            separate_events : bool, optional
                Si True, genera gráficos separados para cada tipo de evento (por defecto True).
                Si False, genera un único gráfico con todos los eventos promediados.
            color: str, optional
                Color de visulización del gráfico (por defecto 'magma')

        Returns:
            power : mne.time_frequency.AverageTFR or dict
                Si separate_events=True, retorna un diccionario con objetos AverageTFR para cada evento.
                Si separate_events=False, retorna un único objeto AverageTFR con todos los eventos.

        Raises:
            ValueError:
                - Si no se encuentran eventos en las anotaciones.
                - Si los canales especificados no existen en los datos.
                - Si low_freq >= high_freq.

        Notes:
            - Utiliza wavelets de Morlet para el análisis tiempo-frecuencia.
            - Aplica corrección de línea base usando el período pre-estímulo (-0.2 a 0 segundos).
            - El cálculo se realiza con use_fft=True para mejor rendimiento computacional.
            - Los resultados se muestran como cambio porcentual respecto a la línea base.
            - Importante: este método **utiliza** `self.events` y `self.event_id` generados
            en la inicialización (mediante `_from_ann_to_events()`). Asegurate de que las
            anotaciones ya se hayan convertido a eventos antes de llamar a `freq_time`.
            Si modificaste `self.anotaciones` tras crear la instancia, llamá a `refresh_events()`
            para regenerar los eventos.
            - Evita mensajes de log ejecutando `mne.set_log_level('ERROR')` antes.

        Examples:
            >>> # Análisis para todos los eventos en el canal Cz
            >>> power = eeg_signal.freq_time(low_freq=1, high_freq=40, channels='Cz', separate_events=False)
            >>>
            >>> # Análisis separado por eventos para múltiples canales
            >>> power_dict = eeg_signal.freq_time(low_freq=4, high_freq=30, 
            ...                                  channels=['C3', 'C4', 'Fz'], separate_events=True)
            >>> # Acceder a los resultados para un evento específico
            >>> power_left = power_dict['left']
            >>>
            >>> # Análisis específico para banda beta en canal Pz
            >>> power = eeg_signal.freq_time(low_freq=13, high_freq=30, channels='Pz', separate_events=True)
        """
        from mne.time_frequency import tfr_morlet

        # Montaje estándar de MNE
        montage = mne.channels.make_standard_montage('standard_1005')
        mne_info = mne.create_info(ch_names=self.info.ch_names, sfreq=self.sfreq, ch_types=self.info.ch_types)
        mne_info.set_montage(montage)

        if len(self.events) == 0:
            raise ValueError("No se encontraron eventos en las anotacines. No se puede realizar el análisis tiempo-frecuencia.")

        # Creo objeto Raw de MNE con los datos referenciados
        raw = mne.io.RawArray(self.data_ref, mne_info)
        epochs = mne.Epochs(raw, self.events, event_id=self.event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0))
        erp = epochs.average()

        if low_freq >= high_freq:
            raise ValueError(f"low_freq debe ser menor que high_freq: {low_freq} vs {high_freq}")

        freqs = np.arange(low_freq, high_freq, 0.5)  # Frecuencias de 2 a 60 Hz
        n_cycles = freqs / 3.0  # Número de ciclos por frecuencia

        channels = _normalize_ch_list(channels, montage='standard_1005')[0] if channels else None

        for ch in channels:
            if ch not in self.info.ch_names:
                raise ValueError(f"El canal {ch} no se existe dentro de los datos. Canales disponibles: {self.info.ch_names}")

        if separate_events:
            power_dict = {}
            for event_name, event_code in self.event_id.items():
                epochs_event = epochs[self.events[:,2] == event_code]
                power = tfr_morlet(epochs_event, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, use_fft=True)
                power.apply_baseline(baseline=(-0.2, 0), mode='percent')

                power_dict[event_name] = power

                for i in range(len(channels)):
                    power.plot(picks=channels[i], title=f'Tiempo-Frecuencia (TFR) - Evento: {event_name.capitalize()}, Canal: {channels[i]}', 
                               show=True, cmap=color)
            
            return power_dict
        else:
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, use_fft=True)

            power.apply_baseline(baseline=(-0.2, 0), mode='percent') # percent para cambio porcentual
            
            for ch in channels:
                power.plot(picks=ch, title=f'Tiempo-Frecuencia (TFR) en {ch}', show=True, 
                        cmap=color)
                
            return power

    def hilbert(self, channels:list[str]|str, freq_band:tuple=(1,60), plot:bool=True, tmin:float=1, tmax:float=30):
        """
        Aplica la transformada de Hilbert a los datos EEG para obtener la señal analítica y su envolvente.

        Args:
            channels : str or list[str]
                Canal(es) a analizar. Los nombres se normalizarán al estándar MNE.
            freq_band : tuple
                Banda de frecuencia para filtrar (low_freq, high_freq), por defecto (1,60).
            plot : bool, optional
                Si True, genera gráficos de los resultados (por defecto True).
            tmin : float, optional
                Tiempo inicial en segundos para el análisis (por defecto 1).
            tmax : float, optional
                Tiempo final en segundos para el análisis (por defecto 30).

        Returns:
            hilbert_signal : np.array
                Señal analítica compleja resultante de la transformada de Hilbert.
            envelope : np.array
                Envolvente de amplitud de la señal.
        """
        from scipy.signal import hilbert, butter, filtfilt

        channels = _normalize_ch_list(channels, montage='standard_1005')[0]

        ch_idx = [self.info.ch_names.index(ch) for ch in channels]

        # Calculo índices de tiempo
        n_samps_total = self.data_ref.shape[1]
        tmin_samps = int(tmin * self.sfreq)
        tmax_samps = int(tmax * self.sfreq)

        # Aseguro que los índices estén dentro del rango
        tmin_samps = max(0, tmin_samps)
        tmax_samps = min(n_samps_total, tmax_samps)

        # Extraemos datos de los canales seleccionados en el intervalo de tiempo
        data = self.data_ref[ch_idx, tmin_samps:tmax_samps]

        # Filtro en la banda de frecuencia especificada
        if freq_band is not None:
            nyquist = self.sfreq / 2
            low_freq = freq_band[0] / nyquist
            high_freq = freq_band[1] / nyquist

            b_butter, a_butter = butter(4, [low_freq, high_freq], btype='band')
            data = filtfilt(b_butter, a_butter, data, axis=1)

        # Aplico transformada de Hilbert
        hilbert_signal = hilbert(data, axis=1)
        envelope = np.abs(hilbert_signal)

        # Ploteo
        if plot:
            self._plot_hilbert(channels, hilbert_signal, envelope, tmin, tmax)

        return hilbert_signal, envelope

    def refresh_events(self):
        """
        Regenera y actualiza `self.events` y `self.event_id` a partir de `self.anotaciones`.

        Uso recomendado cuando se han modificado `self.anotaciones` después de la inicialización.
        Esta función es la interfaz pública para actualizar los eventos y evita que el usuario
        tenga que invocar el método privado `_from_ann_to_events()`.

        Returns:
            tuple: (events, event_id)
                events: np.ndarray
                    Array de eventos actualizado en formato MNE ([muestra_relativa, 0, código]).
                event_id: dict
                    Diccionario de mapeo actualizado {descripción: código}.
        """
        self.events, self.event_id = self._from_ann_to_events()
        return self.events, self.event_id

    def _plot_hilbert(self, channels, hilbert_signal, envelope, tmin, tmax):
        """
        Genera gráficos de la señal analítica y su envolvente para cada canal.

        Args:
            channels : list[str]
                Nombres de los canales a graficar.
            hilbert_signal : np.array
                Señal analítica compleja.
            envelope : np.array
                Envolvente de amplitud.
            tmin : float
                Tiempo inicial del segmento.
            tmax : float
                Tiempo final del segmento.
            sfreq : float
                Frecuencia de muestreo.
        """
        # Creo vector de tiempo para el segmento
        n_samps_segment = hilbert_signal.shape[1]
        t = np.linspace(tmin, tmax, n_samps_segment)

        for idx, ch in enumerate(channels):
            plt.figure(figsize=(12, 6))

            # Grafico la parte real de la señal analítica (que es la señal filtrada)
            plt.plot(t, np.real(hilbert_signal[idx]), label='Señal filtrada')

            # Grafico la envolvente
            plt.plot(t, envelope[idx], label='Envolvente', linewidth=2)
            plt.title(f"Transformada de Hilbert - Canal: {ch}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (µV)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def _from_ann_to_events(self):
        """
        Convierte anotaciones a eventos en formato MNE.

        (Método privado — uso interno.)

        Asume que las anotaciones están en tiempo absoluto respecto al inicio
        original del registro. Ajusta automáticamente restando `first_samp`
        para devolver índices de muestra relativos al objeto actual.

        Returns:
            tuple: (events, event_id)
                events: np.ndarray
                    Array de eventos en formato MNE con filas [muestra_relativa, 0, código].
                    Las muestras son calculadas como int(onset * sfreq) - first_samp.
                event_id: dict
                    Diccionario de mapeo {descripción: código} generado a partir de las
                    descripciones de las anotaciones.

        Notes:
            - Este método se invoca internamente durante la inicialización para generar
            `self.events` y `self.event_id`. No está pensado para ser llamado
            directamente por usuarios; usá `refresh_events()` si necesitas regenerar
            los eventos tras modificar `self.anotaciones`.
            - Los eventos devueltos ya están ajustados por `first_samp`; no deben restarse
            de nuevo al pasarlos a funciones de MNE.
        """
        events, event_id = [], {}
        current_id = 1

        for onset, duration, description in zip(self.anotaciones.onset, self.anotaciones.duration, self.anotaciones.description):

            samples = int(onset * self.sfreq) - self.first_samp

            # Verificar que la muestra esté dentro del rango actual
            if 0 <= samples < self.data.shape[1]:
                if description not in event_id:
                    event_id[description] = current_id
                    current_id += 1
                events.append([samples, 0, event_id[description]])
            else:
                logging.warning(f"Evento en muestra {samples + self.first_samp} fuera del rango actual")

        return np.array(events), event_id
    
    def _plot_waveform(self, erp, events, event_id, channels_of_interest, time_points, event_index):
        """
        Visualiza waveforms ERP para canales específicos en un evento determinado.
        
        Args:
            erp: 
                Objeto ERP de MNE
            events: 
                Array de eventos
            event_id: 
                Diccionario de mapeo de eventos
            channels_of_interest: 
                Lista de canales a visualizar
            time_points: 
                Lista de tiempos para líneas verticales
            event_index: 
                Índice del evento a visualizar
        """
        # Segundo gráfico: Waveforms de canales específicos
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Canales de interés
        channels_of_interest = channels_of_interest if isinstance(channels_of_interest, list) else [channels_of_interest]
        
        # Grafico waveform de cada canal de interés
        for ch in channels_of_interest:
            if ch in erp.ch_names:
                ch_idx = erp.ch_names.index(ch)
                ax.plot(erp.times, erp.data[ch_idx], label=ch, linewidth=2)
        
        event_num_to_name = {v: k for k, v in event_id.items()}
        eventos_mostrados = set()

        # Grafico las lineas temporales
        for t in time_points:
            sample = int(t * self.sfreq)
            idx_event = np.argmin(np.abs(events[:,0] - sample))
            event_num = events[idx_event, 2]
            event_name = event_num_to_name.get(event_num, 'Desconocido')

            # Solo agrega el label si el evento no ha sido mostrado
            if event_name not in eventos_mostrados:
                ax.axvline(t, color='r', linestyle='--', alpha=0.7, label='Puntos temporales')
                eventos_mostrados.add(event_name)
            else:
                ax.axvline(t, color='r', linestyle='--', alpha=0.7)

        ax.axhline(0, color="#000000", linestyle='-', alpha=0.5)
        ax.axvline(0, color="#000000", linestyle='-', alpha=0.9, label='Evento')

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Amplitud (µV)')
        ax.legend()
        ax.set_title(f'Waveforms ERP para Canales de Interés ({event_num_to_name[event_index].capitalize()})')
        
        plt.tight_layout()
        plt.show()

    def _topomap(self, erp, events, mne_info, event_id, event_index, time_points):
        """
        Genera mapas topográficos para visualización de datos EEG.
        
        Args:
            events: 
                Array de eventos en formato MNE
            mne_info: 
                Información de canales compatible con MNE
            event_id: 
                Diccionario de mapeo de eventos
            t_after_event: 
                Tiempo después del evento para análisis ERP
            t_previous_event: 
                Tiempo antes del evento para análisis ERP  
            event_index: 
                Índice del evento a visualizar
            time_points: 
                Lista de tiempos para mapas topográficos
            
        Returns:
            erp: 
                Objeto ERP de MNE o None si no hay eventos
        """
        from scipy import signal

        # Si no hay eventos, muestro mapas topográficos de potencia RMS por bandas de frecuencia
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
                filtered_data = signal.filtfilt(b, a, self.data_ref, axis=1)

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

            return None
        
        else:
            # Tiempos de interés para los mapas topográficos
            time_points = time_points if isinstance(time_points, list) else [time_points]

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

            event_num_to_name = {v: k for k, v in event_id.items()}
            plt.suptitle(f'Mapas Topográficos de Amplitud ERP en Tiempos Específicos - ({event_num_to_name[event_index].capitalize()})')
            plt.tight_layout()
            plt.show()

            return erp

