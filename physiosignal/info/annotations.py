from physiosignal.utils import _checking, _sort
import numpy as np
import pandas as pd
import logging

class Annotations:
    """
    Representa un conjunto de anotaciones temporales para señales fisiológicas.

    Attributes:
        onset (np.ndarray): Array de tiempos de inicio de las anotaciones
        duration (np.ndarray): Array de duraciones de las anotaciones
        description (np.ndarray): Array de descripciones textuales
        ch_names (np.ndarray | None): Array de nombres de canales asociados

    Methods:
        add: Añade nuevas anotaciones a la colección
        remove: Elimina anotaciones basado en criterios de búsqueda
        get_annotations: Retorna todas las anotaciones como DataFrame
    """

    def __init__(self, onset, duration, description, ch_names=None, see_logs:bool=True):
        """
        Inicializa un nuevo conjunto de anotaciones.

        Args:
            onset: Tiempos de inicio (scalar o array-like)
            duration: Duraciones (scalar o array-like)
            description: Descripciones (scalar o array-like)
            ch_names: Nombres de canales (None, scalar o array-like)

        Raises:
            ValueError: Si los parámetros no pasan las validaciones de formato
        """
        # Hago comprobaciones de los parámetros
        self. onset, self.duration, self.description, self.ch_names = _checking(onset, duration, description, ch_names)
         
        # Ordeno los parámetros en base a onset
        _sort(self)

        self.logs = see_logs
        self._logger = logging.getLogger(__name__) # Inicializo el logger
        if not self.logs: # Activa o desactiva el estado
            self._logger.disabled = True

    def add(self, onset, duration, description, ch_names=None):
        """
        Añade nuevas anotaciones al conjunto existente.

        Args:
            onset: Tiempos de inicio de nuevas anotaciones
            duration: Duraciones de nuevas anotaciones
            description: Descripciones de nuevas anotaciones
            ch_names: Nombres de canales asociados (opcional)

        Notes:
            Realiza validación de formato antes de añadir
            Mantiene el orden temporal después de la adición
        """
        on, dur, des, ch = _checking(onset, duration, description, ch_names)

        if not any(on==o and dur==d and des == de for o, d, de in zip(self.onset, self.duration, self.description)):
            self.onset = np.append(self.onset, on)
            self.duration = np.append(self.duration, dur)
            self.description = np.append(self.description, des) 

            if ch_names is not None and ch not in self.ch_names:
                self.ch_names = np.append(self.ch_names, ch)

            self._logger.info("Anotación añadida correctamente")

        else:
            self._logger.info(f"La anotación con tiempos: {onset} y descprición: {description} ya existe.")

    def remove(self, onset=None, duration=None, description=None, ch_names=None):
        """
        Elimina anotaciones basado en criterios especificados.

        Args:
            onset: Valor(es) de tiempo de inicio para eliminar
            duration: Valor(es) de duración para eliminar
            description: Valor(es) de descripción para eliminar
            ch_names: Valor(es) de nombres de canal para eliminar

        Notes:
            Si se especifican múltiples criterios, se aplican en AND
            Si no se especifica ningún criterio, no se elimina nada
            Reconstruye los arrays internos después de la eliminación
        """
        df = pd.DataFrame.from_dict(self._getInfo(), orient="columns")

        dic = {'onset':onset,
               'duration':duration,
               'description':description,
               'ch_names':ch_names}
        
        for key, value in dic.items():
            if value is None:
                continue

            if key not in df.columns:
                raise ValueError(f"La columna {key} no se encuentra dentro de las anotaciones")
            
            if isinstance(value, (list, tuple)):
                quit_idx = df[df[key].isin(value)].index
            else:
                quit_idx = df[df[key] == value].index

            if len(quit_idx)>0:
                df.drop(quit_idx)

            df = df.reset_index(drop=True)

            # Reconstruyo los array
            self.onset = df["onset"].values
            self.duration = df['duration'].values
            self.description = df['description'].values
            self.ch_names = df['ch_names'].values
            
    def get_annotations(self):
        """
        Retorna todas las anotaciones como un DataFrame estructurado.

        Returns:
            pd.DataFrame: DataFrame con columnas:
                - onset: Tiempos de inicio
                - duration: Duraciones
                - description: Descripciones
                - ch_names: Nombres de canales (si existen)

        Notes:
            Excluye la columna ch_names si no hay información de canales
        """
        df = pd.DataFrame.from_dict(self._getInfo(), orient="columns")

        if self.ch_names is not None: # Quito o no los canales
            return df
        else:
            df = df.drop(["ch_names"], axis=1)
            return df

    def find(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
    
    def _getInfo(self):
        """
        Método interno para obtener las anotaciones como diccionario.

        Returns:
            dict: Diccionario con la estructura:
                {
                    'onset': lista de tiempos,
                    'duration': lista de duraciones,
                    'description': lista de descripciones,
                    'ch_names': lista de canales o None
                }
        """
        return {
            'onset': self.onset.tolist(),
            'duration': self.duration.tolist(),
            'description': self.description.tolist(),
            'ch_names': self.ch_names.tolist() if self.ch_names is not None else [None]*len(self.onset)
        }