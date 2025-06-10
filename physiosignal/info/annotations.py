from physiosignal.utils import _checking, _sort
import numpy as np
import pandas as pd
import logging

class Annotations:
    """
    Representa un conjunto de anotaciones temporales para señales fisiológicas.

    Attributes:
        onset (np.ndarray): Array de tiempos de inicio de las anotaciones.
        duration (np.ndarray): Array de duraciones de las anotaciones.
        description (np.ndarray): Array de descripciones textuales.
        ch_names (np.ndarray | None): Array de nombres de canales asociados. Cada elemento
            puede ser None o una lista de cadenas (si la anotación tiene uno o varios canales).

    Methods:
        add: Añade nuevas anotaciones al conjunto, evitando duplicados e incorporando
             uno o varios canales por anotación.
        remove: Elimina anotaciones según criterios de onset, duration, description o canales.
        get_annotations: Retorna todas las anotaciones como un DataFrame con columnas
                         onset, duration, description y ch_names, ajustando ch_names
                         para que coincida en longitud con las otras columnas.
    """

    def __init__(self, onset=None, duration=None, description=None, ch_names=None, see_logs:bool=True):
        """
        Inicializa una nueva instancia de anotaciones con validación, normalización y ordenamiento.

        Args:
            onset (float, list o array, opcional): Tiempo(s) de inicio de las anotaciones.
            duration (float, list o array, opcional): Duración(es) de las anotaciones.
            description (str, list o array, opcional): Descripción(es) de las anotaciones.
            ch_names (None, str, lista simple o lista de listas, opcional): Canales asociados a cada anotación.
            see_logs (bool, opcional): Controla si se habilitan los logs para esta instancia (por defecto True).

        Raises:
            ValueError: Si los parámetros no cumplen con los requisitos de formato o longitud.
            TypeError: Si alguno de los parámetros no puede convertirse al tipo requerido.

        Notes:
            - Utiliza la función `_checking()` para validar y formatear los parámetros de entrada, asegurando que 
            todos sean arrays 1D con longitudes coherentes.
            - Aplica `_sort()` para ordenar las anotaciones según el tiempo de inicio (`onset`) en orden ascendente.
            - Convierte `ch_names` a un array de objetos donde cada elemento puede ser `None` o una lista de cadenas,
            facilitando el manejo uniforme de canales asociados.
            - Configura el logger para la instancia y permite habilitar o deshabilitar la salida de logs mediante `see_logs`.
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
        Añade nuevas anotaciones evitando duplicados exactos.

        Args:
            onset: Tiempo(s) de inicio de nuevas anotaciones
            duration: Duración(es) de nuevas anotaciones
            description: Descripción(es) de nuevas anotaciones
            ch_names: Canales asociados (None, string, lista o lista de listas)

        Returns:
            None: Modifica los atributos internos in-place

        Notes:
            - Realiza validación de formato idéntica al constructor
            - Considera duplicado solo si todos los campos (incluyendo canales) coinciden
            - Mantiene el orden temporal después de la adición
            - Genera logs informativos sobre éxito o duplicados detectados
        """
        on, dur, des, ch = _checking(onset, duration, description, ch_names)
        
        # # Verificar duplicados incluyendo canales
        existing_tuples = list(zip(self.onset, self.duration, self.description, self.ch_names))
        new_tuples = list(zip(on, dur, des, ch))
        
        if not any(nt in existing_tuples for nt in new_tuples):
            # Inicializar ch_names si es necesario
            if self.ch_names is None:
                self.ch_names = np.array([None] * len(self.onset))
            
            # Añadir nuevas anotaciones
            self.onset = np.append(self.onset, on)
            self.duration = np.append(self.duration, dur)
            self.description = np.append(self.description, des)
            self.ch_names = np.append(self.ch_names, ch)

            _sort(self)
            
            self._logger.info("Anotación añadida correctamente")

        else:
            dup_example = next(nt for nt in new_tuples if nt in existing_tuples)
            self._logger.info(
                f"Anotación duplicada: onset={dup_example[0]}, "
                f"duración={dup_example[1]}, "
                f"descripción='{dup_example[2]}', "
                f"canales={dup_example[3]}"
            )

    def remove(self, onset=None, duration=None, description=None, ch_names=None):
        """
        Elimina anotaciones del conjunto actual según criterios de búsqueda específicos.

        Los criterios pueden incluir uno o más de los siguientes campos: `onset`, `duration`,
        `description` y `ch_names`. Si se proporcionan múltiples criterios, se aplican de manera
        secuencial, descartando las coincidencias en cada paso.

        Args:
            onset (float | list | np.ndarray, opcional): Tiempo(s) de inicio a eliminar.
            duration (float | list | np.ndarray, opcional): Duración(es) a eliminar.
            description (str | list, opcional): Descripción(es) a eliminar.
            ch_names (str | list, opcional): Nombre(s) de canal asociado(s) a eliminar.

        Notes:
            - El filtrado por criterios se hace de forma independiente (no se combinan con AND).
            - Si no se especifica ningún criterio, no se eliminará ninguna anotación.
            - Se actualizan internamente los arrays `onset`, `duration`, `description` y `ch_names`.
            - Se mantiene el orden cronológico después de la eliminación.
            - Si se encuentra al menos una coincidencia, se registrará un mensaje informativo.
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
                df.drop(quit_idx, axis=0, inplace=True)

            df = df.reset_index(drop=True)

            # Reconstruyo los array
            self.onset = df["onset"].values
            self.duration = df['duration'].values
            self.description = df['description'].values
            self.ch_names = df['ch_names'].values
            
            _sort(self)

            logging.info(f"Anotación eliminada correctamente")

    def get_annotations(self):
        """
        Retorna todas las anotaciones como DataFrame con longitud consistente.

        Returns:
            pd.DataFrame: DataFrame con columnas:
                - onset: Tiempos de inicio
                - duration: Duraciones
                - description: Descripciones textuales
                - ch_names: Canales asociados (None o listas de strings)

        Notes:
            - Ajusta automáticamente la columna ch_names para igualar el número de filas:
                * Si hay más canales que anotaciones: agrupa los excedentes en la última fila
                * Si hay menos canales que anotaciones: completa con None
            - Mantiene la integridad de los datos originales en los atributos de la clase
            - El DataFrame resultante es independiente de los datos internos
        """
        info = self._getInfo()
        on = info['onset']
        du = info['duration']
        de = info['description']
        ch = info['ch_names']

        n = len(on)
        m = len(ch)

        if m > n:
            # Tomamos los primeros n-1 canales individuales,
            # y relegamos todos los “extras” a la última posición como lista.
            nuevos = []
            for i in range(n - 1):
                nuevos.append(ch[i])
            # 'ch[n-1:]' es la lista de todos los canales restantes
            nuevos.append(ch[n - 1 :])
            info['ch_names'] = nuevos

        elif m < n:
            # Si hubiera menos canales que anotaciones,
            # rellenamos por detrás con None para que iguale longitud n.
            info['ch_names'] = ch + [None] * (n - m)

        # Ahora len(info['ch_names']) == n y el DataFrame no dará error
        return pd.DataFrame.from_dict(info, orient="columns")

    def find(self, filtros:tuple|list):
        """
        Busca y filtra anotaciones según un valor específico y, opcionalmente, una columna.

        Args:
            filtros (tuple | list): 
                Puede ser una tupla o lista con uno o dos elementos:
                
                - Si contiene un único valor (`float` o `str`), se buscará en todas las columnas.
                - Si contiene dos elementos `(valor, columna)`, se buscará `valor` únicamente dentro de la columna especificada.

        Returns:
            pd.DataFrame: Un DataFrame con las filas o columnas filtradas según el criterio dado.

        Raises:
            LookupError: Si no se encuentra la columna indicada o si no se hallan registros que coincidan con el filtro.

        Notes:
            - Si no se proporciona ningún filtro, se devuelve el DataFrame completo.
            - La búsqueda es exacta y sensible a mayúsculas/minúsculas.
        """
        info = self.get_annotations()

        df = pd.DataFrame.from_dict(info, orient="columns")

        fila_val, col_val = (None, None)
        if isinstance(filtros, (tuple, list)) and len(filtros) == 2:
            fila_val, col_val = filtros
        elif isinstance(filtros, (float, str)):
            fila_val, col_val = filtros, None

        # Inicializo df_filas con todo el DataFrame por defecto
        df_filas = df.copy()

        # Filtro filas si se indicó fila_val
        if fila_val is not None:
            mask = df.eq(fila_val)
            df_filas = df.loc[mask.any(axis=1)]

        # Filtro filas si se indicó col_val
        if col_val is not None:
            if col_val not in df.columns:
                raise LookupError(f"No se encontró columna con nombre {col_val}")
            
            df_filas = df_filas.loc[:,[col_val]]

        if df_filas.empty:
            raise LookupError(f"No se encontraron registros con los filtros proporcionados {filtros}")

        return df_filas

    def save(self, filename:str):
        """
        Guarda las anotaciones actuales en un archivo CSV.

        Args:
            filename (str): Nombre (o ruta) del archivo sin extensión donde se guardarán las anotaciones.  
                            La función añadirá automáticamente la extensión '.csv'.

        Returns:
            None

        Notes:
            - El archivo CSV se guardará sin incluir índices de fila.
            - La información guardada corresponde a los datos retornados por el método `get_annotations()`.
            - Se genera un log de información confirmando el guardado exitoso del archivo.
        """
        df = self.get_annotations()

        df.to_csv(f"{filename}.csv", index=False)

        logging.info(f"Archivo {filename}.csv guardado correctamente")

    def load(self, path:str):
        """
        Carga anotaciones desde un archivo CSV y crea una instancia de `Annotations`.

        Args:
            path (str): Ruta al archivo CSV que contiene las anotaciones.  
                        El archivo debe contener al menos las columnas 'onset', 'duration' y 'description'.  
                        La columna 'ch_names' es opcional.

        Returns:
            Annotations: Una instancia de la clase `Annotations` inicializada con los datos cargados.

        Raises:
            FileNotFoundError: Si el archivo especificado no existe.
            pd.errors.ParserError: Si el archivo CSV no tiene el formato correcto.
            KeyError: Si faltan las columnas obligatorias ('onset', 'duration' o 'description') en el CSV.
        """
        df = pd.read_csv(path)

        raw_onset = df['onset']
        raw_duration = df['duration']
        raw_description = df['description']

        if "ch_names" in df.columns:
            raw_ch_names = df['ch_names']
        else:
            raw_ch_names = None
        
        return Annotations(onset=raw_onset, duration=raw_duration, description=raw_description, ch_names=raw_ch_names)
    
    def _getInfo(self):
        """
        Método interno: Retorna representación diccionario de las anotaciones.

        Returns:
            dict: {
                'onset': lista de floats,
                'duration': lista de floats,
                'description': lista de strings,
                'ch_names': lista (None o listas de strings)
            }

        Warning:
            Este método es para uso interno. Para obtener datos públicos use get_annotations().
        """
        return {
            'onset': self.onset.tolist(),
            'duration': self.duration.tolist(),
            'description': self.description.tolist(),
            'ch_names': self.ch_names.tolist() if self.ch_names is not None else [None]*len(self.onset)
        }

        