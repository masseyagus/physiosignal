from dataclasses import dataclass, field, asdict
from typing import Union, Dict, List
import numpy as np

@dataclass
class Info:
    """
    Clase para almacenar y gestionar metadatos de registros de señales fisiológicas.
    Implementa comportamiento similar a diccionario con funcionalidades extendidas para
    manejo seguro de metadatos biomédicos.

    Args:
        ch_names: Lista de nombres de canales (default: lista vacía)
        ch_types: Tipo(s) de canal - cadena única o lista por canal (default: lista vacía)
        sfreq: Frecuencia de muestreo en Hz (default: 512.0)
        bad_channels: Lista de canales marcados como defectuosos (default: lista vacía)
        experimenter: Nombre del experimentador/responsable (opcional)
        subject_info: Información adicionañ del sujeto en formato str/dict (opcional)
        register_type: Tipo de registro/experimento (opcional)

    Raises:
        KeyError: Al acceder a atributos no existentes mediante operador []
        ValueError: En operaciones de renombrado inválidas o datos inconsistentes
    """
    ch_names:List[str] = field(default_factory=list)
    ch_types:Union[str, List[str]] = field(default_factory=list)
    sfreq:float = field(default_factory=512.0)
    bad_channels:List[str] = field(default_factory=list)
    experimenter:str=None
    subject_info:str=None
    register_type:str=None

    def __post_init__(self):
        """
        Inicialización posterior para validar consistencia de datos.
        Convierte tipos de canal a formato lista cuando es necesario.

        Actions:
            1. Normaliza ch_types a lista si es string único
            2. Verifica igual longitud de ch_names y ch_types
            3. Valida sfreq > 0
        
        Raises:
            ValueError: Si hay discrepancia en cantidad de canales/tipos
                        Si la frecuencia de muestreo es menor a 1 Hz
        """

        # Convierto a ch_types en lista si es string
        if isinstance(self.ch_types, str):
            self.ch_types = [self.ch_types] * len(self.ch_names) if self.ch_names else []

        # Genero la misma cantidad de valores que ch_names si es una lista de un solo
        elif isinstance(self.ch_types, list):
            if len(self.ch_types) == 1:
                self.ch_types = self.ch_types * len(self.ch_names)

        # Valido longitud de ch_types vs ch_names
        if len(self.ch_names) != len(self.ch_types):
            raise ValueError("La cantidad de ch_names y ch_types debe ser igual")
        
        # Validar sfreq positivo
        if self.sfreq < 1:
            raise ValueError("La frecuencia de muestreo debe ser ≥ 1 Hz")

    def __contains__(self, key:str) -> bool:
        """
        Verifica si un atributo existe en la instancia.

        Args:
            key: Nombre del atributo a verificar

        Returns:
            bool: True si el atributo existe, False en caso contrario
        """
        # Uso de hasattr para chequeo directo de atributos
        return hasattr(self, key)

    def __getitem__(self, key:str):
        """
        Acceso tipo diccionario a los atributos de la instancia.

        Args:
            key: Nombre del atributo a recuperar

        Returns:
            Valor del atributo solicitado

        Raises:
            KeyError: Si el atributo no existe en la instancia
        """
        if hasattr(self, key):
            return getattr(self, key)
        
        # Lanzo error en caso de clave faltante
        raise KeyError(f"La clave {key} no existe en el objeto")
    
    def __len__(self) -> int:
        """
        Devuelve la cantidad de atributos declarados en la clase.

        Returns:
            int: Número total de campos en la dataclass
        """
        # Uso de __annotations__ para obtener campos declarados
        return len(self.__annotations__)
    
    def get(self, key:str=None, value=None):
        """
        Recupera un atributo con valor por defecto opcional.

        Args:
            key: Nombre del atributo a recuperar
            value: Valor a devolver si el atributo no existe (default: None)

        Returns:
            Valor del atributo o valor por defecto si no existe
        """
        return getattr(self, key, value)
    
    def items(self) -> Dict:
        """
        Devuelve todos los atributos como diccionario clave-valor.

        Returns:
            dict: Diccionario con {nombre_atributo: valor}
        """
        return asdict(self)

    def keys(self) -> List[str]:
        """
        Devuelve los nombres de todos los atributos definidos en la instancia de Info.

        Returns:
            List[str]: Lista de cadenas con el nombre de cada atributo de la dataclass.

        Example:
            >>> info = Info(ch_names=['Fp1','Fp2'], ch_types='eeg', sfreq=512)
            >>> info.keys()
            ['ch_names', 'ch_types', 'sfreq', 'bad_channels', 'experimenter', 'subject_info', 'register_type']
        """
        return list(self.__annotations__.keys())

    def rename_channels(self, new_names:dict) -> "Info":
        """
        Renombra canales manteniendo la integridad de los datos asociados
        
        Args:
            new_names: Diccionario con mapeo {nombre_actual: nombre_nuevo}
            
        Returns:
            Info: Instancia actualizada (permite method chaining)

        Example:
            >>> info.rename_channels({'Fp1': 'FP1', 'Fp2': 'FP2'})

        Raises:  
            ValueError: Si algún nombre antiguo no existe
                        Si se generan nombres duplicados
        """
        # Validación de nombres existentes
        for old_name in new_names: # Tomo cada nombre nuevo
            if old_name not in self.ch_names: # Verifico que existan todos los nombres
                raise ValueError(f"El canal {old_name} no existe en el objeto")
        
        # Genero nueva lista de nombres
        new = [new_names.get(name, name) for name in self.ch_names] # Actualizo los nombres

        #Verifico que no existan duplicados
        if len(set(new)) != len(new):
            duplicado = {name for name in new if new.count(name) > 1}
            raise ValueError(f"Existen nombres duplicados dentro de la lista: {duplicado}")
       
        # Actualización de atributos
        self.ch_names = new 
        self.bad_channels = [new_names.get(bad, bad) for bad in self.bad_channels]

        return self

    def visualizeInfo(self):
        """
        Muestra los metadatos en formato de tabla estilizada (HTML).
        Requiere entorno compatible con visualización HTML (ej: Jupyter Notebook).
        """
        import pandas as pd
        from IPython.display import display

        # Convertir valores a listas de un solo elemento si no son iterables tipo lista/dict
        def normalize_value(val):
            if isinstance(val, (list, dict)):
                return [val]  # Lo dejamos como lista de un solo elemento
            return [str(val)]  # Convertimos todo lo demás a string y lo metemos en una lista

        # Normalizar los valores
        data = {k: normalize_value(v) for k, v in self.items().items()}

        # Construcción del DataFrame
        df = pd.DataFrame.from_dict(data, orient="index", columns=["Datos"])

        # Formateo de la tabla
        styled_table = (
            df.reset_index()
            .rename(columns={"index": "Atributo"})
            .style.hide(axis="index")
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center'), ('font-weight', 'bold')]
            }])
        )
        
        display(styled_table)

    def filter_by_type(self, ch_type:str) -> List[str]:
        """
        Filtra canales por tipo especificado.

        Args:
            ch_type: Tipo de canal a filtrar (ej: 'eeg', 'ecg')

        Returns:
            List[str]: Lista de nombres de canales que coinciden con el tipo

        Example:
            >>> canales_eeg = info.filter_by_type('eeg')
            >>> print(canales_eeg)
            ['Fp1', 'Fp2', 'C3', 'C4', ...]
        """

        return [nombre for nombre, tipo in zip(self.ch_names, self.ch_types) if tipo == ch_type.lower()]
    
    def _select(self, select) -> None:
        """
        Filtra los canales conservando solo los especificados en `select`,
        respetando el orden pedido en `select` y sin reordenar inesperadamente.

        - select puede ser: int, str, list/tuple/np.ndarray (con nombres y/o índices).
        - Si hay índices en 'select', se resuelven respecto al orden actual de self.ch_names.
        - Los nombres/índices inexistentes se ignoran (se registra un warning).
        - También actualiza self.ch_types si existe para mantener la correspondencia.

        No devuelve nada; modifica self.ch_names (y self.ch_types si aplica).
        """
        # Normalizar select a lista
        if isinstance(select, (list, tuple, np.ndarray)):
            select_list = list(select)
        else:
            select_list = [select]

        # Guardar estado original
        orig_names = list(self.ch_names)
        orig_types = list(self.ch_types) if hasattr(self, "ch_types") else None

        resolved = []
        for item in select_list:
            # si es índice (int o numpy integer) resuelvo a nombre (si está en rango)
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if 0 <= idx < len(orig_names):
                    name = orig_names[idx]
            else:
                # candidato nombre (convertir a str para comparar)
                name = str(item)

            # si el nombre existe en los originales y no lo añadí ya, lo agrego
            if name in orig_names and name not in resolved:
                resolved.append(name)

        # Actualizar ch_names
        self.ch_names = resolved

        # Actualizar ch_types para mantener correspondencia si existían antes
        if orig_types is not None:
            new_types = []
            for name in resolved:
                orig_idx = orig_names.index(name)
                new_types.append(orig_types[orig_idx])
            self.ch_types = new_types


