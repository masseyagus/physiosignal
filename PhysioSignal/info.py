from dataclasses import dataclass, field, asdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s.%(funcName)s]: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

@dataclass
class Info():
    """
    Clase para almacenar y gestionar metadatos de registros de señales fisiológicas.
    Implementa comportamiento similar a un diccionario con funcionalidades adicionales.
    
    Args:
        ch_names: Nombres de los canales de señal
        ch_types: Tipo(s) de los canales (str o lista por canal)
        sfreq: Frecuencia de muestreo en Hz (valor numérico)
        bad_channels: Lista de canales marcados como defectuosos
        experimenter: Nombre del responsable del registro
        subject_info: Información adicional del sujeto
        register_type: Tipo de registro o experimento
        
    Raises:
        KeyError: Al acceder a atributos inexistentes
        ValueError: En operaciones de renombrado inválidas
    """
    ch_names:list[str] = field(default_factory=list)
    ch_types:str|list[str] = field(default_factory=list)
    sfreq:float = field(default_factory=512.0)
    bad_channels:list = field(default_factory=list)
    experimenter:str=None
    subject_info:str=None
    register_type:str=None

    def __contains__(self, key:str) -> bool:
        """
        Verifica la existencia de un atributo en la instancia
        """
        # Uso de hasattr para chequeo directo de atributos
        return hasattr(self, key)

    def __getitem__(self, key:str):
        """
        MAcceso a atributos mediante sintaxis de diccionario
        """
        if hasattr(self, key):
            return getattr(self, key)
        
        # Lanzo error en caso de clave faltante
        raise KeyError(f"La clave {key} no existe en el objeto")
    
    def __len__(self) -> int:
        """
        Cantidad total de atributos registrados
        """
        # Uso de __annotations__ para obtener campos declarados
        return len(self.__annotations__)
    
    def get(self, key:str=None, value=None):
        """
        Acceso seguro a atributos con valor por defecto opcional
        """
        return getattr(self, key, value)
    
    def items(self) -> dict:
        """
        Devuelve pares key-value como diccionario
        """
        return asdict(self)

    def rename_channels(self, new_names:dict) -> None:
        """
        Renombra canales manteniendo la integridad de los datos asociados
        
        Args:
            new_names: Diccionario con mapeo {nombre_actual: nombre_nuevo}
            
        Returns:
            Instancia actualizada para permitir method chaining
            
        Raises:
            ValueError: Si hay nombres inválidos o duplicados
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
       
    #    Actualización de atributos
        self.ch_names = new 
        self.bad_channels = [new_names.get(bad, bad) for bad in self.bad_channels]

        return self
              

    def visualizeInfo(self):
        """
        Visualización de los atributos del objeto en formato de tabla
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

def main():
    canales=[i+1 for i in range(5)]

    info = Info(
    ch_names=canales,
    ch_types=["eeg"]*len(canales),
    bad_channels=['Cz'],
    sfreq=512,
    register_type="Registro EEG para análisis de patrones ERDS",
    experimenter="MSc. PEREYRA Magalí",
    subject_info={"edad": 22, "sexo": "F"}
    )
    info.visualizeInfo()

if __name__ == "__main__":
    main()
