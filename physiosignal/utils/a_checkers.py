import numpy as np

def _checking(onset, duration, description, ch_names):
    """
    Valida y formatea los parámetros de entrada para crear anotaciones.

    Args:
        onset: Tiempos de inicio de las anotaciones (puede ser escalar o array-like)
        duration: Duración de las anotaciones (puede ser escalar o array-like)
        description: Descripción de las anotaciones (puede ser escalar o array-like)
        ch_names: Nombres de canales asociados (puede ser None, escalar o array-like)

    Returns:
        tuple: (onset, duration, description, ch_names) formateados como arrays 1D

    Raises:
        ValueError: Si los parámetros no son arrays 1D o tienen longitudes inconsistentes
        TypeError: Si los tipos de entrada no son convertibles al formato requerido
    """
    onset = np.atleast_1d(np.array(onset, dtype=float))
    if onset.ndim != 1:
        raise ValueError("Onset debe ser un array de 1 dimensión")
    
    duration = np.array(duration, dtype=float)
    if duration.ndim == 0 or duration.shape == (1,):
        duration = np.repeat(duration, len(onset))
    if duration.ndim != 1:
        raise ValueError("Duration debe ser un array de 1 dimensión")
    
    description = np.array(description, dtype=str)
    if description.ndim == 0 or description.shape == (1,):
        description = np.repeat(description, len(onset))
    if description.ndim != 1:
        raise ValueError("Descrpition debe ser un array de 1 dimensión")
    
    params = list(map(len, [onset, duration, description]))

    if ch_names is not None:
        ch_names = np.array(ch_names, dtype=str)
        if ch_names.ndim == 0 or ch_names.shape == (1,):
            ch_names = np.repeat(ch_names, len(onset))
        if ch_names.ndim != 1:
            raise ValueError("Ch_names debe ser un array de 1 dimensión")
        params.append(len(ch_names))
        
    if len(set(params)) != 1:
        raise ValueError(f"Todos los parámetros deben poseer el mismo largo. Largos hallados: {params}")

    return onset, duration, description, ch_names

def _sort(self):
    """
    Ordena las anotaciones en base al tiempo de inicio (onset) de forma ascendente.

    Modifica los siguientes atributos in-place:
        self.onset, self.duration, self.description, self.ch_names

    Notes:
        Utiliza np.argsort para determinar el orden de los índices
        Mantiene la coherencia entre todos los atributos durante el reordenamiento
    """
    orden = np.argsort(self.onset)
    self.onset = self.onset[orden]
    self.duration = self.duration[orden]
    self.description = self.description[orden]
    if self.ch_names is not None:
        self.ch_names = self.ch_names[orden]