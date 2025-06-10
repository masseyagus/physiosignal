import numpy as np

def _checking(onset, duration, description, ch_names):
    """
    Valida y convierte los parámetros de anotación a arrays 1D consistentes.

    Esta función asegura que todos los parámetros de entrada tengan dimensiones compatibles
    y tipos apropiados para ser utilizados en la clase `Annotations`.

    Args:
        onset (float | array-like): Tiempo(s) de inicio de las anotaciones.
        duration (float | array-like): Duración(es) de las anotaciones.
        description (str | array-like): Descripción(es) asociada(s) a cada anotación.
        ch_names (None | str | array-like): Nombre(s) de canal o lista de canales asociados.
            Puede ser:
                - None (se asigna como array de None),
                - una cadena (para una sola anotación),
                - una lista de cadenas,
                - una lista de listas (se convierte a cadenas separadas por comas).

    Returns:
        tuple: Cuádrupla de arrays 1D:
            - onset (np.ndarray[float])
            - duration (np.ndarray[float])
            - description (np.ndarray[str])
            - ch_names (np.ndarray[object])

    Raises:
        ValueError: Si los arrays no son unidimensionales o tienen longitudes incompatibles.
        TypeError: Si los valores no pueden convertirse a los tipos esperados.
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
    
    if ch_names is not None:
        # Convertimos a array y manejamos dimensiones
        ch_names = np.array(ch_names, dtype=object)  # dtype=object para permitir listas
               
        # Si tenemos una lista de listas de canales
        if ch_names.ndim == 1 and len(ch_names) > 0 and isinstance(ch_names[0], (list, np.ndarray)):
            # Convertimos cada lista de canales a cadena separada por comas
            ch_names = np.array([','.join(map(str, ch)) for ch in ch_names])

        # Si es escalar, convertimos a lista de cadenas
        if ch_names.ndim == 0:
            ch_names = np.array([ch_names])

        # Validación de longitud
        if ch_names.shape[0] == 1 and len(onset) > 1:
            ch_names = np.repeat(ch_names, len(onset))
        elif ch_names.shape[0] != len(onset):
            raise ValueError(
                f"ch_names debe tener la misma longitud que onset. "
                f"Onset: {len(onset)}, ch_names: {len(ch_names)}"
            )
    else:
        # Creamos array de None si no se proporcionan canales
        ch_names = np.repeat(None, len(onset))

    params = list(map(len, [onset, duration, description, ch_names]))

    if len(set(params)) != 1:
        raise ValueError(f"Todos los parámetros deben poseer el mismo largo. Largos hallados: {params}")

    return onset, duration, description, ch_names

def _sort(self):
    """
    Ordena las anotaciones internamente por tiempo de inicio (`onset`) en orden ascendente.

    Esta función modifica los atributos `self.onset`, `self.duration`, `self.description` y
    `self.ch_names` de forma in-place para mantener la coherencia temporal y estructural.

    Notes:
        - Usa `np.argsort` para obtener los índices de ordenamiento.
        - El ordenamiento garantiza que las anotaciones estén cronológicamente alineadas.
        - También reordena `ch_names` si no es None.
    """
    orden = np.argsort(self.onset)
    self.onset = self.onset[orden]
    self.duration = self.duration[orden]
    self.description = self.description[orden]
    if self.ch_names is not None:
        self.ch_names = self.ch_names[orden]