import numpy as np

def _checking(onset, duration, description, ch_names):
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

    ch_names = np.array(ch_names, dtype=str)
    if ch_names.ndim == 0 or ch_names.shape == (1,):
        ch_names = np.repeat(ch_names, len(onset))
    if ch_names.ndim != 1:
        raise ValueError("Ch_names debe ser un array de 1 dimensión")
    
    if not len(onset) == len(duration) == len(description) == len(ch_names):
        raise ValueError(f"Todos los parámetros deben poseer el mismo largo {len(onset)}")
    

    return onset, duration, description, ch_names

def _sort(self):
    orden = np.argsort(self.onset)
    self.onset = self.onset[orden]
    self.duration = self.durarion[orden]
    self.description = self.description[orden]
    if self.ch_names is not None:
        self.ch_names = self.ch_names[orden]