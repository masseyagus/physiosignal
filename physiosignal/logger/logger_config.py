import logging

def log_config(see_log):
    """
    Configura el sistema de logging para el paquete.

    Esta función inicializa la configuración global del módulo `logging` de Python.
    Si `see_log` es True, se activa el registro de mensajes de nivel INFO y superiores
    con un formato detallado. Si es False, se suprime la salida de logs utilizando un `NullHandler`.

    La configuración solo se aplica si no existen handlers previos en el logger raíz,
    para evitar duplicación de mensajes cuando se importan múltiples módulos.

    Args:
        see_log (bool): Si es True, se habilita la salida de mensajes de logging;
                        si es False, se desactiva la salida visible de logs.
    """
    if not logging.getLogger().hasHandlers():
        if see_log:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s [%(name)s.%(funcName)s]: %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S"
            )
        else:
            logging.basicConfig(handlers=[logging.NullHandler()])