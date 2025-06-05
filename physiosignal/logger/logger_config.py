import logging

def log_config(see_log):
    if not logging.getLogger().hasHandlers():
        if see_log:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s [%(name)s.%(funcName)s]: %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S"
            )
        else:
            logging.basicConfig(handlers=[logging.NullHandler()])