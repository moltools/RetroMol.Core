"""
Author:         David Meijer
Licence:        MIT License
Description:    General utility functions.
Dependencies:   python>=3.10
"""
import logging 
import os.path as osp
import typing as ty
from enum import Enum 

class LoggerLevel(Enum):
    DEBUG       = logging.DEBUG 
    INFO        = logging.INFO
    WARNING     = logging.WARNING
    ERROR       = logging.ERROR
    CRITICAL    = logging.CRITICAL

def config_logger(
    name: str,
    level: int = logging.INFO,
    logger_file_path: ty.Optional[str] = None,
    add_stream_handler: bool = True,
    initialize_with: ty.Optional[str] = None
) -> logging.Logger:
    """
    Configures a logger with the given name and level.

    NOTE: logging level is set to INFO by default and is the same for the stream
    and file handler. This means that the logger will log all messages with a
    level of INFO or higher to both the stream and file handler.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int, optional
        Level of the logger, by default logging.INFO.
    logger_file_path : ty.Optional[str], optional
        Path to the log file, by default None.
    add_stream_handler : bool, optional
        Whether to add a stream handler to the logger, by default True.
    initialize_with : ty.Optional[str], optional
        Message to initialize the logger with, by default None.
    
    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a custom formatter for the logger that displays the time, name,
    # level and message of the log.
    log_format = "[%(asctime)s; %(name)s; %(levelname)s] %(message)s"
    formatter = logging.Formatter(log_format)

    # Add stream handler to logger.
    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(stream_handler)

    # Add file handler to logger if file path is given for log file. 
    if logger_file_path:

        # Check if file exists. If it exists, overwrite it with warning message.
        # If the file does not exists, create it.
        if osp.isfile(logger_file_path):
            msg = f"Log file {logger_file_path} already exists. Overwriting it."
            logger.warning(msg)

        open(logger_file_path, "a").close()

        file_handler = logging.FileHandler(logger_file_path, "w+")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    # Initialize logger with message, if given.
    if initialize_with: 
        logger.info(initialize_with)

    return logger
