
import logging

from typing import Union, Any
from datetime import datetime

def set_logger_defaults(
    logger:logging.Logger, 
    level:int=logging.INFO, 
    addFileHandler:bool=False
) -> None:
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    streamHandler.setFormatter(formatter)
    
    if addFileHandler:
        fileHandler = logging.FileHandler(logger.name, mode='w')
        fileHandler.setLevel(level)
        fileHandler.setFormatter(formatter)
    
    logger.handlers = []
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler) if addFileHandler else None
    

def get_unique_id(
    id_type:type, offset_base:Union[int, str]=2019
) -> Union[int, str]:
    now_str = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    offset = str(offset_base).ljust(len(now_str), '0')
    unique_id = id_type(int(now_str) - int(offset))
    return unique_id 