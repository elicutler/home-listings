
import logging

def set_logger_defaults(
    logger:logging.Logger, 
    level:int=logging.INFO, 
    addFileHandler:bool=False
) -> None:
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
    