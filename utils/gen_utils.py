
import sys; sys.path.insert(0, '.')
import logging
import pandas as pd
import numpy as np

from typing import Union, Any
from datetime import datetime
from constants import COLUMN_ORDER

def set_logger_defaults(
    logger:logging.Logger, level:int=logging.INFO, addFileHandler:bool=False
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
    
logger = logging.getLogger(__name__)
set_logger_defaults(logger)


def get_unique_id(
    id_type:type, offset_base:Union[int, str]=2019
) -> Union[int, str]:
    now_str = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    offset = str(offset_base).ljust(len(now_str), '0')
    unique_id = id_type(int(now_str) - int(offset))
    return unique_id 


def put_columns_in_order(in_df:pd.DataFrame) -> pd.DataFrame:
    df = in_df.copy()
    
    assert len(df.columns) == len(set(df.columns)), 'df contains duplicate columns'
    assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER)), 'COLUMN_ORDER contains duplicate columns' 
    assert len(set(df.columns) - set(COLUMN_ORDER)) == 0, 'df contains columns not in COLUMN_ORDER'
    
    if len(set(COLUMN_ORDER) - set(df.columns)) > 0:
        missing_cols = list(set(COLUMN_ORDER) - set(df.columns))
        logger.warning(
            'The following cols in COLUMN_ORDER are missing from df'
            f' and will be added: {missing_cols}'
        )
        for col in missing_cols:
            df[col] = np.nan
            
    df = df[COLUMN_ORDER]
    return df

def filter_df_missing_col(in_df:pd.DataFrame, col:str) -> pd.DataFrame:
    df = in_df.copy()
    df = df[df[col].notna()]
    return df