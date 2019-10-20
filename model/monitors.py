
import logging
import pandas as pd

from typing import Union

from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

def check_missing_pcts(df:Union[pd.DataFrame, pd.Series]) -> None:
    if isinstance(df, pd.Series):
        if df.name is None:
            df.name = 'series'
        df = pd.DataFrame(df)
        
    cols_missing_pcts = [(c, df[c].isna().mean()*100) for c in df.columns]
    cols_missing_pcts.sort(key=lambda x: x[1], reverse=True)

    for col, missing_pct in cols_missing_pcts:
        logger.info(f'{col} missing %: {missing_pct}')

        