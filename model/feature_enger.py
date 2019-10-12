
import logging
import pandas as pd

from typing import Union

from gen_utils import set_logger_defaults
from constants import TAB_CAT_COLS

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class FeatureEnger:
        
    @staticmethod 
    def one_hot_encode_cat_cols(tab_df) -> pd.DataFrame:
        non_cat_cols = [c for c in tab_df.columns if c not in TAB_CAT_COLS]
        non_dummies_df = tab_df[non_cat_cols].copy()
        
        dummies_df = pd.get_dummies(tab_df[TAB_CAT_COLS], dummy_na=True)
        
        df = pd.concat([non_dummies_df, dummies_df], axis='columns', sort=False)
        return df
        
        
        