
import logging
import pandas as pd

from typing import Union

from gen_utils import set_logger_defaults
from constants import TAB_CAT_COLS

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class FeatureEnger:
    
    def __init__(
        self, in_tab_df:pd.DataFrame, in_desc_df:pd.DataFrame, 
        in_img_df:pd.DataFrame
    ):
        self.in_tab_df = in_tab_df
        self.in_desc_df = in_desc_df
        self.in_img_df = in_img_df
        
    def eng_tab_df(self) -> pd.DataFrame:
        tab_df = self.one_hot_encode_cat_cols()
        return tab_df
        
    def one_hot_encode_cat_cols(self) -> pd.DataFrame:
        non_cat_cols = [c for c in self.in_tab_df.columns if c not in TAB_CAT_COLS]
        non_dummies_df = self.in_tab_df[non_cat_cols].copy()
        
        dummies_df = pd.get_dummies(tab_df[TAB_CAT_COLS], dummy_na=True)
        
        df = pd.concat([non_dummies_df, dummies_df], axis='columns', sort=False)
        return df
        
        
        