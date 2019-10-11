
import logging
import pandas as pd

from typing import Union

from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class FeatureEnger:
    
    def __init__(
        self, in_tab_df:pd.DataFrame, in_desc_df:pd.DataFrame, 
        in_img_df:pd.DataFrame, tab_cat_cols:Union[list, str]
    ):
        self.in_tab_df = in_tab_df
        self.in_desc_df = in_desc_df
        self.in_img_df = in_img_df
        self.tab_cat_cols = (
            tab_cat_cols if isinstance(list, tab_cat_cols)
            else [tab_cat_cols]
        )
        
    def eng_tab_df(self) -> pd.DataFrame:
        tab_df = self.one_hot_encode_cat_cols()
        
        