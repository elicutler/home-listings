
import logging
import pandas as pd
import numpy as np

from typing import Union, Optional

from gen_utils import set_logger_defaults
from constants import TAB_CAT_FEATURES, TAB_DT_FEATURES
from image_coders import ImageDecoder

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class FeatureEnger:
        
    def __init__(
        self, df_tab:Optional[Union[pd.DataFrame, np.array]]=None,
        df_text:Optional[Union[pd.DataFrame, np.array]]=None,
        df_img:Optional[Union[pd.DataFrame, np.array]]=None,
        img_decoder:Optional[ImageDecoder]=None
    ):
        self.df_tab = None if df_tab is None else df_tab.copy()
        self.df_text = None if df_text is None else df_text.copy()
        self.df_img = None if df_img is None else df_img.copy()
        self.df_tab_cols_train = None
        
    def one_hot_encode_cat_cols(
        self, mode:str='train', train_cols=Optional[Union[pd.Index, np.array, list]],
    ) -> None:
        assert self.df_tab is not None, 'first call self.set_df_tab()'
        assert mode in ['train', 'val', 'test']
        if mode in ['val', 'test']:
            assert train_cols is not None, 'Need to provide columns from training set'
        
        non_cat_cols = [
            c for c in self.df_tab.columns if c not in TAB_CAT_FEATURES
        ]
        non_dummies_df = self.df_tab[non_cat_cols].copy()
        
        dummies_df = pd.get_dummies(self.df_tab[TAB_CAT_FEATURES], dummy_na=True)
        
        df = pd.concat([non_dummies_df, dummies_df], axis='columns', sort=False)
        
        if mode == 'train':
            self.df_tab = df
            self.df_tab_cols_train = self.df_tab.columns
        elif mode in ['val', 'test']:
            self.df_tab = df[train_cols]
            
    def get_df_tab_cols_train(self) -> Union[pd.Index, np.array, list]:
        assert self.df_tab_cols_train is not None, (
            'df_tab_cols_train not set. First call one_hot_encode_cat_cols()'
            ' with mode="train"'
        )
        return self.df_tab_cols_train
        
    def datetime_cols_to_int(self) -> None:
        assert self.df_tab is not None, 'first call self.set_df_tab()'
        
        for c in TAB_DT_FEATURES:
            self.df_tab[c] = pd.to_datetime(self.df_tab[c]).astype('int') // 10**9           
            
    def fill_all_nans(self, how:str='empirical_dist') -> None:
        assert self.df_tab is not None, 'first call self.set_df_tab()'
        assert how in ['empirical_dist']

        for c in self.df_tab.columns:
            if self.df_tab[c].isna().mean() == 1:
                logger.warning(f'{c} has no present values. Imputing all zeros.')
                self.df_tab[c] = 0
            elif self.df_tab[c].isna().sum() == 0:
                continue
            else:
                if how == 'empirical_dist':
                    self._fill_nans_from_empirical_dist(c)

    def _fill_nans_from_empirical_dist(self, col:str) -> None:
        n_missing_rows = self.df_tab.loc[self.df_tab[col].isna(), col].shape[0]
        present_rows = self.df_tab.loc[self.df_tab[col].notna(), col]
        
        self.df_tab.loc[self.df_tab[col].isna(), col] = (
            np.random.choice(present_rows, size=n_missing_rows)
        )        
        
    def img_arr_list_to_arr(self) -> None:
        assert self.df_img is not None, 'first call self.set_df_img()'
        
        df_img = self.df_img.applymap(lambda x: ImageDecoder(x).arr_list_to_arr())
        self.df_img = df_img
    
    def set_df_tab(
        self, df_tab:Union[pd.DataFrame, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.df_tab is None, (
                'self.df_tab already exists. Run with overwrite=True to allow ovewriting'
            )
        self.df_tab = df_tab.copy()
        
    def set_df_text(
        self, df_text:Union[pd.DataFrame, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.df_text is None, (
                'self.df_text already exists. Run with overwrite=True to allow ovewriting'
            )
        self.df_text = df_text.copy()
        
    def set_df_img(
        self, df_img:Union[pd.DataFrame, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.df_img is None, (
                'self.df_img already exists. Run with overwrite=True to allow ovewriting'
            )
        self.df_img = df_img.copy()
        
        
        
        
        