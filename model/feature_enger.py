
import logging
import pandas as pd
import numpy as np

from typing import Union, Optional, Tuple

from gen_utils import set_logger_defaults
from constants import TAB_CAT_FEATURES, TAB_DT_FEATURES, IMG_FEATURE
from image_coders import ImageDecoder

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class FeatureEnger:
        
    def __init__(
        self, df_tab:Optional[Union[pd.DataFrame, np.array]]=None,
        ser_text:Optional[Union[pd.Series, np.array]]=None,
        ser_img:Optional[Union[pd.Series, np.array]]=None,
        img_decoder:Optional[ImageDecoder]=None
    ):
        self.df_tab = None if df_tab is None else df_tab.copy()
        self.ser_text = None if ser_text is None else ser_text.copy()
        self.ser_img = None if ser_img is None else ser_img.copy()
        self.df_tab_cols_train = None
        self.df_img_max_dim_train = None
        
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
            train_cols_not_in_df = [c for c in train_cols if c not in df.columns]
            for c in train_cols_not_in_df:
                df[c] = 0
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
            
    def fill_all_tab_nans(self, how:str='empirical_dist') -> None:
        assert self.df_tab is not None, 'first call self.set_df_tab()'
        assert how in ['empirical_dist']

        for c in self.df_tab.columns:
            if self.df_tab[c].isna().mean() == 1:
                logger.warning(f'{c} has no present values. Imputing all zeros.')
                self.df_tab[c] = 0
            elif how == 'empirical_dist':
                self._fill_nans_from_empirical_dist(c)

    def _fill_nans_from_empirical_dist(self, col:str) -> None:
        n_missing_rows = self.df_tab.loc[self.df_tab[col].isna(), col].shape[0]
        present_rows = self.df_tab.loc[self.df_tab[col].notna(), col]
        
        self.df_tab.loc[self.df_tab[col].isna(), col] = (
            np.random.choice(present_rows, size=n_missing_rows)
        )        
        
    def fill_all_img_nans(self) -> None:
        assert self.ser_img is not None, 'first call self.set_ser_img()'
        
        self.ser_img = self.ser_img.apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros((1, 1, 1))
        )
        
    def img_arr_list_str_to_arr(self) -> None:
        assert self.ser_img is not None, 'first call self.set_ser_img()'

        ser_img = self.ser_img.apply(lambda x: ImageDecoder(x).arr_list_str_to_arr())
        self.ser_img = ser_img
    
    def resize_img_arr(self, mode:str='train') -> None:
        assert mode in ['train', 'val', 'test']
        if mode == 'train':
            self.df_img_max_dim_train = self._get_arr_max_dims()
        elif mode in ['val', 'test']:
            assert self.df_img_max_dim_train is not None, (
                'Need to run in train mode first'
            )
        for i, img in zip(self.ser_img.index, self.ser_img):
            arr_max_dim = np.zeros(self.df_img_max_dim_train)
            arr_max_dim[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            # if val/test img has greater dim than max from train, next step will crop it
            arr_max_dim = arr_max_dim[
                :self.df_img_max_dim_train[0],
                :self.df_img_max_dim_train[1],
                :self.df_img_max_dim_train[2]
            ]
            self.ser_img.loc[i] = arr_max_dim
        logger.info(f'CHECK SER_IMG OBS SHAPE: {self.ser_img.loc[0].shape}')
        
    def _get_arr_max_dims(self) -> tuple:
        n_dims = len(self.ser_img.values[0].shape)
        max_dims = [None]*n_dims
        
        for i in range(n_dims):
            max_dims[i] = self.ser_img.apply(lambda x: x.shape[i]).max()
            
        return tuple(max_dims)
            
        
    def set_df_tab(
        self, df_tab:Union[pd.DataFrame, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.df_tab is None, (
                'self.df_tab already exists. Run with overwrite=True to allow ovewriting'
            )
        self.df_tab = df_tab.copy()
        
    def set_ser_text(
        self, ser_text:Union[pd.Series, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.ser_text is None, (
                'self.ser_text already exists. Run with overwrite=True to allow ovewriting'
            )
        self.ser_text = ser_text.copy()
        
    def set_ser_img(
        self, ser_img:Union[pd.Series, np.array], overwrite:bool=False
    ) -> None:
        if not overwrite:
            assert self.ser_img is None, (
                'self.ser_img already exists. Run with overwrite=True to allow ovewriting'
            )
        self.ser_img = ser_img.copy()
        
        
        
        
        