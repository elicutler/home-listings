
import os
import logging
import torch
import torch.utils.data
import pandas as pd

from typing import Any
from gen_utils import (
    set_logger_defaults, put_columns_in_order, remove_rows_missing_y
)
from constants import (
    COLUMN_ORDER, TAB_FEATURES, TAB_CAT_FEATURES, TEXT_FEATURE, IMG_FEATURE
)
from models import PyTorchModel
from feature_enger import FeatureEnger
from monitors import check_missing_pcts

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class Trainer:
    
    def __init__(self):
        
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.df_tab_cols_train = None
        
    def make_data_loader(
        self, which_loader:str, path:str, batch_size:int, outcome:str,
        concat_all:bool=True, data_file:str=None
    ) -> None:
        
        assert which_loader in ['train', 'val', 'test']
        
        assert int(concat_all) + int(data_file is not None) == 1, (
            'Failed to satisfy: concat_all==True XOR data_file is not None'
        )        
        if concat_all:
            df_list = [
                pd.read_csv(f'{path}/{f}', header=None, names=None)
                for f in os.listdir(path) if f.endswith('.csv')
            ]
            df = pd.concat(df_list, ignore_index=True, sort=False)
            
        elif data_file is not None:
            df = pd.read_csv(path/data_file, header=None, names=None)
            
        assert len(df.columns) == len(COLUMN_ORDER)
        df.columns = COLUMN_ORDER
        
        logger.info('FOR TESTING, eliminate rows')
        df = df.iloc[:6, :]
        
        feature_enger = FeatureEnger(
            df_tab=df[TAB_FEATURES], ser_text=df[TEXT_FEATURE], 
            ser_img=df[IMG_FEATURE]
        )
        df_y = df[outcome]
        
        if which_loader == 'train':
            feature_enger.one_hot_encode_cat_cols(which_loader)
            self.df_tab_cols_train = feature_enger.get_df_tab_cols_train()
            
        elif which_loader in ['val', 'test']:
            feature_enger.one_hot_encode_cat_cols(
                mode=which_loader, train_cols=self.df_tab_cols_train
            )
            
        feature_enger.datetime_cols_to_int()
        feature_enger.fill_all_tab_nans()
        feature_enger.fill_all_img_nans()
        feature_enger.img_arr_list_str_to_arr()
        
        assert (
            feature_enger.df_tab.shape[0] 
            == feature_enger.ser_text.shape[0]
            == feature_enger.ser_img.shape[0]
            == df_y.shape[0]
        )
        remove_rows_missing_y(
            df_y, other_dfs=[
                feature_enger.df_tab, feature_enger.ser_text, feature_enger.ser_img
            ]
        )
                
        x_tab = torch.from_numpy(feature_enger.df_tab.values).float()
        x_text = x_tab # FOR TESTING
        x_img = torch.Tensor(feature_enger.ser_img).float()
        y = torch.from_numpy(df_y.values).float()
        
        print('CHECK DIMS')
        print(f'x_tab DIM: {x_tab.size()}')
        print(f'x_text DIM: {x_text.size()}')
        print(f'x_img DIM: {x_img.size()}')
        print(f'y DIM: {y.size()}')
                
        dataset = torch.utils.data.TensorDataset(x_tab, x_text, x_img, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        logger.info(f'N obs loaded in {which_loader}_loader: {len(dataset)}')
        
        if which_loader == 'train':
            self.train_loader = data_loader
        elif which_loader == 'val':
            self.val_loader = data_loader
        elif which_loader == 'test':
            self.test_loader = data_loader
            
    def get_input_dims(self) -> tuple:
        assert self.train_loader is not None

        for batch in self.train_loader:
            x_tab, x_text, x_img, y = batch
            x_tab_input_dim = x_tab.size()[1]
            x_text_input_dim = x_text.size()[1:]
            x_img_input_dim = x_img.size()[1:]
            break
            
        return x_tab_input_dim, x_text_input_dim, x_img_input_dim
    
    def set_model(
        self, model:PyTorchModel, loss_func_cls:Any, optimizer_cls:Any, 
        **optimizer_kwargs
    ) -> None:
        self.model = model
        self.loss_func = loss_func_cls(reduction='sum')
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def train(self, epochs:int) -> None:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        total_train_loss_avg = []
        total_val_loss_avg = []
        
        for e in range(1, epochs+1):
            self.model.train()
            
            epoch_train_loss_sum = 0
            epoch_train_loss_nobs = 0
                       
            for batch in self.train_loader:
                x_tab, x_text, x_img, y = batch
                
                x_tab = x_tab.to(device)
                x_text = x_text.to(device)
                x_img = x_img.to(device)
                y = y.to(device)
                
                self.optimizer.zero_grad()
                y_pred = self.model.forward(x_tab, x_text, x_img)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss_sum += loss.data.item()
                epoch_train_loss_nobs += y_pred.size()[0]
                    
            epoch_train_loss_avg = epoch_train_loss_sum / epoch_train_loss_nobs
                
            logger.info(f'epoch {e}/{epochs} train loss: {epoch_train_loss_avg}')
            total_train_loss_avg.append(epoch_train_loss_avg)
            
            with torch.no_grad():
                self.model.eval()
                
                epoch_val_loss_sum = 0
                epoch_val_loss_nobs = 0
                
                for batch in self.val_loader:
                    x_tab, x_text, x_img, y = batch

                    x_tab = x_tab.to(device)
                    x_text = x_text.to(device)
                    x_img = x_img.to(device)
                    y = y.to(device)
                    
                    y_pred = self.model.forward(x_tab, x_text, x_img)
                    loss = self.loss_func(y_pred, y)
                    
                    epoch_val_loss_sum += loss.data.item()
                    epoch_val_loss_nobs += y_pred.size()[0]
                    
                epoch_val_loss_avg = epoch_val_loss_sum / epoch_val_loss_nobs

                logger.info(f'epoch {e}/{epochs} val loss: {epoch_val_loss_avg}')
                total_train_loss_avg.append(epoch_val_loss_avg)
                    


        
        
        
        
    