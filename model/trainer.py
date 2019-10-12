
import os
import logging
import torch
import torch.utils.data
import pandas as pd

from typing import Any
from gen_utils import set_logger_defaults
from constants import COLUMN_ORDER, TAB_COLS, TAB_CAT_COLS, TEXT_COLS, IMG_COLS
from models import PyTorchModel
from feature_enger import FeatureEnger

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class Trainer:
    
    def __init__(
        self, model:PyTorchModel, loss_func:Any, optimizer:Any, 
        **optimizer_kwargs
    ):
        self.model = model
        self.loss_func = loss_func(reduction='sum')
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def train(self, epochs:int) -> None:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        total_train_loss_avg = []
        
        for e in range(1, epochs+1):
            self.model.train()
            
            epoch_loss = 0
            
            for batch in self.train_loader:
                x_tab, x_text, x_img, y = batch
                
                x_tab.to(device)
                x_text.to(device)
                x_img.to(device)
                
                self.optimizer.zero_grad()
                y_pred = self.model.forward(x_tab, x_text, x_img)
                
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.data.item()
                
            logger.info(f'epoch {e}/{epochs} loss: {epoch_loss}')
            total_train_loss_avg.append(epoch_loss / len(self.train_loader))
            
            # TODO: calc test_loss

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
            
        df.columns = COLUMN_ORDER
        
        feature_enger = FeatureEnger(
            df_tab=df[TAB_COLS], df_text=df[TEXT_COLS], df_img=df[IMG_COLS]
        )
        feature_enger.one_hot_encode_cat_cols()
        feature_enger.img_arr_list_to_arr()
        
        df_y = df[outcome]
        
        
        
        x_tab = torch.from_numpy(feature_enger.df_tab.values).float().squeeze()
        x_desc = torch.from_numpy(df[].values).float().squeeze()
        x_img = torch.from_numpy(df[IMG_COLS].values).float().squeeze()
        y = torch.from_numpy(df[outcome].values).float().squeeze()
        
        dataset = torch.utils.data.TensorDataset(x_tab, x_desc, x_img, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        if which_loader == 'train':
            self.train_loader = dataloader
        elif which_loader == 'val':
            self.val_loader == dataloader
        elif which_loader == 'test':
            self.test_loader == dataloader
        
        
        
        
        
    