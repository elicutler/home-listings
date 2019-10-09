
import sys; sys.path.insert(0, '../utils')
import logging
import torch

from typing import Any
from pathlib import Path
from gen_utils import set_logger_defaults
from constants import COLUMN_ORDER, TAB_COLS
from models import PyTorchModel

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class Trainer:
    
    def __init__(
        self, model:PyTorchModel, loss_func:Any, optimizer:Any
    ):
        self.model = model
        self.loss_func = loss_func(reduction='sum')
        self.optimizer = optimizer
        self.train_loader = None
        self.val_loader = None
    
    def train(self, epochs:int) -> None:
        
        assert self.train_loader is not None, (
            'self.train_loader has not not been initialized.'
            ' Need to call self._make_train_loader() first'
        )
        assert self.val_loader is not None, (
            'self.val_loader has not been initialized.'
            ' Need to call self._make_val_loader() first.'
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        
        total_train_loss_avg = []
        
        for e in range(1, epochs+1):
            self.model.train()
            
            epoch_loss_sum = 0
            
            for batch in self.train_loader:
                X_tab, X_desc, X_img, y = batch
                
                X_tab.to(device)
                X_desc.to(device)
                X_img.to(device)
                
                self.optimizer.zero_grad()
                y_pred = self.model.forward(X_tab, X_desc, X_img)
                
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.data.item()
                
            total_train_loss_avg += epoch_loss / len(self.train_loader)
            
            # TODO: calc test_loss
            
    def _make_train_loader(self, *args, **kwargs) -> None:
        self.train_loader = self._make_data_loader(*args, **kwargs)
        
    def _make_val_loader(self, *args, **kwargs) -> None:
        self.val_loader = self._make_test_laoder(*args, **kwargs)

    @staticmethod
    def _make_data_loader(
        path:Union[Path, str], batch_size:int, outcome:str,
        concat_all:bool=True, data_file:str=None
    ) -> torch.utils.data.DataLoader:
        
        assert int(concat_all) + int(data_file is not None) == 1, (
            'Failed to satisfy: concat_all==True XOR data_file is not None'
        )        
        path = path if isinstance(path, Path) else Path(path)
        
        if concat_all:
            df_list = [
                f: pd.read_csv(path/f, header=None, names=None)
                for f in os.listdir(path) if f.endswith('.csv')
            ]
            df = pd.concat(df_list, ignore_index=True, sort=False)
            
        elif data_file is not None:
            df = pd.read_csv(path/data_file, header=None, names=None)
            
        df.columns = COLUMN_ORDER
        
        X_tab = torch.from_numpy(df[[TAB_COLS]].values).float().squeeze()
        X_desc = torch.from_numpy(df[[DESC_COLS]].values).float().squeeze()
        X_img = torch.from_numpy(df[[IMG_COLS]].values).float().squeeze()
        y = torch.from_numpy(df[[outcome]].values).float().squeeze()
        
        dataset = torch.utils.data.TensorDataset(X_tab, X_desc, X_img, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return dataloader
        
        
        
        
    