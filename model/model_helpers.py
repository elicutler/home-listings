
import sys; sys.path.insert(0, '../utils')
import logging
import torch

from typing import Union
from pathlib import Path
from gen_utils import set_logger_defaults
from constants import COLUMN_ORDER, TAB_COLS
from models import PyTorchModel

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class ModelHelpers:
    
    def __init__(self, model:Union[PyTorchModel], loss_func:str):
        self.model = model
        self.loss_func = loss_func
    
    def train() -> None:
        pass

    @staticmethod
    def _make_data_loader(
        path:Union[Path, str], batch_size:int, outcome:str,
        concat_all:bool=True, data_file:str=None
    ) -> torch.utils.data.DataLoader:
        
        logger.info('Making data loader')
        
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
        
        
        
        
    