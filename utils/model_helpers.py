
import logging
import torch

from typing import Union
from pathlib import Path
from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class ModelHelpers:

    @staticmethod
    def _make_data_loader(
        path:Union[Path, str], batch_size:int, concat_all:bool=True,
        data_file:str=None
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
             
        
        
        
        
    