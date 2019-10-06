
import logging
import urllib.request
import os
import ast
import numpy as np
import pandas as pd
import ast

from typing import Union
from pathlib import Path
from PIL import Image
from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class ImageEncoder:
    '''
    Download image and convert image to list representation,
    '''
    def __init__(self, url:str, local_path:Union[Path, str]):
        self.url = url
        self.local_path = Path(local_path) if not isinstance(local_path, Path) else local_path
                 
    def img_to_arr_list(self) -> list:
        if self.local_path.name not in os.listdir(self.local_path.parent):
            self.download_img()
            
        img = Image.open(self.local_path) 
        img_arr = np.asarray(img)
        img_arr_list = img_arr.tolist()
        return img_arr_list
    
    def download_img(self) -> None:
        urllib.request.urlretrieve(self.url, self.local_path)
        