
import logging
import urllib.request
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


class ImageCoder:
    '''
    Download image, convert image to list representation,
    reconstruct image.
    '''
        
    @staticmethod
    def img_url_to_arr_list(
        url:str, local_path:Union[str, Path]
    ) -> list:
        urllib.request.urlretrieve(url, local_path)
        img = Image.open(img_local) 
        img_arr = np.asarray(img)
        img_arr_list = img_arr.tolist()
        return img_arr_list
    
    