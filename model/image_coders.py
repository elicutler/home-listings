
import logging
import urllib.request
import os
import ast
import numpy as np
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
        self.local_path = (
            local_path if isinstance(local_path, Path) else Path(local_path)
        )
                 
    def img_to_arr_list(self, del_img_on_exit:bool=True) -> list:
        if self.local_path.name not in os.listdir(self.local_path.parent):
            self.download_img()
            
        img = Image.open(self.local_path) 
        img_arr = np.asarray(img)
        img_arr_list = img_arr.tolist()
        
        if del_img_on_exit:
            os.remove(self.local_path)
            
        return img_arr_list
    
    def download_img(self) -> None:
        urllib.request.urlretrieve(self.url, self.local_path)
        
        
class ImageDecoder:
    '''
    Convert image from nested list -> numpy array -> PIL.Image.Image
    '''
    def __init__(self, arr_list:list):
        self.arr_list = arr_list
        
    def arr_list_to_arr(self) -> np.array:
        img_arr = np.array(ast.literal_eval(self.arr_list), dtype='uint8')
        return img_arr
    
    def arr_to_image(self, img_arr:np.array) -> Image.Image:
        img = Image.fromarray(img_arr)
        return img
        
    def arr_list_to_image(self) -> Image.Image:
        img_arr = self.arr_list_to_arr()
        img = self.arr_to_image(img_arr)
        return img
