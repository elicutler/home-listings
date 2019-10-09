
import sys; sys.path.insert(0, '../utils')
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, *_, *__):
        super().__init__()
    
    def forward(self, X_tab, X_desc, X_img):
        pass