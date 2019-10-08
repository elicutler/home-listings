
import sys; sys.path.insert(0, '../utils')
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, *_, *__):
        super().__init__()
    
    def forward(self, x):
        print(x)