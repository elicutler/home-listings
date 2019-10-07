
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, *_, *__):
        super().__init__()
        pass
    
    def forward(self, text, img, tabular):
        pass