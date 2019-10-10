
import sys; sys.path.insert(0, '../utils')
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, x_tab_input_dim):
        super().__init__()
        
        self.x_tab_l0_fc = nn.Linear(x_tab_input_dim, x_tab_input_dim)
        self.x_tab_l0_relu = nn.ReLU()
        self.x_tab_l1_fc = nn.Linear(x_tab_input_dim, 1)
    
    def forward(self, x_tab, x_desc, x_img):
        x_tab = self.x_tab_l0_fc(x_tab)
        x_tab = self.x_tab_l0_relu(x_tab)
        x_tab = self.x_tab_l1_fc(x_tab)
        return x_tab