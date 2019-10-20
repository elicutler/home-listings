
import sys; sys.path.insert(0, '../utils')
import torch
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, x_tab_input_dim, x_text_input_dim, x_img_input_dim):
        super().__init__()
        
        self.x_tab_l0_fc = nn.Linear(x_tab_input_dim, x_tab_input_dim) 
        self.x_tab_l0_dropout = nn.Dropout(p=0.2)
        self.x_tab_l0_relu = nn.ReLU()
        self.x_tab_l1_fc = nn.Linear(x_tab_input_dim, 1)
        
        print(f'x_img_input_dim: {x_img_input_dim}')
        
        
    
    def forward(
        self, x_tab:torch.Tensor, x_text:torch.Tensor, x_img:torch.Tensor
    ) -> torch.Tensor:
                    
        x_tab = self.x_tab_l0_fc(x_tab)
        x_tab = self.x_tab_l0_dropout(x_tab)
        x_tab = self.x_tab_l0_relu(x_tab)
        x_tab = self.x_tab_l1_fc(x_tab)
        return x_tab
    