
import sys; sys.path.insert(0, '../utils')
import torch
import torch.nn as nn


class PyTorchModel(nn.Module):
    '''
    PyTorch model using text, image, and tabular features.
    '''
    def __init__(self, input_dim):
        super().__init__()
        
#         self.x_tab_l0_fc = None # initialize dynamically based on input 
#         self.x_tab_l0_relu = nn.ReLU()
#         self.x_tab_l1_fc = None
        self.x_tab_l0_fc = nn.Linear(input_dim, input_dim) # initialize dynamically based on input 
        self.x_tab_l0_relu = nn.ReLU()
        self.x_tab_l1_fc = nn.Linear(input_dim, 1)
    
    def forward(
        self, x_tab:torch.Tensor, x_text:torch.Tensor, x_img:torch.Tensor
    ) -> torch.Tensor:

#         if self.x_tab_l0_fc is None:
#             self._init_x_tab_fc_layers(x_tab)
                    
        x_tab = self.x_tab_l0_fc(x_tab)
        x_tab = self.x_tab_l0_relu(x_tab)
        x_tab = self.x_tab_l1_fc(x_tab)
        return x_tab
    
#     def _init_x_tab_fc_layers(x_tab:torch.Tensor) -> None:
#         input_dim = x_tab.size()[1]
#         self.x_tab_l0_fc = nn.Linear(input_dim, input_dim)
#         self.x_tab_l1_fc = nn.Linear(input_dim, 1)
        
#         print(f'model is:\n{self}')