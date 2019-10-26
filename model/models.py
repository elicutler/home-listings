
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
        
        self.x_img_l0_conv2d = nn.Conv2d(3, 6, 5)
        self.x_img_l0_dropout = nn.Dropout(p=0.5)
        self.x_img_l0_relu = nn.ReLU()
        self.x_img_l0_flatten = nn.Flatten()
    
    def forward(
        self, x_tab:torch.Tensor, x_text:torch.Tensor, x_img:torch.Tensor
    ) -> torch.Tensor:
                    
        x_tab = self.x_tab_l0_fc(x_tab)
        x_tab = self.x_tab_l0_dropout(x_tab)
        x_tab = self.x_tab_l0_relu(x_tab)
        x_tab = self.x_tab_l1_fc(x_tab)

        # for img, move n_channels dim to front
        x_img = x_img.view(-1, x_img.size()[3], x_img.size()[1], x_img.size()[2])
        x_img = self.x_img_l0_conv2d(x_img)
        x_img = self.x_img_l0_dropout(x_img)
        x_img = self.x_img_l0_relu(x_img)
        x_img = self.x_img_l0_flatten(x_img)
        
        # TODO: 
        # finish processing x_img
        # process x_text
        # concat x_tab, x_img, x_text and send thru additional FC layers
        return x_tab
    
