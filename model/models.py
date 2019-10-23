
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
        
        img_c, img_h, img_w = x_img_input_dim[2], x_img_input_dim[0], x_img_input_dim[1]
        
        print(f'IMG_C:{img_c}, IMG_H:{img_h}, IMG_W:{img_w}')
        
        self.x_img_l0_conv2d = nn.Conv2d(3, 6, 5)
        self.x_img_l0_dropout = nn.Dropout(p=0.5)
        self.x_img_l0_relu = nn.ReLU()
        self.x_img_l1_fc = nn.Linear(self._get_nodes('l0'), 1)
        
    def forward(
        self, x_tab:torch.Tensor, x_text:torch.Tensor, x_img:torch.Tensor
    ) -> torch.Tensor:
                    
        print(f' X_TAB SIZE: {x_tab.size()}')
        x_tab = self.x_tab_l0_fc(x_tab)
        x_tab = self.x_tab_l0_dropout(x_tab)
        x_tab = self.x_tab_l0_relu(x_tab)
        x_tab = self.x_tab_l1_fc(x_tab)
        
        # for img, move n_channels dim to front
        print(f'X_IMG ORIG SIZE: {x_img.size()}')
        x_img = x_img.view(-1, x_img.size()[3], x_img.size()[1], x_img.size()[2])
        print(f'X_IMG SIZE AFTER TRANSPOSE: {x_img.size()}')
        x_img = self.x_img_l0_conv2d(x_img)
        x_img = self.x_img_l0_dropout(x_img)
        x_img = self.x_img_l0_relu(x_img)
        x_img = self.x_img_l0_flatten(x_img)
        
        print(f'X_IMG OUT SIZE: {x_img.size()}')
        print(f'FLATTEN SIZE: {x_img.flatten().size()}')
        
        return x_tab


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()