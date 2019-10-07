
import sys; sys.path.insert(0, '../utils')
import logger
import torch
import argparse

from typing import Union
from pathlib import Path
from gen_utils import set_logger_defaults
from models import PyTorchModel

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


def model_fn(model_dir:Union[Path, str]) -> PyTorchModel:
    '''
    Load the PyTorchModel from model_dir
    '''
    model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)
    model = PyTorchModel()
    
    with open(model_dir/'pytorch_model.pt', 'rb') as model_file:
        model.load_state_dict(torch.load(model_file))
        
    return model
    

# https://sagemaker.readthedocs.io/en/stable/using_pytorch.html
# def model_fn(model_dir):
#     model = Your_Model()
#     with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f))
#     return model


# def model_fn(model_dir):
#     """Load the PyTorch model from the `model_dir` directory."""
#     print("Loading model.")

#     # First, load the parameters used to create the model.
#     model_info = {}
#     model_info_path = os.path.join(model_dir, 'model_info.pth')
#     with open(model_info_path, 'rb') as f:
#         model_info = torch.load(f)

#     print("model_info: {}".format(model_info))

#     # Determine the device and construct the model.
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

#     # Load the stored model parameters.
#     model_path = os.path.join(model_dir, 'model.pth')
#     with open(model_path, 'rb') as f:
#         model.load_state_dict(torch.load(f))

#     # set to eval mode, could use no_grad
#     model.to(device).eval()

#     print("Done loading model.")
# return model