
import os
import logging
import torch
import argparse

from typing import Union
from pathlib import Path

from gen_utils import set_logger_defaults
from models import PyTorchModel
from trainer import Trainer

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_data_dir', '-o', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', '-m', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_data_dir', '-t', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val_data_dir', '-v', type=str, default=os.environ['SM_CHANNEL_VAL'])
    
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    
    args = vars(parser.parse_args())
    
    trainer = Trainer()
    trainer.make_data_loader(
        which_loader='train', path=args['train_data_dir'], batch_size=args['batch_size'], 
        outcome='first_sold_price', concat_all=True, data_file=None
    )
    x_tab_input_dim, x_text_input_dim, x_img_input_dim = (
        trainer.get_input_dims(which_loader='train')
    )
    model = PyTorchModel(x_tab_input_dim)
    
    model_params_path = args['model_dir'] + '/model_params.pt'
            
    trainer.set_model(
        model=model, loss_func_cls=torch.nn.L1Loss, 
        optimizer_cls=torch.optim.Adam, lr=1e-5
    )
    trainer.train(epochs=args['epochs'])
       
    with open(model_params_path, 'wb') as file:
        torch.save(trainer.model.cpu().state_dict(), file)
        logger.info(f'Model state_dict saved to {model_params_path}')


# def model_fn(model_dir:Union[Path, str]) -> PyTorchModel:
#     '''
#     Load the PyTorchModel from model_dir
#     '''
#     model_dir = model_dir if isinstance(model_dir, Path) else Path(model_dir)
#     model = PyTorchModel()
    
#     with open(model_dir/'pytorch_model.pt', 'rb') as model_file:
#         model.load_state_dict(torch.load(model_file))
        
#     return model
    

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