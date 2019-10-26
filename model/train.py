
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
    
    parser.add_argument(
        '--output_data_dir', '-o', type=str, 
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    parser.add_argument(
        '--model_dir', '-m', type=str, 
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--train_dir', '-t', type=str, 
        default=os.environ['SM_CHANNEL_TRAIN_DIR']
    )
    parser.add_argument(
        '--val_dir', '-v', type=str, 
        default=os.environ['SM_CHANNEL_VAL_DIR']
    )
    
    parser.add_argument('--outcome', '-O', type=str, default='first_sold_price')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    
    args = parser.parse_args()
    
    logger.info(f'command-line args: {args}')
    
    trainer = Trainer()
    trainer.make_data_loader(
        which_loader='train', path=args.train_dir, batch_size=args.batch_size, 
        outcome=args.outcome, concat_all=True, data_file=None
    )
    x_tab_input_dim, x_text_input_dim, x_img_input_dim = trainer.get_input_dims()
    
    model = PyTorchModel(x_tab_input_dim, x_text_input_dim, x_img_input_dim)
    
    trainer.make_data_loader(
        which_loader='val', path=args.val_dir, batch_size=args.batch_size,
        outcome=args.outcome, concat_all=True, data_file=None
    )
    
    model_params_path = args.model_dir + '/model_params.pt'
            
    trainer.set_model(
        model=model, loss_func_cls=torch.nn.L1Loss, 
        optimizer_cls=torch.optim.Adam, lr=1e-5
    )
    trainer.train(epochs=args.epochs)
       
    with open(model_params_path, 'wb') as file:
        torch.save(trainer.model.cpu().state_dict(), file)
        logger.info(f'Model state_dict saved to {model_params_path}')

