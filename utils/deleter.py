
import logging

from typing import Union
from pathlib import Path
from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class Deleter:
    '''
    Delete different types of things from a directory
    '''
    def __init__(self, path:Union[str, Path]):
        self.path = path

    def delete_json_files(self) -> None:
        json_files = [f for f in os.listdir(self.path) if f.endswith('json')]
        for f in json_files:
            os.remove(f'{self.path}/{f}')

    def delete_csv_files(self) -> None:
        csv_files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        for f in csv_files:
            os.remove(f'{self.path}/{f}')
