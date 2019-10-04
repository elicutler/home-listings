'''
Downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import argparse
import sys; sys.path.insert(0, '..')
import sagemaker

from pathlib import Path
from gen_utils import set_logger_defaults
from datafiniti_downloader import DatafinitiDownloader

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

datafiniti_downloader = DatafinitiDownloader(2)
datafiniti_downloader.download_results_as_local_csv()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    

