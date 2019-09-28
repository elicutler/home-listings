'''
Downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import argparse
import sys; sys.path.insert(0, '..')

from pathlib import Path
from logging_utils import set_logger_defaults
from datafiniti_downloader import DatafinitiDownloader

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

datafiniti_downloader = DatafinitiDownloader(2)
datafiniti_downloader.upload_results_to_s3()

from pprint import pprint as pp
import json

with open('../data/0_0.json', 'r') as file:
    sample = json.load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    

