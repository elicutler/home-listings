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
import datetime

with open('../data/0_0.json', 'r') as file:
    sample = json.load(file)
    
    
# property history (list), which should contain one element with 
# first listed date and first listed price, and another element with
# closing date and closing price
for i in range(len(sample['features'])):
    if 'key' in sample['features'][i].keys():
        if sample['features'][i]['key'] == 'Property History':
            if 'value' in sample['features'][i].keys():
                for j in range(len(sample['features'][i]['value'])):
                    print(sample['features'][i]['value'][j])
                    # TODO: get earliest
                    
# earliest description
if 'descriptions' in sample.keys():
    earliest_desc_date = None
    earliest_desc = None
    for i in range(len(sample['descriptions'])):
        date_seen_str = sample['descriptions'][i]['dateSeen']
        date_seen = datetime.datetime.strptime(date_seen_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        if earliest_desc_date is None or date_seen < earliest_desc_date:
            earliest_desc_date = date_seen
            earliest_desc = sample['descriptions'][i]['value']
            
# first image
sample['imageURLs'][0]

# other features
sample['latitude']
sample['longitude']
sample['floorSizeValue'] if sample['floorSizeUnit'] == 'sq. ft.' else None
sample['lotSizeValue'] if sample['lotSizeUnit'] == 'sq. ft.' else None

                    
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    

