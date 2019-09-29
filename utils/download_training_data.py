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
from datetime import datetime

with open('../data/0_0.json', 'r') as file:
    sample = json.load(file)
    
class A:
    attributes = {}
    def loop_over_features(feature):
        actual_feature = feature
        def decorator(func):
            @wraps(func)
            def wrapper(self):
                for i in range(len(sample['features'])):
                    sample_key = sample['features'][i]['key']
                    sample_val = sample['features'][i]['value']
                    print(sample_key)
                    if sample_key == actual_feature:
                        func(self, sample_key, sample_val)
                        break
            return wrapper 
        return decorator

    @loop_over_features('Built')
    def get_year_built(self, k, v):
        year_built = int(v[0])
        self.attributes['year_built'] = year_built
        print(self.attributes)
a = A()
a.get_year_built()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    

