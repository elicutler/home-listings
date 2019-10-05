'''
Downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import argparse
import os
import sys; sys.path.insert(0, '..')
import sagemaker

from pathlib import Path
from gen_utils import set_logger_defaults
from datafiniti_downloader import DatafinitiDownloader
from deleter import Deleter

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_records', '-n', type=int, default=1, 
        help='number of records to download from Datafiniti'
    )
    parser.add_argument(
        '--chunksize', '-c', type=int, default=None,
        help='if not None, download data in chunks of the given size'
    )
    parser.add_argument(
        '--query_today_updates_only', '-q', action='store_true',
        help='only query listings updated today'
    )
    parser.add_argument(
        '--get_timeout_secs', '-g', type=int, default=10,
        help='maximum number of seconds to allow download attempt before timing out'
    )
    args = parser.parse_args()
    args_dict = vars(args)
    
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    s3_prefix = 'home-listings'
    data_path = '../data'
    
    num_records = args_dict['num_records']
    
    if args_dict['chunksize'] is None:
        chunksize = args_dict['num_records']
    else:
        chunksize = args_dict['chunksize']
        assert chunksize <= num_records, (
            'if chunksize specified, must be <= num_records'
        )
        
    full_chunks = num_records // chunksize
    last_chunksize = num_records % chunksize
    ttl_chunks = full_chunks + 1 if last_chunksize > 0 else full_chunks
    
    for c in range(ttl_chunks):
        if c+1 <= ttl_chunks:
            datafiniti_downloader = DatafinitiDownloader(
                num_records=chunksize, 
                query_today_updates_only=args_dict['query_today_updates_only'],
                get_timeout_secs=args_dict['get_timeout_secs']
            )
        elif last_chunksize > 0:
            datafiniti_downloader = DatafinitiDownloader(
                num_records=last_chunksize,
                query_today_updates_only=args_dict['query_today_updates_only'],
                get_timeout_secs=args_dict['get_timeout_secs']
            )
        csv_samples = datafiniti_downloader.download_results_as_local_csv()
        logger.info(f'Downloaded chunk {c+1}/{ttl_chunks} locally')
        
        session.upload_data(csv_samples, key_prefix=s3_prefix)
        logger.info(f'Uploaded chunk {c+1}/{ttl_chunks} to s3')
        
        deleter = Deleter(data_path)
        deleter.delete_json_files()
        deleter.delete_csv_files()

    logger.info('All chunks processed.')