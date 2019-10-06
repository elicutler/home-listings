'''
Downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import argparse
import sys; sys.path.insert(0, '..')
import sagemaker

from gen_utils import set_logger_defaults, delete_file_types
from datafiniti_downloader import DatafinitiDownloader

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_records', '-n', type=int, default=1, 
        help='number of records to download from Datafiniti'
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
    
    datafiniti_downloader = DatafinitiDownloader(
        num_records=args_dict['num_records'],
        query_today_updates_only=args_dict['query_today_updates_only'],
        get_timeout_secs=args_dict['get_timeout_secs']
    )
    csv_samples = datafiniti_downloader.download_results_as_local_csv()
    session.upload_data(csv_samples, key_prefix=s3_prefix)

    delete_file_types(data_path, '.json')
    delete_file_types(data_path, '.csv')
    logger.info('Finished.')