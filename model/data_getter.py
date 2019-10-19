'''
Downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import argparse
import sys
import sagemaker

from gen_utils import set_logger_defaults, delete_file_types
from datafiniti_downloader import DatafinitiDownloader
from constants import S3_PREFIX

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
    parser.add_argument(
        '--s3_subfolder', '-s', type=str, default='train',
        help=(
            's3 subdirectory within home-listings to download data to.'
            ' Typically "train", "val", or "test".'
        )
    )
    
    mode = (
        'interactive' if len(sys.argv) > 1 and 'ipykernel' in sys.argv[0]
        and sys.argv[1] == '-f' else 'scripting'
    )
    if mode == 'interactive':
        args = parser.parse_args([ # set args here when running interactively
            '--s3_subfolder', 'val'            
        ]) 
    else:
        args = parser.parse_args()
    
    logger.info(f'{args}')
    
    logger.info('Initializing Sagemaker session...')
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    datafiniti_downloader = DatafinitiDownloader(
        num_records=args.num_records,
        query_today_updates_only=args.query_today_updates_only,
        get_timeout_secs=args.get_timeout_secs
    )

    try:
        logger.info('Downloading results to local machine...')
        csv_samples = datafiniti_downloader.download_results_as_local_csv()

        s3_dir = f'{S3_PREFIX}/{args.s3_subfolder}'
        logger.info(f'Uploading results to {s3_dir}...')
        session.upload_data(csv_samples, key_prefix=s3_dir)
    finally:
        logger.info(f'Removing local data files...')
        data_path = '../data'
        delete_file_types(data_path, '.json')
        delete_file_types(data_path, '.csv')
        delete_file_types(data_path, '.jpg')
    
    logger.info('Finished.')