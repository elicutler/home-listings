'''
Defines DatafinitiDownloader class, which downloads data from Datafiniti via 
API call and uploads it to S3. Requires DATAFINITI_API_TOKEN in 
..credentials.py (not in repo). Data formatted in JSON.
'''

import logging
import requests
import urllib
import time
import os
import sys; sys.path.insert(0, '..')
import pandas as pd

from pathlib import Path
from typing import Union
from credentials import DATAFINITI_API_TOKEN
from gen_utils import (
    set_logger_defaults, get_unique_id, put_columns_in_order, filter_df_missing_col
)
from json_listing_parser import JsonListingParser

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class DatafinitiDownloader:
    '''
    Download data from Datafiniti as JSON to local
    machine and upload it to destinations.
    '''
    
    data_path = Path('../data')
    json_listing_prefix = 'listing'

    sold_homes_query = '''\
        statuses.type:\"For Sale\"\
        AND statuses.type:Sold\
        AND prices.amountMin:*\
        AND prices.amountMax:*\
        AND features:*\
        AND descriptions.value:*\
        AND features.key:\"Property History\"\
        AND propertyType:(Home OR \"Multi-Family Dwelling\" OR \
            \"Single Family Dwelling\" OR Townhouse)\
        AND sourceURLs:redfin.com\
        AND dateAdded:[2017-01-01 TO *]
    '''
    
    def __init__(
        self, num_records:int, query_today_updates_only:bool=False,
        get_timeout_secs:int=10
    ):
        self.num_records = num_records
        self.query_today_updates_only = query_today_updates_only
        self.get_timeout_secs = get_timeout_secs
        
        self.query = self._set_query()
        self.request_headers = {
            'Authorization': 'Bearer ' + DATAFINITI_API_TOKEN,
            'Content-Type': 'application/json'
        }
        self.request_data = {
            'query': self.query,
            'format': 'JSON',
            'num_records': self.num_records,
            'download': True
        }
        
    def _set_query(self) -> str:
        if self.query_today_updates_only:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            query = self.sold_homes_query + f'\nAND dateUpdated:{today}'         
        else:
            query = self.sold_homes_query
        return query
    
    def download_results_as_local_csv(self) -> str:
        all_listings_dict = {}
        results = self._summon_data()
        logger.info(f'Unlocked {len(results)} sets of results.')
        
        for res in results:
            result_filepath = self.data_path/'results_group.txt'
            urllib.request.urlretrieve(res, result_filepath)
            
            self._unpack_to_json_files(result_filepath)
            os.remove(result_filepath)
            
        json_listings = [
            f for f in os.listdir(self.data_path) 
            if f.startswith(self.json_listing_prefix) and f.endswith('.json')
        ]
        logger.info(f'Downloaded {len(json_listings)} records from Datafiniti')
        
        for i, f in enumerate(json_listings):
            logger.info(f'Parsing listing {i}/{len(json_listings)}')
            listing_dict = self._parse_json_listing(f)
            all_listings_dict.update({Path(f).stem: listing_dict}) 
        
        all_listings_frame = pd.DataFrame(all_listings_dict).transpose()
        listings_frame_ordered = put_columns_in_order(all_listings_frame)
        listings_frame_w_id = filter_df_missing_col(listings_frame_ordered, 'id')

        data_id = get_unique_id(str)
        csv_path = f'../data/listings_{data_id}.csv'
        listings_frame_w_id.to_csv(csv_path, header=False, index=False)
        return csv_path
                
    def _summon_data(self) -> list:
        download_id = self._send_post_req()
        results = self._send_get_req(download_id)
        return results
    
    def _send_post_req(self) -> int:
        logger.info('Sending POST request to Datafiniti...')
        post_resp_obj = requests.post(
            'https://api.datafiniti.co/v4/properties/search',
            json=self.request_data, headers=self.request_headers
        )
        post_resp_json = post_resp_obj.json()
        download_id = post_resp_json['id']
        return download_id
    
    def _send_get_req(self, download_id:str) -> list:
        status = None
        elapsed_time = 0
        start_time = time.time()
        
        while status != 'completed':
            logger.info('Sending GET request to Datafiniti...')
            get_resp_obj = requests.get(
                f'https://api.datafiniti.co/v4/downloads/{download_id}', 
                headers=self.request_headers
            )
            get_resp_json = get_resp_obj.json()
            status = get_resp_json['status']
            
            elapsed_time = time.time() - start_time
            if elapsed_time > self.get_timeout_secs:
                raise Exception(
                    'Could not retrieve the download url.'
                    f' Exceeded max get time: {self.get_timeout_secs:,} seconds.'
                )
            
        results = get_resp_json['results']
        return results
            
    def _unpack_to_json_files(self, result_filepath:Union[str, Path]) -> None:
        # result_filepath will contain a file where each line contains a JSON 
        # record, but as a whole the file is not valid JSON. The following
        # block writes each line in into its own - valid - JSON file.
        with open(result_filepath, 'r') as read_file:
            logger.info(f'Reading results set from {result_filepath.resolve()}')
            for line in read_file:
                file_id = get_unique_id(str)
                listing_filepath = self.data_path/f'{self.json_listing_prefix}_{file_id}.json'
                
                with open(listing_filepath, 'w') as write_file:
                    logger.info(f'Writing listing result to {listing_filepath.resolve()}')
                    write_file.write(line)
        
    def _parse_json_listing(self, filepath:Union[str, Path]) -> dict:
        json_listing_path = self.data_path/filepath
        json_listing_parser = JsonListingParser(json_listing_path)
        json_listing_parser.set_all_attributes()
        return json_listing_parser.attributes
                        