'''
Defines DatafinitiDownloader class, which downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import requests
import urllib
import json
import datetime
import os
import sys; sys.path.insert(0, '..')

from pathlib import Path
from typing import Union, Tuple, Optional
from credentials import DATAFINITI_API_TOKEN
from logging_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class DatafinitiDownloader:
    '''
    Download data from Datafiniti. 
    public methods:
        - download_data_and_upload_to_s3
    '''
    
    data_path = Path('../data')

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
        AND dateAdded:[2017-01-01 TO *]\
    '''
    
    def __init__(self, num_records:int, query_today_updates_only:bool=False):
        self.num_records = num_records
        self.query_today_updates_only = query_today_updates_only
        
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
        if not self.query_today_updates_only:
            query = self.sold_homes_query
        elif self.query_today_updates_only:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            query = self.sold_homes_query + f'\nAND dateUpdated:{today}'
        return query
    
    def download_data_and_upload_to_s3(self) -> None:
        post_resp_json, download_id = self._send_post_req()
        get_resp_json, results = self._send_get_req(download_id)
        self._download_all_results_and_upload_to_s3(results, download_id)
    
    def _send_post_req(self) -> Tuple[Union[list, dict], str]:
        post_resp_obj = requests.post(
            'https://api.datafiniti.co/v4/properties/search',
            json=self.request_data, headers=self.request_headers
        )
        logger.info(f'Post response: {post_resp_obj}')
        breakpoint()
        post_resp_json = post_resp_obj.json()
        download_id = post_resp_json['id']
        logger.info(f'Download ID: {download_id}')
        return post_resp_json, download_id
    
    def _send_get_req(self, download_id:int) -> Union[list, dict]:
        get_resp_obj = requests.get(
            f'https://api.datafiniti.co/v4/downloads/{download_id}',
            headers=self.request_headers
        )
        get_resp_json = get_resp_obj.json()
        results = get_resp_json['results']
        return get_resp_json, results
    
    def _download_all_results_and_upload_to_s3(
        self, results:list, download_id:int
    ) -> None:
        results_flattened = [i for sublist in results for i in sublist]
        for i, res in enumerate(results_flattened):
            self._download_results_locally(i, res, download_id)
            # TODO: parse JSON
            # TODO: upload results to s3
            
    def _download_results_locally(
        self, indexer:int, result:str, download_id:int
    ) -> None:
        result_filepath = self.data_path/f'{download_id}_{indexer}_recs.json'
        urllib.request.urlretrieve(result, result_filepath)
        logger.info(f'result written to: {result_path}')
            
        
        