'''
Defines DatafinitiDownloader class, which downloads data from Datafiniti via API call and uploads it to S3.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import logging
import requests
import urllib
import json
import os
import sys; sys.path.insert(0, '..')
import pandas as pd

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
        AND dateAdded:[2017-01-01 TO *]\
    '''
    
    def __init__(
        self, num_records:int, query_today_updates_only:bool=False,
        max_get_attempts:int=100
    ):
        self.num_records = num_records
        self.query_today_updates_only = query_today_updates_only
        self.max_get_attempts = max_get_attempts
        
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
    
    def upload_results_to_s3(self) -> None:
        post_resp_json, download_id = self._send_post_req()
        get_resp_json, results = self._send_get_req(download_id)
        for i, r in enumerate(results):
            self._download_result_locally(i, r)
            listings = self._parse_json_listings()
    
    def _send_post_req(self) -> Tuple[Union[list, dict], int]:
        post_resp_obj = requests.post(
            'https://api.datafiniti.co/v4/properties/search',
            json=self.request_data, headers=self.request_headers
        )
        post_resp_json = post_resp_obj.json()
        download_id = post_resp_json['id']
        return post_resp_json, download_id
    
    def _send_get_req(self, download_id:str) -> Tuple[Union[list, dict], list]:
        status = None
        get_attempts = 0
        
        while status != 'completed' and get_attempts <= self.max_get_attempts:
            get_resp_obj = requests.get(
                f'https://api.datafiniti.co/v4/downloads/{download_id}', 
                headers=self.request_headers
            )
            get_resp_json = get_resp_obj.json()
            status = get_resp_json['status']
            get_attempts += 1
            
        assert status == 'completed', (
            'Could not retrieve the download url.'
            f' Exceeded max get attempts: {self.max_get_attempts}'
        )

        results = get_resp_json['results']
        return get_resp_json, results
            
    def _download_result_locally(self, i:int, result:str) -> None:
        result_filepath = self.data_path/'results_group'
        urllib.request.urlretrieve(result, result_filepath)
        
        # result_filepath will contain a file where each line contains a JSON 
        # record. The following block writes each line in result_filepath
        # into its own JSON file.
        with open(result_filepath, 'r') as read_file:
            for j, line in enumerate(read_file):
                listing_filepath = (
                    self.data_path/f'{self.json_listing_prefix}_{i}_{j}.json'
                )
                with open(listing_filepath, 'w') as write_file:
                    write_file.write(line)
        os.remove(result_filepath)
        
    def _parse_json_listings(self) -> pd.DataFrame:
        json_listings = [
            f for f in os.listdir(self.data_path) 
            if f.startswith(self.json_listing_prefix) and f.endswith('.json')
        ]
        for f in json_listings:
            json_listing_parser = JsonListingParser(f)
            # TODO

                        
class JsonListingParser:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(json_file, 'r') as file:
            self.listing = json.load(json_file)
            
    #TODO
        
    
                
