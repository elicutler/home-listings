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
    if sample['features'][i]['key'] == 'Property History':
        if 'value' in sample['features'][i].keys():
            for j in range(len(sample['features'][i]['value'])):
                print(sample['features'][i]['value'][j])
                # TODO: get earliest
                    
# earliest description
earliest_desc_date = None
earliest_desc = None
if 'descriptions' in sample.keys():
    for i in range(len(sample['descriptions'])):
        if 'dateSeen' in sample['descriptions'][i].keys():
            date_seen_str = sample['descriptions'][i]['dateSeen']
            date_seen = datetime.datetime.strptime(date_seen_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            if earliest_desc_date is None or date_seen < earliest_desc_date:
                earliest_desc_date = date_seen
                earliest_desc = sample['descriptions'][i]['value']

            
# first image
first_image = sample['imageURLs'][0]

# other features

def clean_string(string):
    cleaned_string = re.sub(r'^\W+|\W+$', '', string)
    cleaned_string = re.sub(r'(?<!^)\W+|\W+(?=$)', '_', cleaned_string)
    cleaned_string = cleaned_string.lower()
    return cleaned_string

def extract_price(string):
    price_str = re.findall(r'(?<=\$)(\d+,\d+)', string)[0]
    price = int(price_str.replace(',', ''))
    return price

features = {}
features['latitude'] = float(sample['latitude'])
features['longitude'] = float(sample['longitude'])
features['floor_size'] = sample['floorSizeValue'] if sample['floorSizeUnit'] == 'sq. ft.' else None
features['lot_size'] = sample['lotSizeValue'] if sample['lotSizeUnit'] == 'sq. ft.' else None
for i in range(len(sample['features'])):
    sample_key = sample['features'][i]['key']
    sample_value = sample['features'][i]['value']
    if clean_string(sample_key) in ('built', 'year_built'):
        features['year_built'] = int(sample_value[0])
    if clean_string(sample_key) == 'exterior':
        features['exterior'] = int(sample_value[0])
    if clean_string(sample_key) == 'baths':
        baths_list = sample_value[0].split(',')
        baths_list = [clean_string(i) for i in baths_list]
        for i in range(len(baths_list)):
            if 'full' in baths_list[i]:
                features['baths_full'] = int(re.findall(r'\d+', baths_list[i])[0])
            if 'partial' in baths_list[i]:
                features['baths_partial'] =  int(re.findall(r'\d+', baths_list[i])[0])
            else:
                features['baths_full'] = int(re.findall(r'\d+', baths_list[i])[0])
                features['baths_partial'] = 0
    if clean_string(sample_key) == 'heating_fuel':
        features['heating_fuel'] = clean_string(sample_value[0])
    if clean_string(sample_key) == 'rooms':
        features['num_rooms'] = int(sample_value[0])
    elif clean_string(sample_key) == 'room_information':
        features['num_rooms'] = len(sample_value[1].split(','))
    if clean_string(sample_key) == 'cooling_system':
        features['cooling_system'] = clean_string(sample_value[0])
    if clean_string(sample_key) == 'taxable_value':
        for i in range(len(sample_value)):
            if 'Land' in sample_value[i]:
                features['land_value'] = extract_price(sample_value[i])
            elif 'Additions' in sample_value[i]:
                features['additions_value'] = extract_price(sample_value[i])
    if clean_string(sample_key) == 'building_information':
        for v in range(len(sample_values)):
            if 'foundation_details' in clean_string(sample_values[v]):
                features['foundation_details'] = clean_string(sample_values[v].split(':')[1])
    if clean_string(sample_key) == 'roof':
        features['roof_material'] = clean_string(sample_value[0])
    if clean_string(sample_key) == 'style':
        features['style'] == clean_string(sample_value[0])
    if clean_string(sample_key) == 'parking_spaces':
        features['parking_spaces'] = int(sample_value[0])
    else:
        features['parking_spaces'] = 0
    if clean_string(sample_key) == 'heating':
        features['heating_type'] = sample_value[0]
    if 'postal_code' in clean_string(sample_key):
        zip_homes_info = [clean_string(i) for i in sample_values]
        for i in range(len(zip_homes_info)):
            if 'median_list_price' in zip_homes_info[i]:
                median_price_str = re.findall(r'(?<=\$)(\d+,\d+)', sample_value[i])[0]
                features['median_price'] = int(median_price_str.replace(',', ''))
            if 'median_sale_list' in zip_homes_info[i]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    

