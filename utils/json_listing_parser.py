
import json

from Typing import Union
from pathlib import Path
from datetime import datetime
from logging_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class JsonListingParser:
    
    def __init__(self, json_file):
        self.json_file = json_file
        self.listing_attributes = {}
        with open(json_file, 'r') as file:
            self.json_listing = json.load(json_file)
            

    #TODO
################################################################################################

# property history (list), which should contain one element with 
# first listed date and first listed price, and another element with
# closing date and closing price
earliest_listed_date = None
earliest_listed_price = None
earliest_sold_date = None
earliest_sold_price = None
for i in range(len(sample['features'])):
    if sample['features'][i]['key'] == 'Property History':
        if 'value' in sample['features'][i].keys():
            listings_with_price = [j for j in sample['features'][i]['value'] if '$' in j]
            for k in range(len(listings_with_price)):
                rec = listings_with_price[k]
                update_dt_str = re.findall(r'^.*?(?= \()', rec)[0]
                update_dt = datetime.strptime(update_dt_str, '%a %b %d %Y %H:%M:%S GMT%z')
                price = extract_price(rec)
                if 'Sold' in rec:
                    if earliest_sold_date is None or update_dt < earliest_sold_date:
                        earliest_sold_date = update_dt
                        earliest_sold_price = price
                elif 'Listed' in rec:
                    if earliest_listed_date is None or update_dt < earliest_listed_date:
                        earliest_listed_date = update_dt
                        earliest_listed_price = price
                
                print(datetime_str)
                # TODO: get earliest


# earliest description
earliest_desc_date = None
earliest_desc = None
if 'descriptions' in sample.keys():
    for i in range(len(sample['descriptions'])):
        if 'dateSeen' in sample['descriptions'][i].keys():
            date_seen_str = sample['descriptions'][i]['dateSeen']
            date_seen = datetime.strptime(date_seen_str, '%Y-%m-%dT%H:%M:%S.%fZ')
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
features['property_type'] = sample['propertyType']
for i in range(len(sample['features'])):
    sample_key = sample['features'][i]['key']
    sample_value = sample['features'][i]['value']
    if clean_string(sample_key) in ('built', 'year_built'):
        features['year_built'] = int(sample_value[0])
    if clean_string(sample_key) == 'exterior':
        features['exterior'] = clean_string(sample_value[0])
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
        for v in range(len(sample_value)):
            if 'foundation_details' in clean_string(sample_value[v]):
                features['foundation_details'] = clean_string(sample_value[v].split(':')[1])
    if clean_string(sample_key) == 'roof':
        features['roof_material'] = clean_string(sample_value[0])
    if clean_string(sample_key) == 'style':
        features['style'] = clean_string(sample_value[0])
    if clean_string(sample_key) == 'parking_spaces':
        features['parking_spaces'] = int(sample_value[0])
    if clean_string(sample_key) == 'heating':
        features['heating_type'] = clean_string(sample_value[0])
    if 'postal_code' in clean_string(sample_key):
        zip_homes_info = [clean_string(i) for i in sample_value]
        for i in range(len(zip_homes_info)):
            if 'median_list_price' in zip_homes_info[i]:
                features['median_price'] = extract_price(sample_value[i])
            if 'median_sale_list' in zip_homes_info[i]:
                median_sale_list_ratio_str = re.findall(r'\d+\.?\d+?%?', zip_homes_info[i])[0]
                median_sale_list_ratio = (
                    float(median_sale_list_ratio_str.replace('%', '')) 
                    if '%' in median_sale_list_ratio_str
                    else float(median_sale_list_ratio_str)
                )
                features['median_sale_list_ratio'] = median_sale_list_ratio