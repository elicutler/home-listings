
import logging
import json

from typing import Union
from pathlib import Path
from datetime import datetime
from functools import wraps
from logging_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class JsonListingParser:
    '''
    Parse Datafiniti JSON blob to extract dictionary of attributes.
    '''
    
    def __init__(self, json_file:Union[str, Path]):
        self.json_file = json_file
        self.attributes = {}
        with open(json_file, 'r') as file:
            self.json_listing = json.load(json_file)
    
    @staticmethod
    def clean_string(string:str) -> str:
        cleaned_string = re.sub(r'^\W+|\W+$', '', string)
        cleaned_string = re.sub(r'(?<!^)\W+|\W+(?=$)', '_', cleaned_string)
        cleaned_string = cleaned_string.lower()
        return cleaned_string

    @staticmethod
    def extract_price(string:str) -> int:
        price_str = re.findall(r'(?<=\$)(\d+,\d+)', string)[0]
        price = int(price_str.replace(',', ''))
        return price
            
    def get_listing_history(self) -> None:
        first_listed_date = None
        first_listed_price = None
        first_sold_date = None
        first_sold_price = None
        
        for i in range(len(self.json_listing['features'])):
            if self.json_listing['features'][i]['key'] == 'Property History':
                listings_with_price = [
                    j for j in self.json_listing['features'][i]['value'] 
                    if '$' in j
                ]
                for k in range(len(listings_with_price)):
                    rec = listings_with_price[k]
                    update_dt_str = re.findall(r'^.*?(?= \()', rec)[0]
                    update_dt = datetime.strptime(
                        update_dt_str, '%a %b %d %Y %H:%M:%S GMT%z'
                    )
                    price = extract_price(rec)
                    if 'Sold' in rec:
                        if first_sold_date is None \
                        or update_dt < first_sold_date:
                            first_sold_date = update_dt
                            first_sold_price = price
                    elif 'Listed' in rec:
                        if first_listed_date is None \
                        or update_dt < first_listed_date:
                            first_listed_date = update_dt
                            first_listed_price = price   
                            
        self.attributes['first_sold_date'] = first_sold_date
        self.attributes['first_sold_price'] = first_sold_price
        self.attributes['first_listed_date'] = first_listed_date
        self.attributes['first_listed_price'] = first_listed_price
        
    def get_first_description(self) -> None:
        first_desc_date = None
        first_desc = None
        descriptions = self.json_listing['descriptions']
        for i in range(len(descriptions)):
            if 'dateSeen' in descriptions[i].keys():
                date_seen_str = descriptions[i]['dateSeen']
                date_seen = datetime.strptime(date_seen_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                if first_desc_date is None or date_seen < first_desc_date:
                    first_desc_date = date_seen
                    first_desc = descriptions[i]['value']
                    
        self.attributes['first_desc_date'] = first_desc_date
        self.attributes['first_desc'] = first_desc
                    
    def get_first_jpg_image(self) -> None:
        jpg_image_links = [
            i for i in self.json_listing['imageURLs'] if re.search('.jpg$', i)
        ]
        self.attributes['first_jpg_image_link'] = jpg_image_links[0]
        
    def get_latitude(self) -> None:
        self.attributes['latitude'] = float(self.json_listing['latitude'])
        
    def get_longitude(self) -> None:
        self.attributes['longitude'] = float(self.json_listing['longitude'])
        
    def get_floor_size(self) -> None:
        if re.search(r'sq.*(ft|feet)', self.json_listing['floorSizeUnit'], re.I):
            floor_size = float(self.json_listing['floorSizeValue'])
        self.attributes['floor_size'] = floor_size
            
    def get_lot_size(self) -> None:
        if re.search(r'sq.*(ft|feet)', self.json_listing['lotSizeUnit'], re.I):
            lot_size = float(self.json_listing['lotSizeValue'])
        self.attributes['lot_size'] = lot_size
        
    def _loop_over_features_list(
        features:Union[list, tuple, str], mode:str='pass_val_only'
    ) -> object:
        features_grp = features if type(features) in [list, tuple] else [features]
        def decorator(method):
            @wraps(method)
            def wrap_method(self):
                for i in range(len(self.json_listing['features'])):
                    feat_key = sample['features'][i]['key']
                    feat_val = sample['features'][i]['value']
                    
                    if mode == 'pass_val_only"'
                        if self.clean_string(feat_key) in features_grp:
                            method(self, feat_val)
                            break
                    elif mode == 'pass_key_and_val':
                        if self.clean_string(feat_key) in features_grp:
                            method(self, feat_val, feat_key)
                            break
            return wrapper 
        return decorator
    
    @_loop_over_features_list(['built', 'year_built'])
    def get_year_built(feat_val:str) -> None:
        self.attributes['year_built'] = int(feat_val[0])
        
    @_loop_over_features_list('exterior')
    def get_exterior(feat_val:str) -> None:
        self.attributes['exterior'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list('heating_fuel')
    def get_heating_fuel(feat_val:str) -> None:
        self.attributes['heating_fuel'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list(['rooms', 'room_information'], mode='pass_key_and_val')
    def get_num_rooms(feat_val:str, feat_key:str) -> None:
        if self.clean_string(feat_key) == 'rooms':
            num_rooms = self.clean_string(feat_val[0])
        elif self.clean_string(feat_key) == 'room_information':
            num_rooms = len(sample_value[1].split(','))
        self.attributes['num_rooms'] = num_rooms
        

        
  

        
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