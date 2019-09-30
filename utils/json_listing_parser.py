
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
    
    def __init__(
        self, json_file:Union[str, Path], all_attributes:list=[
            'first_sold_date', 'first_sold_price', 'first_listed_date',
            'first_listed_price', 'first_desc_date', 'first_desc',
            'first_jpg_image_link', 'latitude', 'longitude', 'floor_size',
            'lot_size', 'year_built', 'exterior', 'heating_fuel', 'num_rooms',
            'cooling_system', 'land_value', 'additions_value', 'foundation_details',
            'roof_material', 'style', 'parking_spaces', 'heating_type',
            'median_list_price', 'median_sale_list_price_ratio'
        ]
    ):
        self.json_file = json_file
        self.all_attributes = all_attributes
        self.attributes = {a: None for a in self.all_attributes}
        
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
        features:Union[list, tuple, str], mode:str='check_key_pass_val_only'
    ) -> object:
        features_grp = features if type(features) in [list, tuple] else [features]
        def decorator(method):
            @wraps(method)
            def wrap_method(self):
                for i in range(len(self.json_listing['features'])):
                    feat_key = sample['features'][i]['key']
                    feat_val = sample['features'][i]['value']
                    
                    if mode == 'check_key_pass_val_only':
                        if self.clean_string(feat_key) in features_grp:
                            method(self, feat_val)
                            break
                    elif mode == 'check_key_pass_key_and_val':
                        if self.clean_string(feat_key) in features_grp:
                            method(self, feat_val, feat_key)
                            break
            return wrap_method 
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
        
    @_loop_over_features_list(
        ['rooms', 'room_information'], mode='check_key_pass_key_and_val'
    )
    def get_num_rooms(feat_val:str, feat_key:str) -> None:
        if self.clean_string(feat_key) == 'rooms':
            num_rooms = self.clean_string(feat_val[0])
        elif self.clean_string(feat_key) == 'room_information':
            num_rooms = len(sample_value[1].split(','))
        self.attributes['num_rooms'] = num_rooms
        
    @_loop_over_features_list('cooling_system')
    def get_cooling_system(feat_val:str) -> None:
        self.attributes['cooling_system'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list('taxable_value')
    def get_land_value(feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'land' in clean_string(feat_val[i]):
                self.attributes['land_value'] = self.extract_price(feat_val[i])
                
    @_loop_over_features_list('taxable_value')
    def get_additions_value(feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'additions' in clean_string(feat_val[i]):
                self.attributes['additions_value'] = self.extract_price(feat_val[i])
                
    @_loop_over_features_list('building_information')
    def get_foundation_details(feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'foundation_details' in self.clean_string(feat_val[i]):
                self.attributes['foundation_details'] = (
                    self.clean_string(feat_val[i].split(':')[1])
                )
        
    @_loop_over_features_list('roof')
    def get_roof_material(feat_val:str) -> None:
        self.attributes['roof_material'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list('style')
    def get_style(feat_val:str) -> None:
        self.attributes['style'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list('parking_spaces')
    def get_parking_spaces(feat_val:str) -> None:
        self.attributes['parking_spaces'] = int(feat_val[0])
        
    @_loop_over_features_list('heating_type')
    def get_heating_type(feat_val:str) -> None:
        self.attributes['heating_type'] = self.clean_string(feat_val[0])
        
    @_loop_over_features_list('real_estate_sales')
    def get_median_list_price(feat_val:str) -> None:
        zip_homes_info = [self.clean_string(i) for i in feat_val]
        for j in range(len(zip_homes_info)):
            if 'median_list_price' in zip_homes_info[i]:
                self.attributes['median_list_price'] = self.extract_price(feat_val)
                
    @_loop_over_features_list('real_estate_sales')
    def get_median_sale_list_price_ratio(feat_val:str) -> None:
        zip_homes_info = [self.clean_string(i) for i in feat_val]
        for j in range(len(zip_homes_info)):
            if 'median_sale_list' in zip_homes_info[i]:
                median_sale_list_price_ratio_str = self.extract_price(feat_val)
                median_sale_list_price_ratio = (
                    float(median_sale_list_price_ratio_str.replace('%', '')) 
                    if '%' in median_sale_list_price_ratio_str
                    else float(median_sale_list_price_ratio_str)
                )
                self.attributes['median_sale_list_price_ratio'] = median_sale_list_price_ratio
        
        
parser = JsonListingParser('../data/0_0.json')