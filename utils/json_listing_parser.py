
import logging
import json
import re
import urllib.request
import ast
import numpy as np

from typing import Union
from pathlib import Path
from datetime import datetime
from PIL import Image
from functools import wraps
from gen_utils import set_logger_defaults

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
            self.json_listing = json.load(file)
    
    @staticmethod
    def _clean_string(string:str) -> str:
        cleaned_string = re.sub(r'^\W+|\W+$', '', string)
        cleaned_string = re.sub(r'(?<!^)\W+|\W+(?=$)', '_', cleaned_string)
        cleaned_string = cleaned_string.lower()
        return cleaned_string

    @staticmethod
    def _extract_price(string:str) -> int:
        price_str = re.findall(r'(?<=\$)(\d+,\d+)', string)[0]
        price = int(price_str.replace(',', ''))
        return price
    
    def set_id(self) -> None:
        self.attributes['id'] = self.json_listing['id']
            
    def set_listing_history(self) -> None:
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
                    price = self._extract_price(rec)
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
                            
        listing_to_sale_days = int((first_sold_date - first_listed_date).days)
                            
        self.attributes['first_sold_date'] = first_sold_date
        self.attributes['first_sold_price'] = first_sold_price
        self.attributes['first_listed_date'] = first_listed_date
        self.attributes['first_listed_price'] = first_listed_price
        self.attributes['listing_to_sale_days'] = listing_to_sale_days
        
    def set_first_description(self) -> None:
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
                    
    def set_first_jpg_image(self) -> None:
        jpg_image_links = [
            i for i in self.json_listing['imageURLs'] if re.search('.jpg$', i)
        ]
        img_link = jpg_image_links[0]
        
        urllib.request.urlretrieve(img_link, self.data_path/f'{first_image_link}')
        img = Image.open(self.data_path/f'{img_link}')
        img_arr = np.asarray(img)
        img_arr_list = img_arr.tolist()
        
        self.attributes['first_img_link'] = img_link
        self.attributes['first_img_arr_list'] = img_arr_list

    def set_latitude(self) -> None:
        self.attributes['latitude'] = float(self.json_listing['latitude'])
        
    def set_longitude(self) -> None:
        self.attributes['longitude'] = float(self.json_listing['longitude'])
        
    def set_floor_size(self) -> None:
        if re.search(r'sq.*(ft|feet)', self.json_listing['floorSizeUnit'], re.I):
            floor_size = float(self.json_listing['floorSizeValue'])
        self.attributes['floor_size'] = floor_size
            
    def set_lot_size(self) -> None:
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
                    feat_key = self.json_listing['features'][i]['key']
                    feat_val = self.json_listing['features'][i]['value']
                    
                    if mode == 'check_key_pass_val_only':
                        if self._clean_string(feat_key) in features_grp:
                            method(self, feat_val)
                            break
                    elif mode == 'check_key_pass_key_and_val':
                        if self._clean_string(feat_key) in features_grp:
                            method(self, feat_val, feat_key)
                            break
            return wrap_method 
        return decorator
    
    @_loop_over_features_list(['built', 'year_built'])
    def set_year_built(self, feat_val:str) -> None:
        self.attributes['year_built'] = int(feat_val[0])
        
    @_loop_over_features_list('exterior')
    def set_exterior(self, feat_val:str) -> None:
        self.attributes['exterior'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list('heating_fuel')
    def set_heating_fuel(self, feat_val:str) -> None:
        self.attributes['heating_fuel'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list(
        ['rooms', 'room_information'], mode='check_key_pass_key_and_val'
    )
    def set_num_rooms(self, feat_val:str, feat_key:str) -> None:
        if self._clean_string(feat_key) == 'rooms':
            num_rooms = self._clean_string(feat_val[0])
        elif self._clean_string(feat_key) == 'room_information':
            num_rooms = len(feat_val[1].split(','))
        self.attributes['num_rooms'] = num_rooms
        
    @_loop_over_features_list('cooling_system')
    def set_cooling_system(self, feat_val:str) -> None:
        self.attributes['cooling_system'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list('taxable_value')
    def set_land_value(self, feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'land' in self._clean_string(feat_val[i]):
                self.attributes['land_value'] = self._extract_price(feat_val[i])
                
    @_loop_over_features_list('taxable_value')
    def set_additions_value(self, feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'additions' in self._clean_string(feat_val[i]):
                self.attributes['additions_value'] = self._extract_price(feat_val[i])
                
    @_loop_over_features_list('building_information')
    def set_foundation_details(self, feat_val:str) -> None:
        for i in range(len(feat_val)):
            if 'foundation_details' in self._clean_string(feat_val[i]):
                self.attributes['foundation_details'] = (
                    self._clean_string(feat_val[i].split(':')[1])
                )
        
    @_loop_over_features_list('roof')
    def set_roof_material(self, feat_val:str) -> None:
        self.attributes['roof_material'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list('style')
    def set_style(self, feat_val:str) -> None:
        self.attributes['style'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list('parking_spaces')
    def set_parking_spaces(self, feat_val:str) -> None:
        self.attributes['parking_spaces'] = int(feat_val[0])
        
    @_loop_over_features_list('heating_type')
    def set_heating_type(self, feat_val:str) -> None:
        self.attributes['heating_type'] = self._clean_string(feat_val[0])
        
    @_loop_over_features_list('real_estate_sales')
    def set_median_list_price(self, feat_val:str) -> None:
        zip_homes_info = [self._clean_string(i) for i in feat_val]
        for j in range(len(zip_homes_info)):
            if 'median_list_price' in zip_homes_info[i]:
                self.attributes['median_list_price'] = self._extract_price(feat_val)
                
    @_loop_over_features_list('real_estate_sales')
    def set_median_sale_list_price_ratio(self, feat_val:str) -> None:
        zip_homes_info = [self._clean_string(i) for i in feat_val]
        for j in range(len(zip_homes_info)):
            if 'median_sale_list' in zip_homes_info[i]:
                median_sale_list_price_ratio_str = self._extract_price(feat_val)
                median_sale_list_price_ratio = (
                    float(median_sale_list_price_ratio_str.replace('%', '')) 
                    if '%' in median_sale_list_price_ratio_str
                    else float(median_sale_list_price_ratio_str)
                )
                self.attributes['median_sale_list_price_ratio'] = median_sale_list_price_ratio
        
    def set_all_attributes(self, raise_exception:bool=False) -> None:
        try:
            self.set_id()
        except:
            logger.warning('Could not execute set_id()')
        try:
            self.set_listing_history()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_listing_history()')
        try:
            self.set_first_description()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_first_description()')
        try:
            self.set_first_jpg_image()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_first_jpg_image()')
        try:
            self.set_latitude()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_latitude()')
        try:
            self.set_longitude()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_longitude()')
        try:
            self.set_floor_size()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_floor_size()')
        try:
            self.set_lot_size()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_lot_size()')
        try:
            self.set_year_built()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_year_built()')
        try:
            self.set_exterior()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_exterior()')
        try:
            self.set_heating_fuel()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_heating_fuel()')
        try:
            self.set_num_rooms()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_num_rooms()')
        try:
            self.set_cooling_system()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_cooling_system()')
        try:
            self.set_land_value()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_land_value()')
        try:
            self.set_additions_value()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_additions_value()')
        try:
            self.set_foundation_details()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_foundation_details()')
        try:
            self.set_roof_material()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_roof_material()')
        try:
            self.set_style()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_style()')
        try:
            self.set_parking_spaces()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_parking_spaces()')
        try:
            self.set_heating_type()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_heating_type()')
        try:
            self.set_median_list_price()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_median_list_price()')
        try:
            self.set_median_sale_list_price_ratio()
        except:
            if raise_exception:
                raise
            else:
                logger.warning('Could not execute set_median_sale_list_price_ratio()')

