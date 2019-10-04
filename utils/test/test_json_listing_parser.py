
import unittest
import logging
import datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from gen_utils import set_logger_defaults
from json_listing_parser import JsonListingParser

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class TestJsonListingParser(unittest.TestCase):
    
    def setUp(self):
        self.json_listing_parser = JsonListingParser('../../data/test/0_0.json')

    def test_set_id(self):
        self.json_listing_parser.set_id()
        self.assertEqual(
            self.json_listing_parser.attributes['id'], 'AWuh3x3T0x_BgD4eD3hX'
        )
    def test_set_listing_history(self):
        self.json_listing_parser.set_listing_history()
        self.assertEqual(
            self.json_listing_parser.attributes['first_sold_date'],
            datetime.datetime(2019, 9, 6, 0, 0, tzinfo=datetime.timezone.utc)
        )
        self.assertEqual(
            self.json_listing_parser.attributes['first_sold_price'], 332500
        )
        self.assertEqual(
            self.json_listing_parser.attributes['first_listed_date'],
            datetime.datetime(2019, 6, 28, 0, 0, tzinfo=datetime.timezone.utc)
        )
        self.assertEqual(
            self.json_listing_parser.attributes['first_listed_price'], 340000
        )
        
    def test_set_first_description(self):
        self.json_listing_parser.set_first_description()
        self.assertIsInstance(self.json_listing_parser.attributes['first_desc'], str)
        self.assertGreater(len(self.json_listing_parser.attributes['first_desc']), 0)
    
    def test_set_first_jpg_image_link(self):
        self.json_listing_parser.set_first_jpg_image_link()
        self.assertTrue(
            self.json_listing_parser.attributes['first_jpg_image_link'].endswith('.jpg')
        )
        
    def test_set_latitude(self):
        self.json_listing_parser.set_latitude()
        self.assertAlmostEqual(
            self.json_listing_parser.attributes['latitude'], 39, delta=1
        )
        
    def test_set_longitude(self):
        self.json_listing_parser.set_longitude()
        self.assertAlmostEqual(
            self.json_listing_parser.attributes['longitude'], -75, delta=1
        )
        
    def test_set_floor_size(self):
        self.json_listing_parser.set_floor_size()
        self.assertEqual(self.json_listing_parser.attributes['floor_size'], 2850)
        
    def test_set_lot_size(self):
        self.json_listing_parser.set_lot_size()
        self.assertEqual(self.json_listing_parser.attributes['lot_size'], 8276)
        
    def test_set_year_built(self):
        self.json_listing_parser.set_year_built()
        self.assertEqual(self.json_listing_parser.attributes['year_built'], 2008)
        
    def test_set_exterior(self):
        self.json_listing_parser.set_exterior()
        self.assertEqual(self.json_listing_parser.attributes['exterior'], 'vinyl')
        
    def test_set_heating_fuel(self):
        self.json_listing_parser.set_heating_fuel()
        self.assertEqual(self.json_listing_parser.attributes['heating_fuel'], 'electric')
        
    def test_set_num_rooms(self):
        self.json_listing_parser.set_num_rooms()
        self.assertEqual(self.json_listing_parser.attributes['num_rooms'], 14)
        
    def test_set_cooling_system(self):
        self.json_listing_parser.set_cooling_system()
        self.assertEqual(self.json_listing_parser.attributes['cooling_system'], 'central')
        
    def test_set_land_value(self):
        self.json_listing_parser.set_land_value()
        self.assertEqual(self.json_listing_parser.attributes['land_value'], 13300)
        
    def test_set_additions_value(self):
        self.json_listing_parser.set_additions_value()
        self.assertEqual(self.json_listing_parser.attributes['additions_value'], 89600)
        
    def test_set_foundation_details(self):
        self.json_listing_parser.set_foundation_details()
        self.assertEqual(
            self.json_listing_parser.attributes['foundation_details'], 'concrete_perimeter'
        )
        
    def test_set_roof_material(self):
        self.json_listing_parser.set_roof_material()
        self.assertEqual(self.json_listing_parser.attributes['roof_material'], 'asphalt')
        
    def test_set_style(self):
        self.json_listing_parser.set_style()
        self.assertEqual(self.json_listing_parser.attributes['style'], 'colonial')
        
    def test_set_parking_spaces(self):
        self.json_listing_parser.set_parking_spaces()
        self.assertEqual(self.json_listing_parser.attributes['parking_spaces'], 6)
        
if __name__ == '__main__':
    unittest.main()