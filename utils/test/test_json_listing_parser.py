
import unittest
import logging
import sys; 
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from logging_utils import set_logger_defaults
from json_listing_parser import JsonListingParser

logger = logging.getLogger(__name__)
set_logger_defaults(logger)


class TestJsonListingParser(unittest.TestCase):
    
    json_listing_parser = JsonListingParser('../../data/test/0_0.json')

    def test_set_listing_history(self):
        pass