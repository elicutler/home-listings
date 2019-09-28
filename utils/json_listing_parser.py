
import json

from logging_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

class JsonListingParser:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(json_file, 'r') as file:
            self.json_listing = json.load(json_file)
            
    #TODO
        