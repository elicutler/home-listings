'''
Downloads data in from Datafiniti via API call.
Requires DATAFINITI_API_TOKEN in ..credentials.py (not in repo).
Data formatted in JSON.
'''

import requests
import urllib
import json
import os
import sys; sys.path.insert(0, '..')

from pathlib import Path

from credentials import DATAFINITI_API_TOKEN