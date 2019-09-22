
import requests
import urllib
import json

from secrets import api_token

query = 'dateAdded:* AND propertyType:(Home)'
num_records = 50
download = True

request_headers = {
    'Authorization': 'Bearer ' + api_token,
    'Content-Type': 'application/json'
}
request_data = {
    'query': query,
    'format': 'JSON',
    'num_records': num_records,
    'download': download
}
post_req = requests.post(
   'https://api.datafiniti.co/v4/properties/search', 
    json=request_data, headers=request_headers
)
post_resp = post_req.json()
download_id = post_resp['id']
get_req = requests.get(
    'https://api.datafiniti.co/v4/downloads/' + str(download_id),
    headers=request_headers
)
download_resp = get_req.json() 
