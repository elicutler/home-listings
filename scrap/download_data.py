
import requests
import urllib
import json
import os
import sys; sys.path.insert(0, '..')

from pathlib import Path
from pprint import pprint

from credentials import DATAFINITI_API_TOKEN

query = '''\
    statuses.type:\"For Sale\" \
    AND statuses.type:Sold \
    AND prices.amountMin:* \
    AND prices.amountMax:* \
'''
# query = 'features.value:\"Days on Market\"'
num_records = 1
download = True
data_path = Path('../data')

request_headers = {
    'Authorization': 'Bearer ' + DATAFINITI_API_TOKEN,
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

result_list = download_resp['results']
if len(result_list) == 1:
    result = result_list[0]
    all_recs_file = data_path/(str(download_id) + '_' + 'all_recs.json')
    urllib.request.urlretrieve(result, all_recs_file)
else:
    print(f'Warning: len(result_list) == {len(result_list)}')
    
with open(all_recs_file, 'r') as all_recs:
    for i, line in enumerate(all_recs):
        with open(data_path/(str(download_id) + '_rec_' + str(i) + '.json'), 'w') as rec:
            rec.write(line)
            
os.remove(all_recs_file)

rec_dict = {}
rec_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
for rec_name in rec_files:
    with open(data_path/rec_name, 'r') as rec_file:
        rec = {rec_name: json.load(rec_file)}
        rec_dict.update(rec)
        
# with open(data_path/'11460_rec_0.json', 'r') as sample_file:
with open(data_path/'11621_rec_0.json', 'r') as sample_file:
    sample = json.load(sample_file)

# is_sale_flattened = []
# for r in rec_dict.values():
#     if 'prices' in r.keys():
#         for i in range(len(r['prices'])):
#             if 'isSale' in r['prices'][i].keys():
#                 is_sale_flattened.append(r['prices'][i]['isSale'])