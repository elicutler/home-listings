
import json
import urllib
import numpy as np
import pandas as pd
import codecs
import io
import re
import ast
import sys

from PIL import Image

img_url = 'http://site.meishij.net/r/58/25/3568808/a3568808_142682562777944.jpg'
img_local = 'img.jpg'

urllib.request.urlretrieve(img_url, img_local)

img= Image.open(img_local)
img

# img_arr = np.asarray(img)
# img_arr_local_recon = Image.fromarray(img_arr)
# img_arr_local_recon # works 
# (img_arr == img_arr_local_recon).all() # True

# img_arr_list = img_arr.tolist()

# with codecs.open('json_img.json', 'w', encoding='utf-8') as outfile:
#     json.dump(img_arr_list, outfile, separators=(',', ':'), sort_keys=True, indent=4)

# with codecs.open('json_img.json', 'r', encoding='utf-8') as outfile:
#     img_str = outfile.read()
    
# img_list_recon = json.loads(img_str)
# img_arr_recon = np.array(img_list_recon)
# (img_arr_recon == img_arr).all() # True

# img_recon = Image.fromarray(img_arr_recon)
    
# img_arr_recon_2 = np.array(img_arr_recon)
# (img_arr_recon_2 == img_arr_recon).all() # True

# img_recon = Image.fromarray(img_arr_recon_2)

# #TODO: try pandas.to_csv

# img_arr = np.asarray(img)
# img_df_dict = {i: pd.DataFrame(img_arr[:, :, i]) for i in range(img_arr.shape[-1])}

# for i in img_df_dict.keys():
#     img_df_dict[i].to_csv(f'img_part_{i}.csv', header=False, index=False)
    
# recon_df_dict = {}
# for i in img_df_dict.keys():
#     with open(f'img_part_{i}.csv', 'r') as img_part_file:
#         recon_df_dict[i] = img_part_file.read()
        
# img_arr_recon = np.array([recon_df_dict[i] for i in recon_df_dict.keys()])

# # try Image tobytes, frombytes

# img_bytes = io.BytesIO(img.tobytes())
# img_from_bytes = Image.frombytes(
#     mode='RGB', size=np.asarray(img).shape[:-1], data=img_bytes_io
# )
# # img_bytes_io = io.BytesIO(img_bytes)
# # img_from_bytes = Image.frombytes(img_bytes)

# # try pandas again but differently
# img_arr = np.asarray(img)
# df2 = pd.DataFrame({'x': [img_arr]})
# df2.to_csv('img_df.csv', header=False, index=False)

# df2_recon = pd.read_csv('img_df.csv', header=None)
# df2_recon.rename(columns={0: 'x'}, inplace=True)
# img_arr_str = df2_recon['x'].values[0]
# img_arr_str = img_arr_str.replace('\n', '')
# img_arr_str = re.sub('(?<=\d) ', ', ', img_arr_str)
# img_arr_str = re.sub(r'(?<=\])(?!\]|$)', ',', img_arr_str)
# img_arr_recon = np.array(ast.literal_eval(img_arr_str))


# img_arr = eval(img_arr_str)

# # numpy string
# import re, ast
# img_arr = np.asarray(img)
# img_arr_str = str(img_arr)
# img_arr_str_cln = re.sub('\s+', ',', img_arr_str)
# img_arr_recon  = np.array(ast.literal_eval(img_arr_str_cln))


# # try lists again

img_arr = np.asarray(img)
img_arr_list = img_arr.tolist()
# img_arr_recon_from_list = np.array(img_arr_list)
# np.all(img_arr_list == img_arr_recon_from_list) # True

df = pd.DataFrame({'x': [img_arr_list]})
df.to_csv('img_df.csv', header=False, index=False)
df_recon = pd.read_csv('img_df.csv', header=None)
df_recon.rename(columns={0:'x'}, inplace=True)
img_arr_recon = np.array(ast.literal_eval(df_recon['x'].values[0]), dtype='uint8')
img_recon = Image.fromarray(img_arr_recon)
