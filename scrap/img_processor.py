
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

img = Image.open(img_local)

img_arr = np.asarray(img)
img_arr_list = img_arr.tolist()

df = pd.DataFrame({'x': [img_arr_list]})
df.to_csv('img_df.csv', header=False, index=False)

df_recon = pd.read_csv('img_df.csv', header=None)
df_recon.rename(columns={0:'x'}, inplace=True)

img_arr_recon = np.array(ast.literal_eval(df_recon['x'].values[0]), dtype='uint8')
img_recon = Image.fromarray(img_arr_recon)

img
img_recon