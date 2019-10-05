
import urllib
from PIL import Image

img_url = 'http://site.meishij.net/r/58/25/3568808/a3568808_142682562777944.jpg'
img_local = 'img.jpg'

urllib.request.urlretrieve(img_url, img_local)

img= Image.open(img_local)
img.show() # may not work if utility not installed