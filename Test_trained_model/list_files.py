
from os import listdir
from os.path import isfile, join
import os

datapath = './test_data'

tifffiles = []
for f in listdir(datapath):
	if isfile(join(datapath, f)) and (f.endswith('.tiff') or f.endswith('.tif')):
		tifffiles.append(join(datapath, f))


print(os.path.splitext(tifffiles[0])[0])