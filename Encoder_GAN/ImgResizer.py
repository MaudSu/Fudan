directory = "Pictures"
new_dir = "Pictures_parsed"
PIXEL_SIZE = 64

#Create a folder called Pictures. This should contain the folders which further contain pictures of any type.
#The script will go through each folder through each picture and store pictures in a new folder called Pictures_parsed in a random order labelled as 1,2,3.
#The output images will be .png and can accept any image format.   

import os
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.image import imread 
from scipy.misc import imresize, imsave 


#Modifying the images to reduce their size

filepaths = [] 
for dir_ , _, files in os.walk(directory):
	for filename in files:
		relDir = os.path.relpath(dir_,directory)
		relfile = os.path.join(relDir, filename)
		filepaths.append(directory +"/"+ relfile)

for i, fp in enumerate(filepaths):
	img = imread(fp); #/255 
	print(fp) 
	img = imresize(img,size=(PIXEL_SIZE,PIXEL_SIZE),mode='RGB')
	#img = imresize(img,PIXEL_SIZE,PIXEL_SIZE)
	imsave(new_dir+"/"+str(i)+".png",img)

# filepaths_new = []
# for dir_ , _, files in os.walk(directory):
# 	for filename in files:
# 		if not filename.endswith(".png"):
# 			continue
# 		relDir = os.path.relpath(dir_,directory)
# 		relfile = os.path.join(relDir, filename)
# 		filepaths.append(directory +"/"+ relfile)

#definition of a method to access 40 x 40 x 3 face images
#BATCH SIZE is mentioned here

# def next_batch(num=64, data=filepaths_new):
# 	idx = np.arange(0,len(data)) 
# 	np.random.shuffle(idx)
# 	idx=idx[:num]
# 	data_shuffle= [imread(data[i]) for i in idx]
# 	shuffled = np.asarray(data_shuffle)
# 	return np.asarray(data_shuffle)

