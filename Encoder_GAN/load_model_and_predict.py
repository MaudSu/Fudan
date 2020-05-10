from __future__ import print_function, division
# Create './predicted_imgs' before execution
# Must have './saved_model/<model_name.json>' and './saved_model/<model_name_weights.hdf5>'
# To use, -python3 load_model_and_predict.py <input_image_DIR> <name_to_label_output>
import sys
import os
import time
# Not all libraries below are used, no harm just to leave them here
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave

import numpy as np
from PIL import Image


CROPPED_SIZE    = 64
IMG_ROWS        = CROPPED_SIZE   #This is pixel height, make sure height and width are the same
IMG_COLS        = CROPPED_SIZE   #This is pixel width
CHANNEL         = 3              #These are the number of channels 
IMG_DIR         = sys.argv[1]
DESIGN_NAME     = sys.argv[2]

def normalize(arr):
    arr = arr.astype('float')
    # Do not touch the alpha channel
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (1/(maxval-minval))
    return arr

def load_model(model_name):
    model_path = "saved_model/%s.json" % model_name
    weights_path = "saved_model/%s_weights.hdf5" % model_name
    options = {"file_arch": model_path,
                "file_weight": weights_path}

    json_file = open(options['file_arch'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(options['file_weight'])

    return loaded_model

    

# Load the trained Generator model
generator_model = load_model("generator")

# Save archtecture and weight to single file
#generator_model.save('my_GAN.h5')

# Original input in its entirety 
img             = imread(IMG_DIR)
height          = img.shape[0]
width           = img.shape[1]

num_rows        = height//CROPPED_SIZE
num_columns     = width//CROPPED_SIZE


#Crop original input into smaller pieces
imgs= np.ndarray((num_rows*num_columns,IMG_ROWS,IMG_COLS,CHANNEL),'float32')
masked_imgs= np.ndarray((num_rows*num_columns,IMG_ROWS,IMG_COLS,CHANNEL),'float32')
counter = 0
for i in range(0,num_rows):
    for j in range(0,num_columns):
        left    = j*CROPPED_SIZE
        top     = i*CROPPED_SIZE
        right   = left + CROPPED_SIZE
        bottom  = top + CROPPED_SIZE

        cropped_img = img[top:bottom,left:right,:]
        imgs[counter] = cropped_img
        cropped_img[:,:,0] = 0
        masked_imgs[counter] = cropped_img

        counter += 1

#Predict using loaded generator
start_time = time.time()
gen_missing = generator_model.predict(masked_imgs)
final_time = time.time() - start_time
print ("{} has a prediction time of {}".format(DESIGN_NAME,final_time))
# r, c = 3, 20
# imgs = 0.5 * imgs + 0.5
# masked_imgs = 0.5 * masked_imgs + 0.5
# gen_missing = 0.5 * gen_missing + 0.5

# fig, axs = plt.subplots(r, c)
# for i in range(c):
#     axs[0,i].imshow(imgs[i, :,:])
#     axs[0,i].axis('off')
#     axs[1,i].imshow(masked_imgs[i, :,:])
#     axs[1,i].axis('off')
#     filled_in = imgs[i].copy()
#     filled_in[:, :, :] = gen_missing[i,:,:,:]
#     axs[2,i].imshow(filled_in)
#     axs[2,i].axis('off')
# fig.savefig("test_image.png")
# plt.close()

#Extract predicted image
counter = 0
for i in range(0,num_rows):
    for j in range(0,num_columns):
        left    = j*CROPPED_SIZE
        top     = i*CROPPED_SIZE
        right   = left + CROPPED_SIZE
        bottom  = top + CROPPED_SIZE

        img[top:bottom,left:right,:] = gen_missing[counter,:,:,:]
        counter += 1

extracted_img = img.copy()
extracted_img[:,:,:] = 0

actual_height = (height-1)//2
actual_width  = (width-1)//2


for i in range(0, actual_width-2):
    for j in range(0, actual_height-1):
        extracted_img[(2*j+1),(2*i+1+1),0] = img[(2*j+1),(2*i+1+1),0]

for i in range(0, actual_width-1):
    for j in range(0, actual_height-2):
        extracted_img[(2*j+1+1),(2*i+1),0] = img[(2*j+1+1),(2*i+1),0]


#Post processing
for i in range(0, height):
    for j in range(0, width):
        if (extracted_img[i,j,0] < 0):
            extracted_img[i,j,0] = 0




#Dumpy conversion of [0...1] to [0...255]
imsave("./predicted_imgs/final_image_" + DESIGN_NAME + ".png", extracted_img)
img_to_be_converted = Image.open("./predicted_imgs/final_image_" + DESIGN_NAME + ".png")
img_array           = np.asarray(img_to_be_converted)
img_array_r         = img_array[:,:,0]
img_array_r         = img_array_r.flatten()
with open('./predicted_imgs/congestion_' + DESIGN_NAME + '.csv', 'w') as f:
    for item in img_array_r:
        f.write("%s\n" % item)


#Output readable images
extracted_img = normalize(extracted_img)
imsave("./predicted_imgs/final_image_" + DESIGN_NAME + ".png", extracted_img)

img_congestion_heatmap = imread(IMG_DIR)
img_congestion_heatmap[:,:,1]=0
img_congestion_heatmap[:,:,2]=0
imsave("./predicted_imgs/original_image_" + DESIGN_NAME + ".png", img_congestion_heatmap)
