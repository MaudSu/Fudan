#Create './results_img/cropped' before executing this script
#To use, -python3 image_crop.py <image_directory> <crop_size> <name_to_label_image>

import sys
from PIL import Image

IMG_DIR         = sys.argv[1]
CROPPED_SIZE    = (int)(sys.argv[2])
DESIGN_NAME     = sys.argv[3]

img             = Image.open(IMG_DIR)
width, height   = img.size

num_rows        = height//CROPPED_SIZE
num_columns     = width//CROPPED_SIZE
print( (str)(num_rows*num_columns) + " images in total will be created.")

#Left, Top, Right, Bottom
refine_area = (0, 0, width-width%CROPPED_SIZE, height-height%CROPPED_SIZE)
refined_img = img.crop(refine_area) 

print("The refined image size is %s" % (refined_img.size,))
#Starting from the top-left corner, to crop image into smaller pieces
counter = 0
for i in range(0,num_rows):
    for j in range(0,num_columns):
        left    = j*CROPPED_SIZE
        top     = i*CROPPED_SIZE
        right   = left + CROPPED_SIZE
        bottom  = top + CROPPED_SIZE

        crop_area   = (left, top, right, bottom)
        cropped_img = refined_img.crop(crop_area)

        counter += 1
        cropped_img.save('./results_img/cropped/' + DESIGN_NAME + '_' + str(counter) + '.png')