import csv
import numpy as np
from PIL import Image
#import PIL.ImageOps    

# Create './data', './results_img' and './results_img/combined' before executing this script
# Image dimension 
DESIGN_NAME = 'adaptec2'
row =  849     # Height
column = 849   # Width


def normalize(arr):
    arr = arr.astype('float')
    # Do not touch the alpha channel
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0/(maxval-minval))
    return arr

# def demo_normalize():
#     img = Image.open(FILENAME).convert('RGB')  #Could be 'L','RGB' or 'RGBA'
#     arr = np.array(img)
#     new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGB')
#     new_img.save('/tmp/normalized.png')


with open('./data/congestion_heatmap.csv') as f:
    r = csv.reader(f)
    lst = []
    for line in r:          
        lst.extend(map(int, line))
    cong_array = np.array(lst)
    print('Cong_min is:' + (str)(cong_array.min()) + ' Cong_max is:' + (str)(cong_array.max()) + ' for design - ' + DESIGN_NAME)
    cong_array = normalize(cong_array)

imr = Image.fromarray(cong_array.astype('uint8').reshape(row,column), 'L')
    #img.show()
imr.save('./results_img/' + DESIGN_NAME + '_congestion.png')

with open('./data/pin_density.csv') as f:
    r = csv.reader(f)
    lst2 = []
    for line in r:          
        lst2.extend(map(int, line))
    pin_density = np.array(lst2)
    pin_density = normalize(pin_density)/2

img = Image.fromarray(pin_density.astype('uint8').reshape(row,column), 'L')
    #img.show()
img.save('./results_img/' + DESIGN_NAME + '_pin.png')

with open('./data/net_density.csv') as f:
    r = csv.reader(f)
    lst3 = []
    for line in r:          
        lst3.extend(map(int, line))
    net_density = np.array(lst3)
    net_density = normalize(net_density)/2

imb = Image.fromarray(net_density.astype('uint8').reshape(row,column), 'L')
    #img.show()
imb.save('./results_img/' + DESIGN_NAME + '_net.png')

with open('./data/maximum_capacity.csv') as f:
    r = csv.reader(f)
    lst4 = []
    for line in r:          
        lst4.extend(map(int, line))
    maximum_capacity = np.array(lst4)
    maximum_capacity = normalize(maximum_capacity)/2

imx = Image.fromarray(maximum_capacity.astype('uint8').reshape(row,column), 'L')
imy = Image.fromarray(maximum_capacity.astype('uint8').reshape(row,column), 'L')
rgbArray = np.zeros((row,column), 'uint8')
imz = Image.fromarray(rgbArray.astype('uint8').reshape(row,column), 'L')
im_cap=Image.merge("RGB",(imz,imx,imy))
    #img.show()
im_cap.save('./results_img/' + DESIGN_NAME + '_capacity.png')

# rgbArray = np.zeros((row,column), 'uint8')
# imtest = Image.fromarray(rgbArray.astype('uint8').reshape(row,column), 'L')
# imtest.save('test.png')
im_test     = Image.fromarray((maximum_capacity+pin_density).astype('uint8').reshape(row,column), 'L')
im_test_2   = Image.fromarray((maximum_capacity+net_density).astype('uint8').reshape(row,column), 'L')

merged=Image.merge("RGB",(imr,img,imb))
merged.save('./results_img/combined/' + DESIGN_NAME + '_combined.png')

merged=Image.merge("RGB",(imr,im_test,im_test_2))
merged.save('./results_img/combined/' + DESIGN_NAME + '_combined_w_capacity.png')
