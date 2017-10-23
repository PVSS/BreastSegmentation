# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:28:35 2017
This file preprocess the data and thresholds the data.
Takes some manual input for decision making
@author: 310127474
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from IPython.display import clear_output

CURR_DIR = os.getcwd()
IMG_DIR = 'C:\\Users\\310127474\\DDSMData\\MIAS\\MIAS\\'
SAVE_DIR = 'C:\\Users\\310127474\\DDSMData\\MIAS\\MIAS\\Thresholded\\'

def get_hists(image, b):
    hist, bins = np.histogram(img.flatten(), bins=b, range=[0,255])
    cdf = hist.cumsum()
    cdf_normalized = cdf *hist.max()/ cdf.max()
    
    return [hist, cdf_normalized]
def plot(img, img_hists):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    
    plt.subplot(122)
    plt.plot(img_hists[1], color = 'b')
    plt.plot(img_hists[0], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    plt.subplots_adjust(top=0.92, bottom=0.08,
                        left=0.10, right=0.95,
                        hspace=0.25, wspace=0.35)
    
def threshold(img_list, factor = 0.4, select_files = []):
    images_t = []
    
    def internal(data):
        thresholded = cv2.threshold(data['clahe_img'],
                                    np.median(data['clahe_img']) * factor, 255,
                                    cv2.THRESH_BINARY)[1]     # just the binary image
            
        _, l, s, _ = cv2.connectedComponentsWithStats(thresholded)
        images_t.append( {'filename': data['filename'],
                          'clahe_img': data['clahe_img'],
                          'thresh_img': thresholded,
                          'factor': factor,
                          'labels':l,                          # labels: contiguous regions in mammogram, labelled
                          'count':s[:, -1]                     # count: count of pixels in each discrete object
                         })
    
    if not select_files:
        print ('Processing all files')
        for i, data in enumerate(img_list):
            internal(data)
            
    else:
        print('Processing select files {}'.format(select_files))
        for i, data in enumerate(img_list):
            if data['filename'] in select_files:
                internal(data)
                
    return images_t
def save(fn, img, location=SAVE_DIR):
    print( 'Saving: {}'.format(location + fn))
    cv2.imwrite(location + fn, img)
    time.sleep(2)
def mask(image, labels, region):
    labels = copy.deepcopy(labels)  # create a full, unique copy of labels
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if labels[row, col] != region:
                labels[row, col] = 0  # mask the artifact
            else:
                labels[row, col] = 1  # retain the breast
    return labels
def clean_art(images_thresh):
    revist = []
    for i, data in enumerate(images_thresh):
        fn, c_img, t_img = data['filename'], data['clahe_img'], data['thresh_img']
        print( 'Processing File: {}'.format(fn))

        plt.subplot(121)
        plt.imshow(c_img, cmap='gray')
        plt.title('Original')
        plt.subplot(122)
        plt.imshow(t_img, cmap='gray')
        plt.title('Binary Threshold')
        plt.show()
        plt.pause(0.1)
        
        top_regions = np.argpartition(data['count'], -2)[-2:]
        print(len(top_regions))
        top_counts = data['count'][top_regions]
        print ('Top region pixel counts: {}'.format(top_counts))
        my_mask = mask( t_img, data['labels'], region=top_regions[1])
        image = c_img * my_mask
        
        image = np.array(image, dtype = np.uint8)
        
        plt.imshow(image, cmap='gray')
        plt.title(fn)
        plt.show()
        plt.pause(0.1)
                
        input4 = input("Save post processed image (Y/N): ").lower()
        if input4 == 'y':
            save(fn, image)
        
        clear_output()
    return revist


filenames = [ filename for filename in os.listdir(IMG_DIR) if filename.endswith('.pgm')]

clahe_images = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
os.chdir(IMG_DIR)
for filename in filenames:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    clahe_images.append({'filename': filename, 'clahe_img': clahe.apply(img)})
os.chdir(CURR_DIR)

images_thresh = threshold(clahe_images)
print (len(images_thresh))

remaining = clean_art(images_thresh)
remaining_fn = [item['filename'] for item in remaining]

#images_thresh2 = threshold(clahe_images, factor=7, select_files=remaining_fn)
#remaining_fn_2 = clean_art(images_thresh2)

#print ('Total number of CLAHE images: {}'.format(np.count_nonzero(clahe_images)))
#img = clahe_images[99]['clahe_img']
#img_hists = get_hists( img, b=256)
#
#plot(img, img_hists)








#filename = path +'mdb099.pgm'
#x = 714
#y = 340
#r = 23
#
#import cv2
#im = cv2.imread(filename,-1)
#cv2.circle(im,(x,1024-y),r,(255,0,0))
#import matplotlib.pyplot as plt
#
#plt.imshow(im,cmap='gray')

