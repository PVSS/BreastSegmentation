# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:51:24 2017

@author: 310127474
"""

import csv
import collections
from scipy import misc
import numpy as np
import os
import gc
import shutil

def load_meta(metaFile, patho_idx, file_idx):
    bcMetaFile = {}
    bcCounts = collections.defaultdict(int)
    
    with open(metaFile, 'r') as csvfile:
        bcCSV = csv.reader(csvfile)
        headers = next(bcCSV)
        for row in bcCSV:
            patho = row[ patho_idx].lower()
            bcMetaFile[ row[file_idx]] = patho
            bcCounts[patho] += 1
    
        for k in bcCounts:
            print( '{0:10}: {1}'.format(k, bcCounts[k]))
            
    
    return bcMetaFile, bcCounts

def clean_meta(meta_data, imgPath):
    print ('Number of entries in incoming meta_data: {}'.format(len(meta_data)))
    found = 0
    not_found = 0
    notfoundfn = []
    for i, fn in enumerate(meta_data.keys()):
        filepath = os.path.join(imgPath, fn)
        if os.path.exists(filepath):
            found += 1
        else:
            notfoundfn.append(fn) 
            not_found += 1

    for f in notfoundfn:
        del meta_data[f]
        
    print ('Images found: {}'.format(found))
    print ('Images missing: {}'.format(not_found))
    print ('Number of entries of outgoing meta_data: {}'.format(len(meta_data)))

    return meta_data
    
def load_data(metaData, imgPath, categories, imgSize = (255,255), imgResize = None,
              verbose = True,  verboseFreq = 20):
    total = len(metaData)
    
    x, y = imgSize
    if imgResize is not None:
        x, y = imgResize
    
    # Allocate containers for the data
    X_data = np.zeros( [total, x, y])
    Y_data = np.zeros( [total, 1], dtype=np.int8)
    
    # Load images based on meta_data:
    for i, fn in enumerate( metaData.keys()):
        filepath = os.path.join(imgPath, fn)
        if os.path.exists(filepath):
            img = misc.imread(filepath, flatten=True)
        else:
            img = None
            print ("Not Found: " + filepath)
            
        if imgResize is not None:
            img = misc.imresize(img, imgResize)
            gc.collect()
        
        X_data[i] = img
        Y_data[i] = categories[ metaData[fn].lower()]
        
    X_data = X_data.astype('float32')
    X_data /= float(255)

    return X_data, Y_data

def reverseDict(d):
    ndxBC = {}
    for k in d:
        ndxBC[d[k]] = k

    return ndxBC
    
def get_clsCnts(y_data, cats):
    ys = np.ravel(y_data)
    labels = reverseDict(cats)
    bcCounts = collections.defaultdict(int)

    for lab in ys:
        bcCounts[lab] += 1
    try:
        for key, value in labels.items():
            bcCounts[value] = bcCounts.pop(key)
    except KeyError:
        pass
    return bcCounts

def balanceViaSmote(cls_cnts, meta_info, data_dir, aug_dir, catagories,
                    datagen, X_data, Y_data, imgResize = None, seed=None, verbose=False):
    aug_imgs = []

    if seed:
        np.random.seed(seed)

    max_class_key = max(cls_cnts, key=cls_cnts.get)
    max_class_val = cls_cnts[ max_class_key ]

    for key, value in cls_cnts.items():
        if key == max_class_key:
            pass
        else:
            grow_by = max_class_val - value
            imgs = {k:v for k, v in meta_info.items() if v == key}
            # take a random selection of grow_by size, with replacement
            key_indxs = np.random.choice(list(imgs.keys()), size=grow_by, replace=True)
            for k1 in key_indxs:
                aug_imgs.append({k:v for k,v in imgs.items() if k == k1})

            save_dir = aug_dir + key + '/'

            # Overwrite folder and contents if folder exists:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            # Load randomly selected images of given catagory into memory
            aug_X = list()
            aug_Y = list()
            for i in aug_imgs:
                img_x, img_y = load_data(i, data_dir, catagories, imgResize=imgResize)
                aug_X.append(img_x)
                aug_Y.append(img_y)

            # Generate augmented images
            aug_X = np.reshape(aug_X, (len(aug_X), aug_X[0].shape[0], aug_X[0].shape[1], aug_X[0].shape[2]))

            for x_batch, y_batch in datagen.flow(aug_X, aug_Y, batch_size=len(aug_X), seed=seed,
                                                 save_to_dir= save_dir,
                                                 save_prefix= key + '_aug',
                                                 save_format= 'png'):
                X_data = np.concatenate(
                    (X_data, np.reshape(x_batch, (len(x_batch), x_batch.shape[2], x_batch.shape[3]))))
                Y_data = np.concatenate((Y_data, np.reshape(y_batch, (len(y_batch), y_batch.shape[2]))))
                break

    if verbose:
        bcCounts = get_clsCnts(Y_data, catagories)
        
        for k in bcCounts:
            print ('{0:10}: {1}'.format(k, bcCounts[k]))

    return X_data, Y_data
