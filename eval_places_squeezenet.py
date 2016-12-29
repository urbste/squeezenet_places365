import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.nonlinearities import rectify, softmax, linear

import theano
import theano.tensor as T
import pickle
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg
import scipy.ndimage as ndi

from PIL import Image
import random
import time

from squeezenetv2 import build_squeeznetv2

import cv2

from draw_net_graph import draw_to_file, make_pydot_graph
import multiprocessing
from multiprocessing import Process, Queue


############### load mean image file ###############
MEAN_IMAGE = np.load('places365_mean_image.npy')
_,c, h, w = MEAN_IMAGE.shape
MEAN_IMAGE_CENTER_CROP = MEAN_IMAGE[0,:,h//2-112:h//2+112, w//2-112:w//2+112]

BATCH_SIZE = 100
inputH = 224
inputW = 224
max_num_categories = 365
base_path = 'PATHTOPLACES' # change this to your places365 data folder
img_dir = base_path+'val_256/'
############### load training image paths with label from file ###############
filenames = []
label = []
training_data_tuples = tuple(open(base_path+'places365_val.txt', 'r'))
for l in training_data_tuples:
    splitted = l.split()
    # only train on first 100 categories
    if int(splitted[1]) < max_num_categories:
        filenames.append(splitted[0])
        label.append(splitted[1])
    
nrImagesTotal = len(filenames)
val_list_idx = np.arange(0,nrImagesTotal)
############### build the model, set the weights, compile theano functions ###############
# build model and load weights 
input_img = T.tensor4()
target_var = T.ivector('targets')
print('building squeezenet')
block_names = ['fire'+str(i)+'/' for i in range(2,10)]
squeezenet = build_squeeznetv2(input_img,block_names,max_num_categories)

# set model weights from pre-trained network
with open('squeezenet_42.15.params','rb') as f:
    modelweights = pickle.load(f)
lasagne.layers.set_all_param_values(squeezenet['prob'],modelweights)


print('compiling estimation function')
# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(squeezenet['prob'], deterministic=True)
predict_fn = theano.function([input_img], test_prediction)

############### start training ###############                  
print('Start evaluation')  
target_file = open('result_places365_squeezenet.txt', 'w')
  
for start_idx in range(0, nrImagesTotal + 1, BATCH_SIZE):
    if (start_idx+BATCH_SIZE) < nrImagesTotal:
        excerpt = range(start_idx, start_idx + BATCH_SIZE)   
    else:
        excerpt = range(start_idx, nrImagesTotal)
            
    if (len(excerpt) > 0):   
        X_data = np.zeros((len(excerpt),3,inputH,inputW),dtype=np.float32)
        for i in range(len(excerpt)):
            idx = val_list_idx[excerpt[i]]
            path2img = img_dir+filenames[idx]
            Xtemp = np.asarray(Image.open(path2img.rstrip('\n')),dtype=np.float32)
            h, w, _ = Xtemp.shape
            Xsmall = Xtemp[h//2-112:h//2+112, w//2-112:w//2+112]
            Xsmall = np.transpose(Xsmall,(2,0,1)) - MEAN_IMAGE_CENTER_CROP
            X_data[[i],:,:,:] = np.float32(Xsmall)/255. 

        y_predicted = predict_fn(X_data)
        sorted_array = np.argsort(y_predicted,axis=1)
        sorted_array = sorted_array[:,::-1]
        y_res = sorted_array[:,0:5]
        for f in range(len(excerpt)):
            target_file.write(filenames[val_list_idx[excerpt[f]]]+' ')
            for r in range(y_res.shape[1]):
                target_file.write(str(y_res[f,r])+' ')
            target_file.write('\n')


            
target_file.close()