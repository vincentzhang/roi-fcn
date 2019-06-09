#!/usr/bin/env python
"""Demo of ROI-FCN on automatic acetabulum segmentation"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_seg
from utils.timer import Timer
import os, sys
import caffe
import pprint
import cv2 # required to read and resize image
import time
import numpy as np


''' May need to change these paths if file location differs for data, experiments etc'''
def load_model():
    """ return the net model """
    gpu_id = 0 # default to use the first GPU available
    base_dir = './'
    print("base_dir: {}".format(base_dir))
    prototxt = base_dir + 'models/socket/VGG16/detect_end2end/demo.prototxt'
    caffemodel = base_dir + 'data/vgg16_acce_fg50_1e-04_iter_42960.caffemodel'
    cfg_file = base_dir + 'experiments/cfgs/acce_end2end.yml'

    cfg_from_file(cfg_file)
    cfg.GPU_ID = gpu_id

    #print('Using config:')
    #pprint.pprint(cfg)

    assert os.path.exists(caffemodel), 'Cannot find {}'.format(caffemodel)

    start = time.time()
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu() # call this line if only cpu is present
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Loading the model took {:.3f}s'.format(time.time()-start))
    return net

def autoseg(image_name, net=None):
    """ Auto-segmentation

        Params:
            net: the network weights of pre-trained network
    """
    if not net: # if the network model isn't given
        net = load_model()
    caffe.set_mode_gpu()
    # Load the demo image
    if isinstance(image_name, basestring):
        im = cv2.imread(image_name)
    else:
        im = image_name
    #import pdb;pdb.set_trace()
    if im.ndim != 3:
        # need make 3 channels if it's a 2 channel- image
        im = np.dstack((im,im,im))
    start = time.time()
    print("Starting detection-segmentation")
    pred = im_seg(net, im)
    print ('Detection-Segmentation took {:.3f}s').format(time.time()-start)
    if pred[0].ndim == 3:
        return pred[0][0, ...]
    else:
        return pred[0]

