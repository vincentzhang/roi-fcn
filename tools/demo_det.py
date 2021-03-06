#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections based FCN in sample images.
"""
import matplotlib
#matplotlib.rcParams['figure.dpi'] = 600
#matplotlib.rcParams['savefig.dpi'] = 100
#matplotlib.rc('font', size=6)
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_seg
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time
import PIL
from VOClabelcolormap_hor import color_map_viz

import ipdb, pdb


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# Caffe model of the det-FCN network
NETS = {'vgg16': ('VGG16',
                  'vgg16_detect_iter_70000.caffemodel')}

def demo(net, image_name, img_path, label_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file, label_file = load_img_label(image_name, img_path, label_path)
    im = cv2.imread(im_file)
    # Load the label
    try:
        label_obj = PIL.Image.open(label_file)
        show_label = True
    except:
        # for test, there's no labels
        show_label = False
        label_obj = PIL.Image.open("/data/repo/py-faster-rcnn/data/VOCdevkit2007/VOC2011/SegmentationClass/2007_000033.png")
    palette = label_obj.getpalette()
    label = np.asarray(label_obj)
    max_val = float(np.iinfo(label.dtype).max)
    cmap = np.array(palette).reshape(len(palette)/3, 3)/max_val
    # create colormap obj
    cm = LinearSegmentedColormap.from_list('label_cm',cmap, N=cmap.shape[0])
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    cfg.TEST.HAS_RPN = True
    im_pred, boxes, boxes_score = im_seg(net, im, label)
    timer.toc()
    print ('Detection-Segmentation took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    NMS_THRESH = 0.3
    dets = np.hstack((boxes,boxes_score)).astype(np.float32)
    vis_seg(im, im_pred, label, dets, cm, image_name, show_label, NMS_THRESH)

def vis_seg(im, im_pred, label, dets, cm, img_name='img', display_label=True,
        thresh=0.3):
    fig = plt.figure()
    gs1 = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs1[0])
    #ax = fig.add_subplot(2,3,1)
    ax.imshow(im[:,:,::-1]) # BGR->RGB
    #pdb.set_trace()
    # Display all boxes
    keep = nms(dets, thresh)
    dets = dets[keep, :] # dets[:,:4]: bboxes; dets[-1]: score
    print("number of boxes to keep: {}".format(len(keep)))
    for box in dets:
        ax.add_patch(
            plt.Rectangle((box[0], box[1]),
                  box[2] - box[0],
                  box[3] - box[1], edgecolor='red', linewidth=2,
                  facecolor='blue', alpha=0.1)
        )
    ax = fig.add_subplot(gs1[1])
    #ax = fig.add_subplot(2,3,2)
    ax.imshow(im_pred,vmin=0, vmax=255, cmap=cm) # prediction
    if display_label:
        #ax = fig.add_subplot(2,3,3)
        ax = fig.add_subplot(gs1[2])
        ax.imshow(label, vmin=0,vmax=255, cmap=cm) # gt
    gs2 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs2[0])
    #ax = fig.add_subplot(2,1,2)
    color_map_viz(ax)
    gs1.tight_layout(fig, rect=[0,0.9, 1, 1])
    gs2.tight_layout(fig, rect=[0,0,1,1], h_pad=0.1)
    #plt.subplots_adjust(right=0.85)
    #fig.set_size_inches(8, 6)
    fig.savefig('rpn70k-{}.png'.format(img_name), bbox_inches='tight')
    plt.show()
    #plt.savefig('rpn70k-{}.png'.format(img_name),dpi=fig.dpi)# bbox_inches='tight',

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def load_index_from_gt(img_set_file):
    # read from img_set_file and return the image indices
    with open(img_set_file) as f:
        img_index = [x.strip() for x in f.readlines()]
    return img_index

def load_img_label(idx, img_path, label_path):
    # Load file names for image and label
    img = os.path.join(img_path, idx) + '.jpg'
    label = os.path.join(label_path, idx) + '.png'
    return img, label

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.RPN_POST_NMS_TOP_N = 20
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'detect_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #for i in xrange(2):
    #    _, _= im_pred(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    # data/VOCdevkit2007/VOC2007/ImageSets/Segmentation/trainval.txt
    #data_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007/VOC2007')
    data_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007/VOC2011')
    img_set_file = os.path.join(data_path, 'ImageSets/Segmentation/trainval.txt')
    #img_set_file = os.path.join(data_path, 'ImageSets/Segmentation/val.txt')
    img_path = os.path.join(data_path, 'JPEGImages') # ext: jpg
    label_path = os.path.join(data_path, 'SegmentationClass') # ext: png
    assert os.path.exists(img_set_file), 'img file not exist'
    assert os.path.exists(img_path), 'img path not exist'
    assert os.path.exists(label_path), 'label path not exist'
    # load the images
    im_names = load_index_from_gt(img_set_file)
    #im_names = load_img_label(im_indices[0], img_path, label_path)
    #pdb.set_trace()
    # for each image, do a forward pass and get the predicted pixel labels
    #im_names = ['000004.jpg', '000014.jpg', '000025.jpg', '000062.jpg',
    #            '000069.jpg', '000176.jpg']
    for im_name in im_names:
        #im_name = '2008_000019'
        im_name = '2008_000187'
        #im_name = '000333'
        im_name = '2007_000032'
        #im_name = '001377'
        #im_name = '2008_000001'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, img_path, label_path)
        break

    #pdb.set_trace()
    #plt.show()


