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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ipdb, pdb

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


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# Caffe model of the det-FCN network
NETS = {'vgg16': ('VGG16',
                  'vgg16_detect_iter_70000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}-{}-{}-{}.png'.format(class_name, score, thresh, int(time.time())), bbox_inches='tight')
    #plt.draw()
    #fig.hold(True)

def demo(net, image_name, img_path, label_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file, label_file = load_img_label(image_name, img_path, label_path)
    im = cv2.imread(im_file)
    # Load the label
    label_obj = PIL.Image.open(label_file)
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
    im_pred, boxes = im_seg(net, im, label)
    timer.toc()
    print ('Detection-Segmentation took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    vis_seg(im, im_pred, label, boxes, cm, image_name)
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    #pdb.set_trace()
    #plt.figure()
    #for cls_ind, cls in enumerate(CLASSES[1:]):
    #    cls_ind += 1 # because we skipped background
    #    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #    cls_scores = scores[:, cls_ind]
    #    dets = np.hstack((cls_boxes,
    #                      cls_scores[:, np.newaxis])).astype(np.float32)
    #    keep = nms(dets, NMS_THRESH) # list of indices of the proposals that
        # contain scores > NMS_THRESH
    #    dets = dets[keep, :] # dets[:,:4]: bboxes; dets[-1]: score
    #    vis_detections(im, cls, dets, thresh=CONF_THRESH)

def vis_seg(im, im_pred, label, boxes, cm, img_name='img'):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(im[:,:,::-1]) # BGR->RGB
    plt.subplot(1,3,2)
    plt.imshow(im_pred,vmin=0, vmax=255, cmap=cm) # prediction
    plt.subplot(1,3,3)
    plt.imshow(label, vmin=0,vmax=255, cmap=cm) # gt
    plt.tight_layout()
    plt.show()
    #plt.savefig('img-{}.png'.format(img_name), bbox_inches='tight')

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
    data_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007/VOC2007')
    img_set_file = os.path.join(data_path, 'ImageSets/Segmentation/trainval.txt')
    img_path = os.path.join(data_path, 'JPEGImages') # ext: jpg
    label_path = os.path.join(data_path, 'SegmentationClass') # ext: png
    assert os.path.exists(img_set_file), 'img file not exist'
    assert os.path.exists(img_path), 'img path not exist'
    assert os.path.exists(label_path), 'label path not exist'
    # load the images
    im_names = load_index_from_gt(img_set_file)
    #im_names = load_img_label(im_indices[0], img_path, label_path)
    pdb.set_trace()
    # for each image, do a forward pass and get the predicted pixel labels
    #im_names = ['000004.jpg', '000014.jpg', '000025.jpg', '000062.jpg',
    #            '000069.jpg', '000176.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, img_path, label_path)
        break

    #pdb.set_trace()
    plt.show()


