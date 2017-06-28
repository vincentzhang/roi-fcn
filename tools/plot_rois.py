from __future__ import print_function
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import sys
import numpy as np
import pdb

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def plot():
    """ This rois.p is generated from this image 
        # image source
        img = '/data/repo/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/004622.jpg'
    """
    with open('rois.p', 'rb') as f:
        if sys.version_info > (3,0):
            rois = pickle.load(f, encoding='latin1')
        else:
            rois = pickle.load(f)
    # draw rectangle of the image region
    rect = Rectangle((0, 0), 800, 600, facecolor="none", edgecolor="black",
            linewidth=5)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(rect)
    ax.set_xlim(-10, 850)
    ax.set_ylim(-10, 650)
    ax.invert_yaxis()
    #pdb.set_trace()
    # draw rois
    # rois: [class, 0, x1,y1,x2,y2]
    i = 0
    fg_rois = np.count_nonzero(rois[:,0])
    for roi in rois:
        if roi[0] == 0: #background class in blue
            continue
            roi_rect = Rectangle( (roi[3], roi[2]), roi[5]-roi[3], roi[4]-roi[2],
               alpha=0.1, facecolor="blue", edgecolor="blue" )
        else:
            # continue
            roi_rect = Rectangle( (roi[3], roi[2]), roi[5]-roi[3], roi[4]-roi[2],
               facecolor='red', alpha=0.1,  edgecolor=(1-i/float(fg_rois),
                   i/float(fg_rois), 0), linewidth=1, label='class = {}'.format(CLASSES[int(roi[0])]))
            i += 1
        ax.add_patch(roi_rect)
    ax.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
    plt.show()
    #plt.savefig("roi_fig.png", bbox_inches="tight")


plot()
