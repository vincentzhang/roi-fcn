from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import pdb


def plot():
    with open('rois.p', 'rb') as f:
        rois = pickle.load(f, encoding='latin1')
    # draw rectangle of the image region
    #n = rois.shape[0]
    rect = Rectangle((0, 0), 800, 600, facecolor="none", edgecolor="black",
            linewidth=5)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(rect)
    ax.set_xlim(-10, 850)
    ax.set_ylim(-10, 650)
    ax.invert_yaxis()
    # draw rois
    #pdb.set_trace()
    i = 0
    fg_rois = len([1 for j in rois if j[0] != 0])
    for roi in rois:
        if roi[0] == 0: #background class
            roi_rect = Rectangle( (roi[3], roi[2]), roi[5]-roi[3], roi[4]-roi[2],
               alpha=0.1, facecolor="blue", edgecolor="blue" )
        else:
            continue
            roi_rect = Rectangle( (roi[3], roi[2]), roi[5]-roi[3], roi[4]-roi[2],
               facecolor='red', alpha=0.1,  edgecolor=(1-i/float(fg_rois),
                   i/float(fg_rois), 0), linewidth=1, label='class = {}'.format(roi[0]))
               #facecolor="none", edgecolor="Red", linewidth=3 )
            i += 1
        ax.add_patch(roi_rect)
    #ax.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
    plt.savefig("roi_fig.png", bbox_inches="tight")


plot()
