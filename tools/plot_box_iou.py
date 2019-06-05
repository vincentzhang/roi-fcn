import os
save = False
try:
    env = os.environ['DISPLAY']
except KeyError:
    import matplotlib
    matplotlib.use('Agg')
    save = True
import matplotlib.pyplot as plt
import numpy as np

num_list = [1] + [10 * i for i in list(range(1,31))]
iou = np.load('mean_iou.npy')
iou_img = np.load('mean_iou_with_img.npy')

plt.figure
linewidth = 2.0
c3 =(78./255,185./255,95./255) 
c1 =(43./255,171./255,226./255)
plt.plot(num_list, iou, color=c1, linewidth=linewidth, label='IOU with object')
plt.plot(num_list, iou_img, color=c3, linewidth=linewidth, label='IOU with image')
plt.xlabel('Number of ROI proposals')
plt.ylabel('IOU')
plt.legend()
plt.title('The bounding box IOU versus the number of ROI proposals')
if save:
    plt.tight_layout()
    plt.savefig("iou.pdf", bbox_inches="tight", dpi=600)
else:
    plt.show()

