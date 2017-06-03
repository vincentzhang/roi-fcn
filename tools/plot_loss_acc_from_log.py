# This script reads from log files,
# plot train loss
# of rpn cls loss, box regression loss and pixel-wise classification loss
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
#pdb.set_trace()

dir_name = "../experiments/logs/"
#dir_name = sys.argv[1]

#log_name = dir_name'detect_end2end_VGG16_.txt.2017-06-02_16-18-41'
#'detect_end2end_VGG16_.txt.2017-05-21_14-53-36'
fname = 'detect_end2end_VGG16_.txt.2017-06-02_16-30-06'
log_name = dir_name+fname

# Plotting the loss
print "Parse training loss"
train_pix_loss = []
train_cls_loss = []
train_bbox_loss = []
with open(log_name,'r') as f:
  for line in f:
    if "loss" in line and "#0" in line and "Train net output" in line:
        train_pix_loss.append(float(line.split('=')[1].split("(")[0].strip()))
    elif "loss" in line and "#1" in line and "Train net output" in line:
        train_cls_loss.append(float(line.split('=')[1].split("(")[0].strip()))
    elif "loss" in line and "#2" in line and "Train net output" in line:
        train_bbox_loss.append(float(line.split('=')[1].split("(")[0].strip()))


print("Length of pix_loss {}".format(len(train_pix_loss)))
print("Length of cls_loss {}".format(len(train_cls_loss)))
print("Length of box_loss {}".format(len(train_bbox_loss)))

print("For {} iterations".format(len(train_pix_loss)))
niter = len(train_pix_loss)
# Plot 1, Pixel-wise Loss, per batch (1 image)
f, axarr = plt.subplots(2)
axarr[0].plot(np.arange(len(train_pix_loss)), train_pix_loss)
#ax2 = ax1.twinx()
#ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
#ax2.set_ylabel('RPN loss')
axarr[0].set_xlabel('iteration')
axarr[0].set_title('FCN Loss')
axarr[0].set_ylabel('pixel-wise loss')
# Plot 2, loss on the bbox prediction, per bat
axarr[1].plot(np.arange(len(train_pix_loss)), train_cls_loss, 'b')
axarr[1].plot(np.arange(len(train_pix_loss)), train_bbox_loss, 'r')
axarr[1].set_xlabel('iteration')
axarr[1].legend(['cls_loss','bbox_loss'])
axarr[1].set_title("RPN Loss")
plt.ion()
plt.show()
plt.savefig(dir_name+'imgs/train_loss_'+str(niter)+'.png', bbox_inches='tight')
