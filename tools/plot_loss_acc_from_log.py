# This script reads from log files,
# plot train loss
# of rpn cls loss, box regression loss and pixel-wise classification loss
import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

dir_name = "experiments/logs/"
#dir_name = sys.argv[1]

#log_name = dir_name'detect_end2end_VGG16_.txt.2017-06-02_16-18-41'
#'detect_end2end_VGG16_.txt.2017-05-21_14-53-36'
#fname = 'detect_end2end_VGG16_.txt.2017-06-02_16-30-06'
#fname = 'hip_end2end_VGG16_.txt.2017-07-03_01-59-30'
#fname = 'hiph5.txt'
#fname = 'socket.txt'
#fname = 'hiph5_end2end_VGG16_.txt.2017-08-10_18-15-18'
#fname = 'hiph5_scratch_end2end_VGG16_.txt.2017-08-18_00-17-14'
# lr 1e-5
fname = 'hiph5_scratch_end2end_VGG16_.txt.2017-08-19_17-01-36'
# lr 1e-6
#fname = 'hiph5_scratch_n_end2end_VGG16_.txt.2017-08-19_18-26-04'
#fname = 'hiph5_end2end_VGG16_.txt.2017-08-10_18-32-19'
log_name = dir_name+fname

# Plotting the loss
print "Parse training loss"
train_pix_loss = []
train_smo_loss = []
train_cls_loss = []
train_bbox_loss = []
with open(log_name,'r') as f:
  for line in f:
    if "loss" in line and "#0" in line and "Train net output" in line:
        train_pix_loss.append(float(line.split('=')[1].split("(")[0].strip()))
    elif "loss" in line and "Iteration" in line:
        train_smo_loss.append(float(line.rsplit('=')[1].strip()))
    elif "loss" in line and "#1" in line and "Train net output" in line:
        train_cls_loss.append(float(line.split('=')[1].split("(")[0].strip()))
    elif "loss" in line and "#2" in line and "Train net output" in line:
        train_bbox_loss.append(float(line.split('=')[1].split("(")[0].strip()))


print("Length of pix_loss {}".format(len(train_pix_loss)))
print("Length of smo_loss {}".format(len(train_smo_loss)))
print("Length of cls_loss {}".format(len(train_cls_loss)))
print("Length of box_loss {}".format(len(train_bbox_loss)))

print("For {} iterations".format(len(train_pix_loss)))
niter = len(train_pix_loss)
# Plot 1, Pixel-wise Loss, per batch (1 image)
f, axarr = plt.subplots(2)
axarr[0].plot(np.arange(len(train_pix_loss)), train_pix_loss, 'b')
axarr[0].plot(np.arange(len(train_smo_loss)), train_smo_loss, 'r')
#ax2 = ax1.twinx()
#ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
#ax2.set_ylabel('RPN loss')
axarr[0].set_xlabel('iteration')
axarr[0].legend(['pix_loss','smoothed loss'])
axarr[0].set_title('FCN Loss')
axarr[0].set_ylabel('pixel-wise loss')
# Plot 2, loss on the bbox prediction, per bat
axarr[1].plot(np.arange(len(train_pix_loss)), train_cls_loss, 'b')
axarr[1].plot(np.arange(len(train_pix_loss)), train_bbox_loss, 'r')
axarr[1].set_xlabel('iteration')
axarr[1].legend(['cls_loss','bbox_loss'])
axarr[1].set_title("RPN Loss")
#plt.ion()
#plt.show()
#plt.savefig(dir_name+'imgs/hip_train_loss_'+str(niter)+'.png', bbox_inches='tight')
#plt.savefig(dir_name+'imgs/hip_train_loss_'+str(niter)+'.png', bbox_inches='tight')
#plt.savefig(dir_name+'imgs/hiph5_train_loss_'+str(niter)+'.png', bbox_inches='tight')
#plt.savefig(dir_name+'imgs/socket_train_loss_'+str(niter)+'.png', bbox_inches='tight')
#plt.savefig(dir_name+'imgs/hiph5_scratch_train_loss_'+str(niter)+'.png', bbox_inches='tight')
plt.savefig(dir_name+'imgs/'+fname+'-'+str(niter)+'.png', bbox_inches='tight')
