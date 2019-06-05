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
loss, cls_loss, bbox_loss, seg_loss = np.load('tmpresult/loss_ins.npy')
#import pdb;pdb.set_trace()

#plt.plot(loss, 'k', label='total')
plt.plot(cls_loss,'b', label='box class')
plt.plot(bbox_loss, 'g', label='box reg')
#plt.plot(seg_loss, 'r', label='seg')
#plt.ylim(-0.5, 50.0)
plt.legend()
if save:
    plt.savefig("loss_haha.png", bbox_inches="tight")
else:
    plt.show()

