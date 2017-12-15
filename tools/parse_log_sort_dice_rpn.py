import numpy as np

logname = 'experiments/logs/eval_socket_end2end_dice_VGG16.txt.2017-10-23_01-17-36'

with open(logname, 'r') as f:
    list_dice = []
    for line in f:
        if "Iteration RPNFCN" not in line:
            continue
        #import pdb;pdb.set_trace()
        dice = line.strip().rsplit(":", 1)[-1].strip()
        try:
            list_dice.append(float(dice))
        except:
            import pdb;pdb.set_trace()

import pdb;pdb.set_trace()
list_dice.sort()
np.savetxt("/data/dataset/hip/abhi/rpnfcn-sorted-dice.txt", list_dice)

import matplotlib.pyplot as plt

plt.plot(list_dice, 'b')
plt.show()

