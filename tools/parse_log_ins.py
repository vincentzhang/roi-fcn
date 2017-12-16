import numpy as np
import sys
#from tqdm import tqdm

if len(sys.argv) == 2:
    logname = '../experiments/logs/' + sys.argv[-1]
else:
#logname = '../experiments/logs/ins_scratch_n_m_fg10_end2end_VGG16_.txt.2017-12-15_17-07-50'
#logname = '../experiments/logs/ins_scratch_n_m_fg50_end2end_VGG16_.txt.2017-12-15_17-38-48'
    logname = '../experiments/logs/ins_scratch_n_m_fg50_e-4_end2end_VGG16_.txt.2017-12-15_18-19-33'
#  1389 I1215 02:24:18.601630 27007 solver.cpp:218] Iteration 0 (0 iter/s,
#          1.70903s/1 iters), loss = 1.38895                                                                                                         
#    1390 I1215 02:24:18.601667 27007 solver.cpp:237]     Train net output #0:
#    loss_cls = 0.577627 (* 1 = 0.577627 loss)                                                                                               
#      1391 I1215 02:24:18.601673 27007 solver.cpp:237]     Train net output #1:
#      rpn_cls_loss = 0.719531 (* 1 = 0.719531 loss)                                                                                           
#        1392 I1215 02:24:18.601677 27007 solver.cpp:237]     Train net output
#        #2: rpn_loss_bbox = 0.109774 (* 1 = 0.109774 loss)   

with open(logname, 'r') as f:
    count = 0
    loss = [] # total
    cls_loss = []
    bbox_loss = []
    seg_loss = []
    for line in f:
        if "loss" not in line or "solver.cpp" not in line:
            continue
        else:
            count +=1
            line = line.strip()
            print("processing line  {}".format(count))
            #import pdb;pdb.set_trace()
            if " loss =" in line:
                loss.append(float(line.rsplit(",",1)[-1].strip().split("=",1)[-1].strip()))
            elif "loss_cls" in line:
                seg_loss.append(float(line.rsplit('=',1)[-1].strip().split(' ')[0].strip()))
            elif "rpn_cls_loss" in line:
                cls_loss.append(float(line.rsplit('=',1)[-1].strip().split(' ')[0].strip()))
            elif "rpn_loss_bbox" in line:
                bbox_loss.append(float(line.rsplit('=',1)[-1].strip().split(' ')[0].strip()))
            else:
                assert False,"not in any class"


#np.save('tmpresult/loss_ins',(loss,cls_loss,bbox_loss,seg_loss))
np.save('tmpresult/loss_ins',(loss,cls_loss,bbox_loss,seg_loss))

