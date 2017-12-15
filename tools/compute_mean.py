""" Compute the mean of the grayscale image in the training set of hiph5

Output: 
Processing volume AiIs
Processing volume AlAl
Processing volume CeLe
Processing volume FeAv
Processing volume HaIm
Processing volume BaCh
Processing volume BaGr
Processing volume DeJa
Processing volume HaEm
Processing volume IrKe
The pixel mean is : [49.815208885476984, 31.993244832911131, 22.754092016401991, 56.14204836162434, 28.979096706296833, 44.01427871196713, 30.548076199595751, 30.896113618301545, 26.562047807657102, 34.614988402825297] 
The pixel mean is : 35.6319195543 
"""

import os
import h5py
import numpy as np

data_path = 'data/hip'
split_file = os.path.join(data_path, 'train_list.txt')
h5_name = os.path.join(data_path, 'hip3d0_data.h5')

def main():
    h5_f = h5py.File(h5_name,'r') 
    pix_mean_list = []
    
    with open(split_file, 'r') as splitf:
        for vol_name in splitf:
            print("Processing volume {}".format(vol_name.strip()))
            vol = np.asarray(h5_f[vol_name.strip()])
            # sliceidx, height, width
            pix_mean_list.append(vol.mean())
            
    print("The pixel mean is : {} ".format(pix_mean_list))
    print("The pixel mean is : {} ".format(np.array(pix_mean_list).mean()))
    h5_f.close()

main()
