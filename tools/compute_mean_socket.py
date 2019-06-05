""" Compute the mean of the grayscale image in the training set of socket

Output: 
Processing volume Ad5501IM_0028
Processing volume Ad5501IM_0029
Processing volume Be6901IM_0026
Processing volume Be9101IM_0026
Processing volume Be9101IM_0027
Processing volume Bl1001IM_0023
Processing volume Bl1001IM_0025
Processing volume Br6701IM_0027
Processing volume De4601IM_0019
Processing volume Gr4701IM_0021
Processing volume Gr4701IM_0022
Processing volume Gw9301IM_0027
Processing volume Gw9301IM_0032
Processing volume He2001IM_0037
Processing volume He2001IM_0039
Processing volume Ja7501IM_0022
Processing volume Ka6901IM_0020
Processing volume Ka6901IM_0022
Processing volume Ki9501IM_0030
Processing volume Ki9501IM_0032
Processing volume Kn4501IM_0022
Processing volume Kn4501IM_0024
Processing volume Le6001IM_0022
Processing volume Le6001IM_0025
Processing volume Le9701IM_0033
Processing volume Le9701IM_0034
Processing volume Lo6702IM_0023
Processing volume Lo6702IM_0025
Processing volume Lu6001IM_0022
Processing volume Lu6001IM_0023
The pixel mean is : [37.333671862511991, 36.943108799200374, 52.910509987812922, 25.032955914405505, 29.881143997595643, 23.477367820642183, 30.448265550488198, 31.985812318091298, 32.47499431795476, 28.458406648367568, 25.152639195538949, 35.329244602265156, 36.874051951792943, 21.91571985447397, 20.477171758861491, 47.345996749370606, 29.404061163782767, 27.800280636838636, 31.177339961969842, 31.188246158687662, 34.000851994585197, 33.427432945622201, 50.662497812040627, 50.203105347966719, 26.772025157931719, 25.418646744958174, 53.011465613190452, 54.422721268016957, 30.491006134111252, 29.97350709985081] 
The pixel mean is : 34.1331416456
"""

import os
import h5py
import numpy as np

data_path = 'data/socket'
split_file = os.path.join(data_path, 'train.txt')
h5_name = os.path.join(data_path, 'seg_band.h5')

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
