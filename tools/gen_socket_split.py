""" Generate data split of the socket volumes """
import h5py
import numpy as np
import os


RNG_SEED = 3
data_path = 'data/socket'

def gen_split():
    np.random.seed(RNG_SEED)
    # do not use random split any more 
    # because some vols are for the same patients
    # ideally we do not want put them in the same split
    #indices = np.random.permutation(50) 
    #train = indices[:30]
    #test = indices[30:]
    f = h5py.File(os.path.join(data_path, 'seg.h5'))
    names = f.keys()
    names.sort() # sort in alphebetical order
    train = [0,1]+list(range(3, 31))
    test = [2] + list(range(31, 50))
    import pdb;pdb.set_trace()
    train_name = [names[i] for i in train]
    test_name = [names[i] for i in test]

    with open(os.path.join(data_path, 'train.txt'),'w') as txtfile:
        np.savetxt(txtfile, train_name, '%s')
    with open(os.path.join(data_path, 'test.txt'),'w') as txtfile:
        np.savetxt(txtfile, test_name, '%s')


gen_split()
