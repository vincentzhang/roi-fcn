""" Compute the statistics of the training and testing data of socket """
import os
import h5py


data_path = 'data/socket'
bbox_path = os.path.join(data_path, 'seg_band_bbox')
train_split_file = os.path.join(data_path, 'train.txt')
test_split_file = os.path.join(data_path, 'test.txt')
data_h5_f = h5py.File(os.path.join(data_path, 'seg_band.h5'), 'r')

def main():
    count_bbox()
    count_test_slices()

# check how many slices in the training set have bbox
# for socket data

def count_bbox():
    total_slice = 0
    total_bbox = 0
    with open(train_split_file, 'r') as txtfile:
        for line in txtfile:
            bbox_file = os.path.join(bbox_path, line.strip()+'_bbox.txt')
            with open(bbox_file, 'r') as slicef:
                for img in slicef:
                    total_slice += 1
                    if img.strip().split(',')[1] == '1':
                        total_bbox += 1
    print("Training:")
    print("total number of slices: {} ".format(total_slice))
    print("total number of bbox: {} ".format(total_bbox))

# Check how many slices in the testing set
def count_test_slices():
    total_slice = 0
    with open(test_split_file, 'r') as txtfile:
        for line in txtfile:
            vol = data_h5_f[line.strip()]
            total_slice += vol.shape[-1]

    print("Testing:")
    print("total number of slices: {} ".format(total_slice))


main()
