import _init_paths
import os
from utils.cython_bbox import bbox_overlaps
import numpy as np
import pdb

data_path = '../data/socket'

def get_vol_names(split, vol=None):
    if vol:
        print("vol is: ", vol)
        return [vol]
    vol_file = os.path.join(data_path, split + '.txt')
    assert os.path.exists(vol_file), \
            'vol path does not exist: {}'.format(vol_file)
    with open(vol_file) as fvol:
        # list of vol/patient name
        return [x.strip() for x in fvol.readlines()]

def load_image_set_index(vol_names, use_empty=False):
    """
    Load a list of indexes listed in this dataset's image set file.
    Format: volname_sliceidx
    """
    image_index = []
    for name in vol_names:
        bbox_file = os.path.join(data_path, 'cropped_seg_band_bbox', name + '_bbox.txt')
        assert os.path.exists(bbox_file), \
                'bbox path does not exist: {}'.format(bbox_file)
        with open(bbox_file) as fbbox:
            if use_empty:
                vol_index = [x.strip().split(',')[0] for x in fbbox.readlines()]
            else:
                vol_index = [x.strip().split(',')[0] for x in fbbox.readlines() if x.strip().split(',')[1]!='0']
        image_index += vol_index
    return image_index

def read_bbox(index, keep, use_empty=False):
    """ Docstring

    Args:
        index: index of an image in this format
                volname_sliceidx
        keep:  the number of bbox to keep
        use_empty(optinal): whether to use slices that do not contain positives

    Returns:
        iou: the iou of predicted bbox and gt bbox
    """

    # indices: list of [vol_sliceidx]
    vol_name, sliceidx = index.rsplit('_',1)
    image_set_file = os.path.join(data_path,
            'bbox_pred', vol_name + '.txt')
    gt_file = os.path.join(data_path,
            'cropped_seg_band_bbox', vol_name + '_bbox.txt')
    assert os.path.exists(image_set_file), \
            'Loading socket prediction: path does not exist: {}'.format(image_set_file)
    assert os.path.exists(gt_file),\
            'Loading socket gt : path does not exist: {}'.format(gt_file)
    #print("Image Set File is {}".format(image_set_file))
    with open(image_set_file) as f: # only keep the numbers in the middle
        for x in f:
            # idx is the parsed image index
            idx = x.strip().split(',',1)[0]
            if idx != index:
                # if not the slice, continue to the next
                #print("idx: {}; index: {}".format(idx, index))
                continue
            text = x.strip().split(',')
            # text: vol_sliceidx, num_bbox, confidence, x1,y1,x2,y2
            if not use_empty:
                assert int(text[1]) != 0 # only slices with bbox should be used
            num_box = int(text[1])
            pred_boxes = np.zeros((num_box, 4), dtype=float)
            for i in range(num_box):
                pred_boxes[i,:] = np.array(list(map(float, text[5*i+3:5*i+7])))
            #pred_boxes = list(map(float, text[3:]))
            break # break after reading this line
    with open(gt_file) as gt_f:
        for x in gt_f:
            idx = x.strip().split(',',1)[0]
            if idx != index:
                # if not the slice, continue to the next
                #print("idx: {}; index: {}".format(idx, index))
                continue
            text = x.strip().split(',')
            # text: vol_sliceidx, num_bbox, x1,y1,x2,y2
            if not use_empty:
                assert int(text[1]) != 0 # only slices with bbox should be used
            gt_boxes = np.asarray(list(map(float, text[2:]))).reshape(1,4)
            break # break after reading this line
    #iou = compute_iou(gt_boxes, pred_boxes[:1,:]/2.7247956403269753)
    #import pdb;pdb.set_trace()
    iou = compute_iou(gt_boxes, pred_boxes[:keep,:])
    #2.72479564
    #return pred_boxes, gt_boxes
    return iou

def compute_iou(gt, pred):
    # both numpy array
    # gt:[x1,y1,x2,y2]
    # pred: same, num_img x 4
    #print("shape of gt", gt.shape)
    #print("shape of pred", pred.shape)
    mask_pred = mask_from_box(pred)
    mask_gt = mask_from_box(gt)
    I = mask_pred & mask_gt # IOU between pred and gt
    #U = mask_pred | mask_gt # union
    #val = mask_gt # to get the IOU between gt and the img
    I = mask_pred # to get the IOU between pred and the img
    #val = val.sum()
    I = I.sum()
    #U = U.sum()
    # for real IOU
    #val = float(I)/float(U)
    # for gt bbox
    #val = float(I)/float(mask_gt.sum())

    # for full image
    val = float(I)/(367.*192.)
    #for idx in xrange(gt.shape[0]):
    #    print("idx {}/{} ".format(idx, gt.shape[0]))
    # overlaps: (rois x gt_boxes)
    #overlaps = bbox_overlaps(
    #    np.ascontiguousarray(pred, dtype=np.float),
    #    np.ascontiguousarray(gt, dtype=np.float))
    #val = overlaps.sum()
    #val = overlaps[0,0] # top pick
    # mask from boxes
    #val =  overlaps.diagonal()
    return val

def mask_from_box(boxes):
    """ Return a binary mask of the same size as the image

    Args:
        boxes: num_boxes x 4

    Returns:
        mask: same size as the image
    """
    mask = np.zeros((367,192),dtype=bool)
    for box in boxes:
        mask[int(box[1]):int(box[3]+0.5)+1,int(box[0]):int(box[2]+0.5)+1]= True
    return mask

def get_iou():
    #vol_names = get_vol_names('test', 'Me6401IM_0063')
    # This loads all the volumes available
    #pdb.set_trace()
    #vol_names = get_vol_names('train')
    vol_names = get_vol_names('test')
    img_idx = load_image_set_index(vol_names)
    # 1, 10, 20, 30, ... 300
    # a list of num_box, each representing the number of boxes to sample
    num_list = [1] + [10 * i for i in list(range(1,31))]
    mean_list = []
    for num_box in num_list:
        all_pred = []
        all_gt = []
        all_iou = []
        for i, img_index in enumerate(img_idx):
            #print("iter: {}".format(i))
            # both np array: num x 4
            #pred_box, gt_box = read_bbox(img_index)
            iou = read_bbox(img_index, num_box)
            #all_pred.append(pred_box)
            #all_gt.append(gt_box)
            all_iou.append(iou)
        #all_pred = np.asarray(all_pred)
        #all_gt = np.asarray(all_gt)
        #np.save('pred',all_pred)
        #np.save('gt',all_gt)
        #print("Total number of img_idx", len(img_idx))
        #iou = compute_iou(all_gt, all_pred)
        all_iou = np.asarray(all_iou)
        print("mean iou of {} box: {}".format(num_box, all_iou.mean()))
        mean_list.append(all_iou.mean())
    #np.save('mean_iou',mean_list)
    #np.save('mean_iou_with_img',mean_list)

get_iou()
