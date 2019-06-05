# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import PIL
import h5py
import pdb


class hiph5(imdb):
    def __init__(self, image_set, use_empty=False, devkit_path=None):
        imdb.__init__(self, 'hiph5_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path
        # image_set = train or test
        # self._devkit_path: data/hip/
        # self._data_path  : data/hip/
        # train image: data/hip/train.txt
        # test image: data/hip/test.txt
        # bbox: data/hip/seg_bbox
        # image: data/hip/seg.h5
        # pix-label: data/socket/seg_mask.h5
        self._classes = ('background', # always index 0
                         'foreground')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._vol_names = self._get_vol_names()
        self._image_ext = '.h5'
        self._label_ext = '.h5'
        self._h5_name = 'hip3d0'
        self._imagedb_name = os.path.join(self._data_path, self._h5_name+'_data.h5')
        #self._imagedb_name = os.path.join(self._data_path, 'seg.h5')
        # handle to the hdf5 file
        self._image_h5f = h5py.File(self._imagedb_name, 'r')
        self._label_h5f = h5py.File(os.path.join(self._data_path, self._h5_name+'_label.h5'), 'r')
        #self._label_h5f = h5py.File(os.path.join(self._data_path, 'seg_mask.h5'), 'r')
        # socket specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'matlab_eval' : False,
                       'use_empty'   : False}
        # use_empty: True to use the empty slices that do not contain bboxes
        #pdb.set_trace()
        if self._image_set == 'test' or use_empty:
            self.config['use_empty'] = True

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
                'hiph5 path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    @property
    def add_label(self):
        ret = True
        return ret

    @property
    def data_path(self):
        return self._data_path

    @property
    def image_h5f(self):
        return self._image_h5f

    @property
    def label_h5f(self):
        return self._label_h5f

    def _get_vol_names(self):
        vol_file = os.path.join(self._data_path, self._image_set + '_list.txt')
        assert os.path.exists(vol_file), \
                'vol path does not exist: {}'.format(vol_file)
        with open(vol_file) as fvol:
            # list of vol/patient name
            return [x.strip() for x in fvol.readlines()]

    def _get_widths(self):
        """ overwrite the method in the base class """
        # h5: slice, height, width
        return [self._image_h5f[self._image_index[i].rsplit('_',1)[0]].shape[2] for i in xrange(self.num_images)]

    def get_size(self):
        # h5: slice, height, width
        sizes = [self._image_h5f[self._image_index[i].rsplit('_',1)[0]].shape[:0:-1]
                for i in xrange(self.num_images)]
        return sizes
    
    def get_image(self, vol_name, idx):
        im = self._image_h5f[vol_name][idx, :,:]
        return im
    
    def get_label(self, vol_name, idx):
        """ Binary label """
        label = np.asarray(self._label_h5f[vol_name][idx, :,:], dtype='uint8')
        return label

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def _load_image_set_index(self):
        """
        Load a list of indexes listed in this dataset's image set file.
        Format: volname_sliceidx
        """
        image_index = []
        for name in self._vol_names:
            bbox_file = os.path.join(self._data_path, 'bbox', 'bbox_'+name+'.txt')
            assert os.path.exists(bbox_file), \
                    'bbox path does not exist: {}'.format(bbox_file)
            with open(bbox_file) as fbbox:
                if self.config['use_empty']:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines()]
                else:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines() if x.strip().split(',')[1]!='0']

            image_index += vol_index

        return image_index

    def label_path_at(self, i):
        """
        Return the absolute path to label of image i in the image sequence.
        """
        return self._image_index[i]

    def _get_default_path(self):
        """
        Return the default path where hip is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'hip')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_hip_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_img_labels(self):
        """
        Return the database of ground-truth pixel-wise labels.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_img_labels.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_img_labels = cPickle.load(fid)
            print '{} gt img labels loaded from {}'.format(self.name, cache_file)
            return gt_img_labels

        gt_img_labels = [self._load_hip_annotation_segment(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_img_labels, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt img labels to {}'.format(cache_file)

        return gt_img_labels

    def _load_hip_annotation(self, index):
        # the index is the slice name: volume_sliceidx
        vol_name, sliceidx = index.rsplit('_',1)
        image_set_file = os.path.join(self._data_path,
                'bbox', 'bbox_'+vol_name+'.txt')
        assert os.path.exists(image_set_file), \
                'Loading hip annotations: path does not exist: {}'.format(image_set_file)

        num_objs = 1 # assume there's only one bbox in each slice
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for hip data is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        with open(image_set_file) as f: # only keep the numbers in the middle
            idx = 0
            for x in f:
                if idx != int(sliceidx):
                    # if not the slice, continue to the next 
                    idx += 1
                    continue
                text = x.strip().split(',')
                # text: vol_sliceidx, num_bbox, x1,y1,x2,y2
                if not self.config['use_empty']:
                    assert int(text[1]) != 0 # only slices with bbox should be used
                # hard-code, assume only one box
                boxes[0, :] = list(map(int, text[2:]))
                gt_classes[0] = 1 #int(text[-1])
                overlaps[0, 1] = 1.0
                seg_areas[0] = (boxes[0,2] - boxes[0,0] + 1) * (boxes[0,3]
                        - boxes[0,1] + 1)
                break # break after reading this line

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

class hip(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'hip_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path
        # image_set = train or test
        # self._devkit_path: data/hip/
        # self._data_path  : data/hip/
        # image and bbox: train_seg_img/
        # label: train_seg_label/
        self._classes = ('background', # always index 0
                         'foreground')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._label_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'hip path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    @property
    def add_label(self):
        ret = True
        return ret

    @property
    def data_path(self):
        return self._data_path

    def get_label(self, idx):
        """ In the range [0,255], convert to {0,1} """
        label = np.asarray(PIL.Image.open(self.label_path_at(idx)), dtype='uint8')
        # the jpg label contains several shades of color, need to binarize
        # binarize the label
        label = np.where(label > 127, 1, 0)
        return label

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set + '_seg_img',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load a list of indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /train_seg_img/myData_seg.txt
        image_set_file = os.path.join(self._data_path,
                self._image_set+'_seg_img', 'myData_seg.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split(',')[0].split('.')[0] for x in f.readlines()]
        return image_index

    def label_path_at(self, i):
        """
        Return the absolute path to label of image i in the image sequence.
        """
        return self._label_path_from_index(self._image_index[i])

    def _label_path_from_index(self, index):
        """
        Construct an label path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, self._image_set +
                '_seg_label', index + self._label_ext)
        assert os.path.exists(label_path), \
                'Label Path does not exist: {}'.format(label_path)
        return label_path

    def _get_default_path(self):
        """
        Return the default path where hip is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'hip')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_hip_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_img_labels(self):
        """
        Return the database of ground-truth pixel-wise labels.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_img_labels.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_img_labels = cPickle.load(fid)
            print '{} gt img labels loaded from {}'.format(self.name, cache_file)
            return gt_img_labels

        gt_img_labels = [self._load_hip_annotation_segment(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_img_labels, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt img labels to {}'.format(cache_file)

        return gt_img_labels

    def _load_hip_annotation(self, index):
        image_set_file = os.path.join(self._data_path,
                'train_seg_img/myData_seg.txt')
        assert os.path.exists(image_set_file), \
                'Loading hip annotations: path does not exist: {}'.format(image_set_file)
        num_objs = 1 # assume there's only one bbox in each slice
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for hip data is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        with open(image_set_file) as f: # only keep the numbers in the middle
            for x in f.readlines():
                text = x.strip().split(',')
                if text[0].split('.')[0] == index:
                    # hard-code, assume only one box
                    boxes[0, :] = list(map(int, text[1:-1]))
                    if boxes[0, 0] == 192:
                        boxes[0, 0] = 191
                    if boxes[0, 2] == 192:
                        boxes[0, 2] = 191
                    gt_classes[0] = 1 #int(text[-1])
                    overlaps[0, 1] = 1.0
                    seg_areas[0] = (boxes[0,2] - boxes[0,0] + 1) * (boxes[0,3]
                            - boxes[0,1] + 1)
                    break

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def evaluate_seg(self, output_dir):
        seg_file = os.path.join(output_dir, 'seg.pkl')
        with open(seg_file, 'wb') as f:
            cPickle.dump(all_seg, f, cPickle.HIGHEST_PROTOCOL)
        # flatten the images and compute the IOU

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
