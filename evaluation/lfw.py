"""Test protocols on LFW dataset
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import scipy.io as sio
from evaluation import metrics
from collections import namedtuple

StandardFold = namedtuple('StandardFold', ['indices1', 'indices2', 'labels'])
BLUFRFold = namedtuple('BLUFRFold', ['train_indices', 'test_indices', 'probe_indices', 'gallery_indices'])

class LFWTest:
    def __init__(self, image_paths):
        self.image_paths = np.array(image_paths).astype(np.object).flatten()
        self.images = None
        self.labels = None
        self.standard_folds = None
        self.blufr_folds = None
        self.queue_idx = None

    def init_standard_proto(self, lfw_pairs_file):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            index_dict[image_name] = i

        pairs = []
        with open(lfw_pairs_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        # 10 folds
        self.standard_folds = []
        for i in range(10):
            indices1 = np.zeros(600, dtype=np.int32)
            indices2 = np.zeros(600, dtype=np.int32)
            labels = np.array([True]*300+[False]*300, dtype=np.bool)
            # 300 positive pairs, 300 negative pairs in order
            for j in range(600):
                pair = pairs[600*i+j]
                if j < 300:
                    assert len(pair) == 3
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[0] + '_' + '%04d' % int(pair[2])
                else:
                    assert len(pair) == 4
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[2] + '_' + '%04d' % int(pair[3])                
                indices1[j] = index_dict[img1]
                indices2[j] = index_dict[img2]
            fold = StandardFold(indices1, indices2, labels)
            self.standard_folds.append(fold)

    def test_standard_proto(self, features, compare_func):

        assert self.standard_folds is not None
        
        accuracies = np.zeros(10, dtype=np.float32)
        thresholds = np.zeros(10, dtype=np.float32)

        features1 = []
        features2 = []

        for i in range(10):
            # Training
            train_indices1 = np.concatenate([self.standard_folds[j].indices1 for j in range(10) if j!=i])
            train_indices2 = np.concatenate([self.standard_folds[j].indices2 for j in range(10) if j!=i])
            train_labels = np.concatenate([self.standard_folds[j].labels for j in range(10) if j!=i])

            train_features1 = features[train_indices1,:]
            train_features2 = features[train_indices2,:]
            
            train_score = compare_func(train_features1, train_features2)
            _, thresholds[i] = metrics.accuracy(train_score, train_labels)

            # Testing
            fold = self.standard_folds[i]
            test_features1 = features[fold.indices1,:]
            test_features2 = features[fold.indices2,:]
            
            test_score = compare_func(test_features1, test_features2)
            accuracies[i], _ = metrics.accuracy(test_score, fold.labels, np.array([thresholds[i]]))

        accuracy = np.mean(accuracies)
        threshold = - np.mean(thresholds)
        return accuracy, threshold

    def test_roc_cmc(self, features, labels, FAR=None, rank=None):
        labels = np.array(labels)
        score_mat = - facepy.metric.euclidean(features, features)

        if rank is not None:
            label_mat = labels.reshape((-1,1)) == labels.reshape((1,-1))
            CMCs = facepy.evaluation.CMC(score_mat, label_mat, ranks=[rank])
            CMC = CMCs[0]
        else:
            CMC = 0.0
        if FAR is not None:
            score_vec, label_vec = get_pairwise_score_label(score_mat, labels)
            TARs, FARs, thresholds = facepy.evaluation.ROC(score_vec, label_vec, FARs=[FAR])
            TAR, FAR, threshold = (TARs[0], FARs[0], thresholds[0])
        else:
            TAR, FAR, threshold = (0., 0., 0.)

        return TAR, FAR, CMC


    def generate_labels(self):
        labels = []
        label_dict = {}
        next_label = 0
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            identity_name = '_'.join(image_name.split('_')[:-1])
            if not identity_name in label_dict: 
                label_dict[identity_name] = next_label
                next_label += 1
            labels.append(label_dict[identity_name])

        self.labels = np.array(labels)


    def init_blufr_proto(self, blufr_config_file):

        config = sio.loadmat(blufr_config_file)

        # Build an dictionary mapping image names to indices in self.image_paths
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            index_dict[image_name] = i

        # Build an array to map the blufr indices to indices in self.image_paths
        index_mapping = np.ndarray(len(self.image_paths) ,dtype=np.int32)
        for i, image_path in enumerate(list(config['imageList'])):
            image_path = str(image_path[0][0])
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            index_mapping[i] = index_dict[image_name]
        
        # Set up the 10 folds using the indices in self.image_paths
        self.blufr_folds = []
        for i in range(10):
            train_indices = index_mapping[config['trainIndex'][i][0].flatten()-1]
            test_indices = index_mapping[config['testIndex'][i][0].flatten()-1]
            probe_indices = index_mapping[config['probIndex'][i][0].flatten()-1]
            gallery_indices = index_mapping[config['galIndex'][i][0].flatten()-1]
            self.blufr_folds.append(BLUFRFold(train_indices, test_indices, probe_indices, gallery_indices))

        if self.labels is None:
            self.generate_labels()


    def test_blufr_all_folds(self, features, use_euclidean=True, subtract_std=True, sigma_sq=None):

        if use_euclidean:
            score_mat = - facepy.metric.euclidean(features,features)
        else:
            score_mat = np.dot(features, features.T)
        label_mat = self.labels.reshape([-1,1]) == self.labels.reshape([1,-1])


        confident_threshold = np.median(sigma_sq.sum(axis=1))
        confident_idx = sigma_sq.sum(axis=1) < confident_threshold

        VRs = np.zeros(10)
        DIRs = np.zeros(10)
        for i in range(10):
            fold = self.blufr_folds[i]


            test_indices = np.arange(features.shape[0])
            test_indices[fold.test_indices] = True
            test_indices = np.logical_and(test_indices, confident_idx)
            VR, FAR, threshold = facepy.evaluation.ROC_by_mat(score_mat[test_indices][:,test_indices], 
                                                label_mat[test_indices][:,test_indices], FARs=[0.001], triu_k=1)
            VRs[i] = VR[0]            
            DIR, FAR, threshold = facepy.evaluation.DIR_FAR(score_mat[fold.probe_indices][:,fold.gallery_indices],
                                                label_mat[fold.probe_indices][:,fold.gallery_indices], FARs=[0.01])
            DIRs[i] = DIR[0]        
        if subtract_std:
            VR_mean = VRs.mean() - VRs.std()
            DIR_mean = DIRs.mean() - DIRs.std()
            return VR_mean, DIR_mean
        else:
           return VRs.mean(), VRs.std(), DIRs.mean(), DIRs.std()
                
               
    def test_blufr_fold(self, fold, features, use_euclidean=True):
        if use_euclidean:
            score_mat = - facepy.metric.euclidean(features,features)
            # simga_sq_sum = simga_sq.reshape([-1, 1]) + simga_sq.reshape([1, -1])
            # score_mat = - (score_mat / simga_sq_sum + np.log(simga_sq_sum))
        else:
            score_mat = np.dot(features, features.T)
        label_mat = self.labels.reshape([-1,1]) == self.labels.reshape([1,-1])

        fold = self.blufr_folds[fold]
        VR, FAR, threshold = facepy.evaluation.ROC_by_mat(score_mat[fold.test_indices][:,fold.test_indices], 
                                            label_mat[fold.test_indices][:,fold.test_indices], FARs=[0.001], triu_k=1)
        DIR, FAR, threshold = facepy.evaluation.DIR_FAR(score_mat[fold.probe_indices][:,fold.gallery_indices],
                                            label_mat[fold.probe_indices][:,fold.gallery_indices], FARs=[0.01])

        return VR[0], DIR[0]

