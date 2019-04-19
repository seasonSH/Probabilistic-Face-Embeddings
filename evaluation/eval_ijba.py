"""Extract features using pre-trained model
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
import sys
import time
import math
import argparse
import numpy as np

from utils import utils
from utils.dataset import Dataset
from utils.imageprocessing import preprocess
from network import Network


from evaluation.ijba import IJBATest


def force_compare(compare_func):
    def compare(t1, t2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = -9999
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None])   
        return score_vec
    return compare

def extract_feature(paths, num_images=None, verbose=False):
    if num_images is not None:
        num_images = min(len(paths), num_images)
        idx = np.random.permutation(len(paths))[:num_images]
        paths = paths[idx]
    embeddings, confidence = network.extract_feature(paths, args.batch_size, proc_func=proc_func, verbose=verbose)
    return np.concatenate([embeddings, confidence], axis=1)

def main(args):

    network = Network()
    network.load_model(args.model_dir)
    proc_func = lambda x: preprocess(x, network.config, False)

    testset = Dataset(args.dataset_path)
    ijbatest = IJBATest(testset['abspath'].values)
    ijbatest.init_proto(args.protocol_path)


    mu, sigma_sq = network.extract_feature(ijbatest.image_paths, args.batch_size, proc_func=proc_func, verbose=True)
    features = np.concatenate([mu, sigma_sq], axis=1)


    print('Fusing (Average) templates...')
    for t in ijbatest.verification_templates:
        if len(t.indices) > 0:
            t.feature = utils.l2_normalize(np.mean(features[t.indices], axis=0))
        else:
            t.feature = None

    print('---- Average pooling')
    TARs, FARs, threshold = ijbatest.test_verification(force_compare(utils.pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], FARs[i], threshold[i]))


    print('Fusing (PFE) templates...')
    for t in ijbatest.verification_templates:
        if len(t.indices) > 0:
            t.mu, t.sigma_sq = utils.aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
            t.feature = t.mu

    print('---- Uncertainty pooling')
    TARs, FARs, threshold = ijbatest.test_verification(force_compare(utils.pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], FARs[i], threshold[i]))


    print('Fusing (PFE) templates...')
    for t in ijbatest.verification_templates:
        if len(t.indices) > 0:
            t.feature = np.concatenate([t.mu, t.sigma_sq])
    print('---- MLS comparison')
    TARs, FARs, threshold = ijbatest.test_verification(force_compare(utils.pair_MLS_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], FARs[i], threshold[i]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, default=None)
    parser.add_argument("--dataset_path", help="The path to the IJB-A dataset directory",
                        type=str, default='data/ijba_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the IJB-A protocol directory",
                        type=str, default='proto/IJB-A')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=256)
    args = parser.parse_args()
    main(args)
