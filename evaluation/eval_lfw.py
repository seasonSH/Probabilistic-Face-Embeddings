"""Test PFE on LFW.
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
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
import imp
import argparse
import time
import math
import numpy as np

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network
from evaluation.lfw import LFWTest


def main(args):


    paths = Dataset(args.dataset_path)['abspath']
    print('%d images to load.' % len(paths))
    assert(len(paths)>0)

    # Load model files and config file
    network = Network()
    network.load_model(args.model_dir) 
    images = preprocess(paths, network.config, False)

    # Run forward pass to calculate embeddings
    mu, sigma_sq = network.extract_feature(images, args.batch_size, verbose=True)
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
    
    lfwtest = LFWTest(paths)
    lfwtest.init_standard_proto(args.protocol_path)

    accuracy, threshold = lfwtest.test_standard_proto(mu, utils.pair_euc_score)
    print('Euclidean (cosine) accuracy: %.5f threshold: %.5f' % (accuracy, threshold))
    accuracy, threshold = lfwtest.test_standard_proto(feat_pfe, utils.pair_MLS_score)
    print('MLS accuracy: %.5f threshold: %.5f' % (accuracy, threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str)
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default='data/lfw_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the LFW protocol file",
                        type=str, default='./proto/lfw_pairs.txt')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
