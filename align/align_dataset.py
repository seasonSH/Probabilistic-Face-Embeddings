"""Align face images given landmarks."""

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from align.matlab_cp2tform import get_similarity_transform_for_cv2

import numpy as np
from scipy import misc
import sys
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt

def align(src_img, src_pts, ref_pts, image_size, scale=1.0, transpose_input=False):
    w, h = image_size = tuple(image_size)

    # Actual offset = new center - old center (scaled)
    scale_ = max(w,h) * scale
    cx_ref = cy_ref = 0.
    offset_x = 0.5 * w - cx_ref * scale_
    offset_y = 0.5 * h - cy_ref * scale_

    s = np.array(src_pts).astype(np.float32).reshape([-1,2])
    r = np.array(ref_pts).astype(np.float32) * scale_ + np.array([[offset_x, offset_y]])
    if transpose_input: 
        s = s.reshape([2,-1]).T

    tfm = get_similarity_transform_for_cv2(s, r)
    dst_img = cv2.warpAffine(src_img, tfm, image_size)

    s_new = np.concatenate([s.reshape([2,-1]), np.ones((1, s.shape[0]))])
    s_new = np.matmul(tfm, s_new)
    s_new = s_new.reshape([-1]) if transpose_input else s_new.T.reshape([-1]) 
    tfm = tfm.reshape([-1])
    return dst_img, s_new, tfm


def main(args):
    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    ref_pts = np.array( [[ -1.58083929e-01, -3.84258929e-02],
                         [  1.56533929e-01, -4.01660714e-02],
                         [  2.25000000e-04,  1.40505357e-01],
                         [ -1.29024107e-01,  3.24691964e-01],
                         [  1.31516964e-01,  3.23250893e-01]])

    for i,line in enumerate(lines):
        line = line.strip()
        items = line.split()
        img_path = items[0]
        src_pts = [float(item) for item in items[1:]]

        # Transform
        if args.prefix:
            img_path = os.path.join(args.prefix, img_path)
        img = misc.imread(img_path)
        img_new, new_pts, tfm = align(img, src_pts, ref_pts, args.image_size, args.scale, args.transpose_input)

        # Visulize
        if args.visualize:
            plt.imshow(img_new)
            plt.show()
               

        # Output
        if args.output_dir:
            file_name = os.path.basename(img_path)
            sub_dir = [d for d in img_path.split('/') if d!='']
            sub_dir = '/'.join(sub_dir[-args.dir_depth:-1])
            dir_path = os.path.join(args.output_dir, sub_dir)
                
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            img_path_new = os.path.join(dir_path, file_name)
            misc.imsave(img_path_new, img_new)
            if i % 100==0:
                print(img_path_new)


    return


        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='A list file of image paths and landmarks.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.', default=None)
    parser.add_argument('--prefix', type=str, help='The prefix of the image files in the input_file.', default=None)
    parser.add_argument('--image_size', type=int, nargs=2,
        help='Image size (height, width) in pixels.', default=[112, 112])
    parser.add_argument('--scale', type=float,
        help='Scale the face size in the target image.', default=1.0)
    parser.add_argument('--dir_depth', type=int,
        help='When writing into new directory, how many layers of the dir tree should be kept.', default=2)
    parser.add_argument('--transpose_input', action='store_true',
        help='Set true if the input landmarks is in the format x1 x2 ... y1 y2 ...')
    parser.add_argument('--visualize', action='store_true',
        help='Visualize the aligned images.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
