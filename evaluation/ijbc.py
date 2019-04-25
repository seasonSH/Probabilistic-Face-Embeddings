"""
Main file for evaluation on IJB-A and IJB-B protocols.
More instructions can be found in README.md file.
2017 Yichun Shi
"""

import sys
import os
import numpy as np
import utils

from evaluation import metrics
from collections import namedtuple


# Configuration
VerificationFold = namedtuple('VerificationFold', ['train_indices', 'test_indices', 'train_templates', 'templates1','templates2'])

class Template:
    def __init__(self, template_id, label, indices, medias):
        self.template_id = template_id
        self.label = label
        self.indices = np.array(indices)
        self.medias = np.array(medias)
        

def build_subject_dict(image_list):
    subject_dict = {}
    for i, line in enumerate(image_list):
        subject_id, image = tuple(line.split('/')[-2:])
        if subject_id == 'NaN': continue
        subject_id = int(subject_id)
        image, _ = os.path.splitext(image)
        image = image.replace('_','/',1) # Recover filenames 
        if not subject_id in subject_dict:
            subject_dict[subject_id] = {}
        subject_dict[subject_id][image] = i
    return subject_dict

def build_templates(subject_dict, meta_file):
    with open(meta_file, 'r') as f:
        meta_list = f.readlines()
        meta_list = [x.split('\n')[0] for x in meta_list]
        meta_list = meta_list[1:]

    templates = []
    template_id = None
    template_label = None
    template_indices = None
    template_medias = None
    count = 0
    for line in meta_list:
        temp_id, subject_id, image, media = tuple(line.split(',')[0:4])
        temp_id = int(temp_id)
        subject_id = int(subject_id)
        image, _ = os.path.splitext(image)
        if subject_id in subject_dict and image in subject_dict[subject_id]:
            index = subject_dict[subject_id][image]
            count += 1
        else:
            index = None

        if temp_id != template_id:
            if template_id is not None:
                templates.append(Template(template_id, template_label, template_indices, template_medias))
            template_id = temp_id
            template_label = subject_id
            template_indices = []
            template_medias = []

        if index is not None:
            template_indices.append(index)        
            template_medias.append(media)        

    # last template
    templates.append(Template(template_id, template_label, template_indices, template_medias))
    return templates

def read_pairs(pair_file):
    with open(pair_file, 'r') as f:
        pairs = f.readlines()
        pairs = [x.split('\n')[0] for x in pairs]
        pairs = [pair.split(',') for pair in pairs]
        pairs = [(int(pair[0]), int(pair[1])) for pair in pairs]
    return pairs

class IJBCTest:

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.subject_dict = build_subject_dict(image_paths)
        self.verification_folds = None
        self.verification_templates = None
        self.verification_G1_templates = None
        self.verification_G2_templates = None

    def init_verification_proto(self, protofolder):
        self.verification_folds = []
        self.verification_templates = []

        meta_gallery1 = os.path.join(protofolder,'ijbc_1N_gallery_G1.csv')
        meta_gallery2 = os.path.join(protofolder,'ijbc_1N_gallery_G2.csv')
        meta_probe = os.path.join(protofolder,'ijbc_1N_probe_mixed.csv')
        pair_file = os.path.join(protofolder,'ijbc_11_G1_G2_matches.csv')

        gallery_templates = build_templates(self.subject_dict, meta_gallery1)
        gallery_templates.extend(build_templates(self.subject_dict, meta_gallery2))
        gallery_templates.extend(build_templates(self.subject_dict, meta_probe))

        # Build pairs
        template_dict = {}
        for t in gallery_templates:
            template_dict[t.template_id] = t
        pairs = read_pairs(pair_file)
        self.verification_G1_templates = []
        self.verification_G2_templates = []
        for p in pairs:
            self.verification_G1_templates.append(template_dict[p[0]])
            self.verification_G2_templates.append(template_dict[p[1]])

        self.verification_G1_templates = np.array(self.verification_G1_templates, dtype=np.object)
        self.verification_G2_templates = np.array(self.verification_G2_templates, dtype=np.object)
    
        self.verification_templates = np.concatenate([
            self.verification_G1_templates, self.verification_G2_templates])
        print('{} templates are initialized.'.format(len(self.verification_templates)))


    def init_proto(self, protofolder):
        self.init_verification_proto(protofolder)

    def test_verification(self, compare_func, FARs=None):

        FARs = [1e-5, 1e-4, 1e-3, 1e-2] if FARs is None else FARs

        templates1 = self.verification_G1_templates
        templates2 = self.verification_G2_templates

        features1 = [t.feature for t in templates1]
        features2 = [t.feature for t in templates2]
        labels1 = np.array([t.label for t in templates1])
        labels2 = np.array([t.label for t in templates2])

        score_vec = compare_func(features1, features2)
        label_vec = labels1 == labels2

        tars, fars, thresholds = metrics.ROC(score_vec, label_vec, FARs=FARs)
        
        # There is no std for IJB-C
        std = [0. for t in tars]

        return tars, std, fars

