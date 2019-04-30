''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'sphere64_msarcface_am_PFE'

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 100

# Training dataset path
train_dataset_path = "./data/ms_arcface"

# Target image size for the input of network
image_size = [112, 96]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    ['center_crop', (112, 96)],
    ['random_flip'],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['center_crop', (112, 96)],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# The network architecture
embedding_network = "models/sphere_net_PFE.py"

# The network architecture
uncertainty_module = "models/uncertainty_module.py"

# Number of dimensions in the embedding space
embedding_size = 512


####### TRAINING STRATEGY #######

# Base Random Seed
base_random_seed = 0

# Number of samples per batch
batch_format = {
    'size': 256,
    'num_classes': 64,
}

# Number of batches per epoch
epoch_size = 1000

# Number of epochs
num_epochs = 12

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 1e-3
learning_rate_schedule = {
    0:      1 * lr,
    8000:   0.1 * lr,
}

# Restore model
restore_model = './pretrained/sphere64_msarcface_am'

# Keywords to filter restore variables, set None for all
restore_scopes = ['SphereNet/conv', 'SphereNet/Bot']

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0


