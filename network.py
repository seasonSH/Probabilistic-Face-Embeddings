"""Main implementation class of PFE
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
import time

import numpy as np
import tensorflow as tf
from utils.tflib import mutual_likelihood_score_loss

class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes=None):
        '''
            Initialize the graph from scratch according to config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images')
                self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # Initialialize the backbone network
                network = imp.load_source('embedding_network', config.embedding_network)
                mu, conv_final = network.inference(self.images, config.embedding_size)

                # Initialize the uncertainty module
                uncertainty_module = imp.load_source('uncertainty_module', config.uncertainty_module)
                log_sigma_sq = uncertainty_module.inference(conv_final, config.embedding_size, 
                                        phase_train = self.phase_train, weight_decay = config.weight_decay,
                                        scope='UncertaintyModule')

                self.mu = tf.identity(mu, name='mu')
                self.sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')

                # Build all losses
                loss_list = []
                self.watch_list = {}

               
                MLS_loss = mutual_likelihood_score_loss(self.labels, mu, log_sigma_sq)
                loss_list.append(MLS_loss)
                self.watch_list['loss'] = MLS_loss


                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                loss_list.append(reg_loss)
                self.watch_list['reg_loss'] = reg_loss


                total_loss = tf.add_n(loss_list, name='total_loss')
                grads = tf.gradients(total_loss, self.trainable_variables)


                # Training Operaters
                train_ops = []

                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                apply_gradient_op = opt.apply_gradients(list(zip(grads, self.trainable_variables)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Collect TF summary
                for k,v in self.watch_list.items():
                    tf.summary.scalar('losses/' + k, v)
                tf.summary.scalar('learning_rate', self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=99)
 
        return

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='UncertaintyModule')

    def save_model(self, model_dir, global_step):
        with self.sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...')
            self.saver.save(self.sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                self.saver.export_meta_graph(metagraph_path)

    def restore_model(self, model_dir, restore_scopes=None):
        var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with self.sess.graph.as_default():
            if restore_scopes is not None:
                var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
            model_dir = os.path.expanduser(model_dir)
            ckpt_file = tf.train.latest_checkpoint(model_dir)

            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            self.images = self.graph.get_tensor_by_name('images:0')
            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.mu = self.graph.get_tensor_by_name('mu:0')
            self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))



    def train(self, images_batch, labels_batch, learning_rate, keep_prob):
        feed_dict = {   self.images: images_batch,
                        self.labels: labels_batch,
                        self.learning_rate: learning_rate,
                        self.keep_prob: keep_prob,
                        self.phase_train: True,}
        _, wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict = feed_dict)

        step = self.sess.run(self.global_step)

        return wl, sm, step

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = len(images)
        num_features = self.mu.shape[1]
        mu = np.ndarray((num_images, num_features), dtype=np.float32)
        sigma_sq = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            images_batch = images[start_idx:end_idx]
            if proc_func:
                images_batch = proc_func(images_batch)
            feed_dict = {self.images: images_batch,
                        self.phase_train: False,
                    self.keep_prob: 1.0}
            mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq], feed_dict=feed_dict)
        if verbose:
            print('')
        return mu, sigma_sq


