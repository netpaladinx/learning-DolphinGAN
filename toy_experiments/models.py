from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import toy_experiments.samplers as samp
import toy_experiments.visualizer as vis

class GAAN(object):
    def __init__(self, model_name="GAAN",
                 dim_z=256,
                 dim_hidden=128,
                 alpha=1.0,
                 beta=1.0,
                 batch_size=512,
                 learning_rate=0.0002,
                 max_epochs=100000,
                 display_freq=10,
                 print_freq=10,
                 output_dir='output',
                 sampler=samp.sampler2,
                 random_seed=1234):  # for consistency of each trail
        self.model_name = model_name
        self.dim_z = dim_z
        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.beta = beta
        self.max_epochs = max_epochs
        self.display_freq = display_freq
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.sampler = sampler
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.replay_buffer = deque(maxlen=10000)

        self.tf_graph = tf.Graph()
        self._set_random_seed()
        self._build_model()
        self.tf_session = tf.Session(graph=self.tf_graph)

    def _set_random_seed(self):
        np.random.seed(self.random_seed)
        with self.tf_graph.as_default():
            tf.set_random_seed(self.random_seed)

    def _create_inputs(self):
        self.x_pl = tf.placeholder(tf.float32, shape=[None, 2], name='data')
        self.g_old_pl = tf.placeholder(tf.float32, shape=[None, 2], name='g_old')
        self.z_pl = tf.placeholder(tf.float32, shape=[None, self.dim_z], name='z')

    def _create_generator(self):
        hidden1 = slim.fully_connected(self.z_pl, self.dim_hidden, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                       scope='g_hidden1')
        hidden2 = slim.fully_connected(hidden1, self.dim_hidden, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                       scope='g_hidden2')
        out = slim.fully_connected(hidden2, 2, activation_fn=None, scope='g_out')
        return out

    def _create_discriminator(self, inputs):
        hidden = slim.fully_connected(inputs, self.dim_hidden, activation_fn=tf.nn.relu,
                                      weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                      scope='d_hidden')

        # using softplus activation from D2GAN
        # out = slim.fully_connected(hidden, self.dim_hidden, activation_fn=tf.nn.softplus,
        #                           weights_initializer=tf.random_normal_initializer(stddev=0.01),
        #                           scope='d_out')

        out = slim.fully_connected(hidden, self.dim_hidden, activation_fn=None,
                                   weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                   scope='d_out')
        return out

    def _create_optimizer(self, loss, var_list):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(loss, var_list=var_list)

    def _build_model(self):
        with self.tf_graph.as_default():
            self._create_inputs()

            with tf.variable_scope('generator'):
                self.g_new = self._create_generator()

            with tf.variable_scope('discriminator_pa') as scope:
                self.d_pa_g_new = self._create_discriminator(self.g_new)
                scope.reuse_variables()
                self.d_pa_x = self._create_discriminator(self.x_pl)

            with tf.variable_scope('discriminator_ma') as scope:
                self.d_ma_g_new = self._create_discriminator(self.g_new)
                scope.reuse_variables()
                self.d_ma_g_old = self._create_discriminator(self.g_old_pl)
                self.d_ma_x = self._create_discriminator(self.x_pl)

            # using losses from D2GAN
            # self.loss_d_pa = tf.reduce_mean(- self.alpha * tf.log(self.d_pa_x) + self.d_pa_g_new)
            # self.loss_d_ma = tf.reduce_mean(- self.beta * tf.log(self.d_ma_g_new) + self.d_ma_g_old)
            # self.loss_g = tf.reduce_mean(- self.d_pa_g_new - self.beta * tf.log(self.d_ma_g_new))

            self.loss_d_pa = tf.reduce_mean(tf.nn.softplus(-self.d_pa_x)
                                            - tf.nn.softplus(-self.d_pa_g_new))
            self.loss_d_ma = tf.reduce_mean(tf.nn.softplus(-self.d_ma_g_new)
                                            - tf.nn.softplus(-self.d_ma_x))
            self.loss_g = tf.reduce_mean(tf.nn.softplus(-self.d_pa_g_new)
                                         - tf.nn.softplus(-self.d_ma_g_new))

            # using traditional GAN
            #self.loss_d_pa = tf.reduce_mean(tf.nn.softplus(-self.d_pa_x) + tf.nn.softplus(-self.d_pa_g_new) + self.d_pa_g_new)
            #self.loss_d_ma = tf.reduce_mean(- tf.nn.softplus(-(self.d_ma_g_new * self.d_pa_g_new)) + tf.nn.softplus(-self.d_ma_g_old) + self.d_ma_g_old)
                            # + slim.losses.get_regularization_losses(scope='discriminator_ma')[0]
            #self.loss_g = tf.reduce_mean(- tf.nn.softplus(-self.d_pa_g_new) - self.d_pa_g_new - tf.nn.softplus(-(self.d_ma_g_new * self.d_pa_g_new)))

            """
            self.loss_d_pa = tf.reduce_mean(tf.nn.softplus(-self.d_pa_x) +
                                            tf.nn.softplus(self.d_pa_g_new))
            self.loss_d_ma = tf.reduce_mean(tf.nn.softplus(-self.d_ma_g_new) +
                                            tf.nn.softplus(-self.d_pa_g_new) +
                                            tf.nn.softplus(self.d_ma_g_old))
            self.loss_g = tf.reduce_mean(tf.nn.softplus(-self.d_ma_g_new) +
                                         tf.nn.softplus(-self.d_pa_g_new))
            """
            """
            self.loss_d_pa = tf.reduce_mean(tf.nn.softplus(-self.d_pa_x) +
                                            -tf.nn.softplus(-self.d_pa_g_new))
            self.loss_d_ma = tf.reduce_mean(tf.nn.softplus(-self.d_ma_g_new) +
                                            -tf.nn.softplus(-self.d_ma_g_old))
            self.loss_g = tf.reduce_mean(tf.nn.softplus(-self.d_ma_g_new) +
                                         tf.nn.softplus(-self.d_pa_g_new))
            """
            self.params_d_pa = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_pa')
            self.params_d_ma = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_ma')
            self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            self.opt_d_pa = self._create_optimizer(self.loss_d_pa, self.params_d_pa)
            self.opt_d_ma = self._create_optimizer(self.loss_d_ma, self.params_d_ma)
            self.opt_g = self._create_optimizer(self.loss_g, self.params_g)

            self.init_op = tf.global_variables_initializer()
            self.init_op_d_pa = tf.variables_initializer(self.params_d_pa)

    def _generate(self, n_samples=1000):
        z = np.random.normal(0.0, 1.0, [n_samples, self.dim_z])
        return self.tf_session.run(self.g_new, feed_dict={self.z_pl: z})

    def _display(self, epoch, n_samples=1000):
        x = self.sampler(n_samples)
        g = self._generate(n_samples)
        fig, ax, _ = vis.display_scatter(x, g)
        fig.tight_layout()
        fig.savefig("{}/output{}.png".format(self.output_dir, epoch))

    def fit(self):
        self.tf_session.run(self.init_op)

        epoch = 0
        while epoch < self.max_epochs:
            for i in range(10):
                x = self.sampler(self.batch_size)
                z = np.random.normal(0.0, 1.0, [self.batch_size, self.dim_z])
                d_pa_x, d_pa_g_new, loss_d_pa, _ = self.tf_session.run(
                    [self.d_pa_x, self.d_pa_g_new, self.loss_d_pa, self.opt_d_pa],
                    feed_dict={self.x_pl: x, self.z_pl: z})

            if len(self.replay_buffer) >= self.batch_size:
                g_old = random.sample(self.replay_buffer, self.batch_size)
                z = np.random.normal(0.0, 1.0, [self.batch_size, self.dim_z])
                x = self.sampler(self.batch_size)
                d_ma_g_new, d_ma_g_old, loss_d_ma, _ = self.tf_session.run(
                    [self.d_ma_g_new, self.d_ma_g_old, self.loss_d_ma, self.opt_d_ma],
                    feed_dict={self.g_old_pl: g_old, self.z_pl: z, self.x_pl: x})
            else:
                loss_d_ma = 0.

            z = np.random.normal(0.0, 1.0, [self.batch_size, self.dim_z])
            loss_g, _ = self.tf_session.run([self.loss_g, self.opt_g], feed_dict={self.z_pl: z})

            z = np.random.normal(0.0, 1.0, [1, self.dim_z])
            g_new = self.tf_session.run(self.g_new, feed_dict={self.z_pl: z})
            self.replay_buffer.append(g_new[0])

            epoch += 1

            if epoch % self.print_freq == 0:
                print("epoch: %d/%d | loss_d_pa: %.8f | loss_d_ma: %.8f | loss_g: %.8f" %
                      (epoch, self.max_epochs, loss_d_pa, loss_d_ma, loss_g))

            if epoch % self.display_freq == 0:
                self._display(epoch)

